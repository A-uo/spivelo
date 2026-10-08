"""Module-aware hierarchical Bayesian RNA kinetics model (Pyro, non-VAE).

This implementation is aligned to cell2fate-style choices for:
1. positive LogNormal cell time with a fixed numerical time scale
2. ordered module switch times from cumulative deltas
3. gene-level splicing/degradation rates
4. module-gene soft loadings (`g_fg`, `A_mgON`)
5. detection decomposition (`detection_y_c`, `detection_y_i`, `detection_y_gi`, additive term)
6. overdispersion through `stochastic_v_ag` and `GammaPoisson` likelihood

Design split:
- shared module kinetics carry temporal/state identity
- cell-type residuals are centered small corrections (amplitude + phase shift)
- optional paired equilibrium baselines; disabled by default
- full-graph edge regularization, three smooth state memberships

This is a regularized approximate Bayesian estimator, not a calibrated fate or migration model.
"""

from __future__ import annotations

from typing import Optional
import math
import warnings

import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro.nn import PyroModule
from ._kinetics import propagate, soft_states
from scvi import REGISTRY_KEYS as SCVI_REGISTRY_KEYS


def _gamma_ab_from_mean_sd(mean: torch.Tensor, sd: torch.Tensor):
    var = sd.pow(2).clamp_min(1e-8)
    alpha = (mean.pow(2) / var).clamp_min(1e-4)
    beta = (mean / var).clamp_min(1e-4)
    return alpha, beta


def G_a(mean: torch.Tensor, sd: torch.Tensor):
    return (mean.pow(2) / sd.pow(2).clamp_min(1e-8)).clamp_min(1e-4)


def G_b(mean: torch.Tensor, sd: torch.Tensor):
    return (mean / sd.pow(2).clamp_min(1e-8)).clamp_min(1e-4)


class HVGKModuleKineticsPyroModule(PyroModule):
    """Non-VAE module-aware RNA kinetics model with cell2fate-like observation hierarchy."""

    def __init__(
        self,
        n_obs: int,
        n_vars: int,
        n_batch: int,
        n_cell_types: int,
        module_ids: Optional[torch.Tensor] = None,
        n_modules: Optional[int] = None,
        n_extra_categoricals: Optional[int] = None,
        prior_config: Optional[dict] = None,
        observation_config: Optional[dict] = None,
        time_parameterization: str = "centered",
        direction_target: Optional[torch.Tensor] = None,
        direction_weight: float = 0.0,
        direction_mask: Optional[torch.Tensor] = None,
        spatial_config: Optional[dict] = None,
        spatial_coords: Optional[torch.Tensor] = None,
        spatial_adj: Optional[torch.Tensor] = None,
        cell_type_composition: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_obs = int(n_obs)
        self.n_vars = int(n_vars)
        self.n_batch = int(n_batch)
        self.n_cell_types = int(n_cell_types)
        self.n_extra_categoricals = n_extra_categoricals
        if min(self.n_obs, self.n_vars, self.n_batch, self.n_cell_types) < 1:
            raise ValueError("All data dimensions must be positive.")

        self.use_hard_assignment = module_ids is not None
        if self.use_hard_assignment:
            module_ids = torch.as_tensor(module_ids, dtype=torch.long).view(-1)
            if module_ids.numel() != self.n_vars:
                raise ValueError(f"`module_ids` length must equal n_vars ({self.n_vars}), got {module_ids.numel()}.")
            if int(module_ids.min().item()) < 0:
                raise ValueError("`module_ids` must be non-negative integers.")
            inferred_modules = int(module_ids.max().item()) + 1
            self.n_modules = int(n_modules) if n_modules is not None else inferred_modules
            if inferred_modules > self.n_modules:
                raise ValueError(f"`n_modules` ({self.n_modules}) is smaller than inferred max module id ({inferred_modules - 1}).")
            self.register_buffer("module_ids", module_ids)
        else:
            if n_modules is None:
                raise ValueError("When `module_ids` is not provided, `n_modules` must be set.")
            self.n_modules = int(n_modules)
            self.register_buffer("module_ids", torch.zeros(self.n_vars, dtype=torch.long))

        if self.n_modules < 1:
            raise ValueError("n_modules must be positive.")
        cfg = prior_config or {}
        if time_parameterization not in ("centered", "noncentered"):
            raise ValueError("time_parameterization must be centered or noncentered")
        self.time_parameterization = time_parameterization
        if not math.isfinite(direction_weight) or direction_weight < 0:
            raise ValueError("direction_weight must be finite and nonnegative")
        self.direction_weight = float(direction_weight)
        if direction_target is not None:
            direction_target = torch.as_tensor(direction_target, dtype=torch.float32)
            if direction_target.shape != (self.n_obs, self.n_vars) or not torch.isfinite(direction_target).all():
                raise ValueError("direction_target must be finite with shape (n_obs,n_vars)")
            direction_mask = torch.ones(self.n_obs) if direction_mask is None else torch.as_tensor(direction_mask,dtype=torch.float32)
            if direction_mask.shape != (self.n_obs,) or not torch.isfinite(direction_mask).all() or (direction_mask<0).any():
                raise ValueError("direction_mask must be finite, nonnegative and have shape (n_obs,)")
        elif direction_weight:
            raise ValueError("Positive direction_weight requires a direction_target")
        self.register_buffer("direction_target",direction_target)
        self.register_buffer("direction_mask",direction_mask)
        observation_cfg = observation_config or {}
        unknown = set(observation_cfg) - {"gene_detection", "ambient_enabled"}
        if unknown:
            raise ValueError(f"Unknown observation settings: {sorted(unknown)}")
        self.gene_detection_mode = observation_cfg.get("gene_detection", "independent")
        if self.gene_detection_mode not in ("independent", "shared"):
            raise ValueError("gene_detection must be independent or shared")
        self.ambient_enabled = bool(observation_cfg.get("ambient_enabled", True))
        if "Tmax_prior" in cfg:
            warnings.warn("Tmax_prior is deprecated: only mean sets a fixed numerical scale; sd is not used. Set time_scale instead.", DeprecationWarning, stacklevel=2)
        time_scale = float(cfg.get("time_scale", cfg.get("Tmax_prior", {}).get("mean", 50.0)))
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be finite and positive.")
        splicing_cfg = cfg.get("splicing_rate_hyp_prior", {"mean": 1.0, "alpha": 5.0, "mean_hyp_alpha": 10.0, "alpha_hyp_alpha": 20.0})
        degr_cfg = cfg.get("degradation_rate_hyp_prior", {"mean": 1.0, "alpha": 5.0, "mean_hyp_alpha": 10.0, "alpha_hyp_alpha": 20.0})
        activation_cfg = cfg.get("activation_rate_hyp_prior", {"mean_hyp_prior_mean": 2.0, "mean_hyp_prior_sd": 0.33, "sd_hyp_prior_mean": 0.33, "sd_hyp_prior_sd": 0.1})
        factor_cfg = cfg.get("factor_prior", {"rate": 1.0, "alpha": 1.0, "states_per_gene": 3.0})
        amp_cfg = cfg.get("amp_prior", {"sigma_scale": 0.1})
        time_shift_cfg = cfg.get(
            "time_shift_prior",
            {
                "enabled": True,
                "ton_sigma_scale": 0.03,
                "toff_sigma_scale": 0.03,
                "scale_by_tmax": True,
                "min_gap_frac": 0.01,
            },
        )
        baseline_s_cfg = cfg.get("baseline_s_prior", {"enabled": False, "alpha": 2.0, "beta": 50.0})
        baseline_u_cfg = cfg.get("baseline_u_prior", {"enabled": False, "alpha": 2.0, "beta": 100.0})
        det_cfg = cfg.get("detection_hyp_prior", {"alpha": 10.0, "mean_alpha": 1.0, "mean_beta": 1.0})
        det_i_cfg = cfg.get("detection_i_prior", {"mean": 1.0, "alpha": 100.0})
        det_gi_cfg = cfg.get("detection_gi_prior", {"mean": 1.0, "alpha": 200.0})
        gene_add_alpha_cfg = cfg.get("gene_add_alpha_hyp_prior", {"alpha": 9.0, "beta": 3.0})
        gene_add_mean_cfg = cfg.get("gene_add_mean_hyp_prior", {"alpha": 1.0, "beta": 100.0})
        stochastic_cfg = cfg.get("stochastic_v_ag_hyp_prior", {"alpha": 6.0, "beta": 3.0})
        beta_gamma_spatial_cfg = cfg.get(
            "beta_gamma_spatial_residual_prior",
            {"enabled": False, "sigma_scale": 0.05},
        )

        self.register_buffer("tmax_mean", torch.tensor(time_scale))

        self.register_buffer("splicing_alpha_hyp_alpha", torch.tensor(float(splicing_cfg["alpha_hyp_alpha"])))
        self.register_buffer("splicing_alpha_hyp_mean", torch.tensor(float(splicing_cfg["alpha"])))
        self.register_buffer("splicing_mean_hyp_alpha", torch.tensor(float(splicing_cfg["mean_hyp_alpha"])))
        self.register_buffer("splicing_mean_hyp_mean", torch.tensor(float(splicing_cfg["mean"])))

        self.register_buffer("degr_alpha_hyp_alpha", torch.tensor(float(degr_cfg["alpha_hyp_alpha"])))
        self.register_buffer("degr_alpha_hyp_mean", torch.tensor(float(degr_cfg["alpha"])))
        self.register_buffer("degr_mean_hyp_alpha", torch.tensor(float(degr_cfg["mean_hyp_alpha"])))
        self.register_buffer("degr_mean_hyp_mean", torch.tensor(float(degr_cfg["mean"])))

        self.register_buffer("activation_mean_hyp_prior_mean", torch.tensor(float(activation_cfg["mean_hyp_prior_mean"])))
        self.register_buffer("activation_mean_hyp_prior_sd", torch.tensor(float(activation_cfg["mean_hyp_prior_sd"])))
        self.register_buffer("activation_sd_hyp_prior_mean", torch.tensor(float(activation_cfg["sd_hyp_prior_mean"])))
        self.register_buffer("activation_sd_hyp_prior_sd", torch.tensor(float(activation_cfg["sd_hyp_prior_sd"])))

        self.register_buffer("factor_prior_alpha", torch.tensor(float(factor_cfg["alpha"])))
        self.register_buffer("factor_prior_beta", torch.tensor(float(factor_cfg["alpha"] / max(float(factor_cfg["rate"]), 1e-8))))
        self.register_buffer("factor_states_per_gene", torch.tensor(float(factor_cfg["states_per_gene"])))

        self.register_buffer("amp_sigma_scale", torch.tensor(float(amp_cfg["sigma_scale"])))
        self.time_shift_enabled = bool(time_shift_cfg.get("enabled", True))
        self.register_buffer("time_shift_ton_sigma_scale", torch.tensor(float(time_shift_cfg["ton_sigma_scale"])))
        self.register_buffer("time_shift_toff_sigma_scale", torch.tensor(float(time_shift_cfg["toff_sigma_scale"])))
        self.time_shift_scale_by_tmax = bool(time_shift_cfg.get("scale_by_tmax", True))
        self.register_buffer("time_shift_min_gap_frac", torch.tensor(float(time_shift_cfg.get("min_gap_frac", 0.01))))

        self.baseline_s_enabled = bool(baseline_s_cfg.get("enabled", False))
        self.register_buffer("baseline_s_alpha", torch.tensor(float(baseline_s_cfg["alpha"])))
        self.register_buffer("baseline_s_beta", torch.tensor(float(baseline_s_cfg["beta"])))
        self.baseline_u_enabled = bool(baseline_u_cfg.get("enabled", False))
        self.register_buffer("baseline_u_alpha", torch.tensor(float(baseline_u_cfg["alpha"])))
        self.register_buffer("baseline_u_beta", torch.tensor(float(baseline_u_cfg["beta"])))

        self.register_buffer("det_alpha", torch.tensor(float(det_cfg["alpha"])))
        self.register_buffer("det_mean_alpha", torch.tensor(float(det_cfg["mean_alpha"])))
        self.register_buffer("det_mean_beta", torch.tensor(float(det_cfg["mean_beta"])))
        self.register_buffer("det_i_alpha", torch.tensor(float(det_i_cfg["alpha"])))
        self.register_buffer("det_gi_alpha", torch.tensor(float(det_gi_cfg["alpha"])))

        self.register_buffer("gene_add_alpha_hyp_alpha", torch.tensor(float(gene_add_alpha_cfg["alpha"])))
        self.register_buffer("gene_add_alpha_hyp_beta", torch.tensor(float(gene_add_alpha_cfg["beta"])))
        self.register_buffer("gene_add_mean_hyp_alpha", torch.tensor(float(gene_add_mean_cfg["alpha"])))
        self.register_buffer("gene_add_mean_hyp_beta", torch.tensor(float(gene_add_mean_cfg["beta"])))

        self.register_buffer("stochastic_v_ag_hyp_alpha", torch.tensor(float(stochastic_cfg["alpha"])))
        self.register_buffer("stochastic_v_ag_hyp_beta", torch.tensor(float(stochastic_cfg["beta"])))
        self.beta_gamma_spatial_residual_enabled = bool(beta_gamma_spatial_cfg.get("enabled", False))
        self.register_buffer(
            "beta_gamma_spatial_residual_sigma_scale",
            torch.tensor(float(beta_gamma_spatial_cfg.get("sigma_scale", 0.05))),
        )

        if self.baseline_u_enabled:
            raise ValueError("Independent baseline_u is unsupported; enable baseline_s to derive a paired equilibrium baseline.")
        self.register_buffer("alpha_off", torch.tensor(0.0))
        self.register_buffer("eps", torch.tensor(float(eps)))

        spatial_cfg = spatial_config or {}
        composition_mode = str(spatial_cfg.get("composition_mode", "hard")).lower()
        if composition_mode not in {"hard", "soft"}:
            raise ValueError("`spatial_config['composition_mode']` must be one of {'hard', 'soft'}.")
        self.spatial_enabled = bool(spatial_cfg.get("enabled", False))
        self.spatial_use_transition_layer = bool(spatial_cfg.get("use_transition_layer", False))
        if composition_mode == "soft":
            raise ValueError("Soft spot mixtures require summing type-specific kinetic solutions, not averaging switching times; not supported in this single-cell model.")
        if self.spatial_use_transition_layer:
            raise ValueError("The former transition network had no training objective and has been removed.")
        self.state_temperature = float(spatial_cfg.get("state_temperature", 0.02))
        if not math.isfinite(self.state_temperature) or self.state_temperature <= 0:
            raise ValueError("state_temperature must be positive.")
        for key in ("lambda_time", "lambda_state", "lambda_velocity"):
            if not math.isfinite(float(spatial_cfg.get(key, 0.0))) or float(spatial_cfg.get(key, 0.0)) < 0:
                raise ValueError(f"{key} must be non-negative.")
        self.spatial_composition_mode = composition_mode
        self.register_buffer("spatial_lambda_time", torch.tensor(float(spatial_cfg.get("lambda_time", 0.0))))
        self.register_buffer("spatial_lambda_state", torch.tensor(float(spatial_cfg.get("lambda_state", 0.0))))
        self.register_buffer("spatial_lambda_velocity", torch.tensor(float(spatial_cfg.get("lambda_velocity", 0.0))))


        self._set_optional_global_tensor("spatial_coords_global", spatial_coords, n_rows=self.n_obs)
        self._set_optional_global_tensor("spatial_adj_global", spatial_adj, n_rows=self.n_obs, n_cols=self.n_obs)
        self._set_optional_global_tensor("cell_type_composition_global", cell_type_composition, n_rows=self.n_obs, n_cols=self.n_cell_types)

    def _set_optional_global_tensor(self, name: str, value: Optional[torch.Tensor], n_rows: int, n_cols: Optional[int] = None):
        if value is None:
            self.register_buffer(name, torch.empty(0), persistent=True)
            return
        tensor = torch.as_tensor(value)
        if tensor.layout != torch.strided:
            tensor = tensor.coalesce()
        if tensor.dim() != 2:
            raise ValueError(f"`{name}` must be a rank-2 tensor, got shape={tuple(tensor.shape)}.")
        if tensor.size(0) != n_rows:
            raise ValueError(f"`{name}` first dim must be n_obs={n_rows}, got {tensor.size(0)}.")
        if n_cols is not None and tensor.size(1) != n_cols:
            raise ValueError(f"`{name}` second dim must be {n_cols}, got {tensor.size(1)}.")
        self.register_buffer(name, tensor.float(), persistent=True)

    @staticmethod
    def _get_fn_args_from_batch(tensor_dict):
        u_data = tensor_dict["unspliced"]
        s_data = tensor_dict["spliced"]
        idx = tensor_dict[SCVI_REGISTRY_KEYS.INDICES_KEY].long().squeeze(-1)
        batch_index = tensor_dict[SCVI_REGISTRY_KEYS.BATCH_KEY]
        cell_type = tensor_dict["cell_type"]
        spatial_coords = tensor_dict.get("spatial_coords")
        spatial_adj = tensor_dict.get("spatial_adj")
        cell_type_composition = tensor_dict.get("cell_type_composition")
        # Keep optional spatial inputs in kwargs (not args), so posterior sampling
        # can safely move only required args to device without hitting NoneType.
        return (u_data, s_data, idx, batch_index, cell_type), {
            "spatial_coords": spatial_coords,
            "spatial_adj": spatial_adj,
            "cell_type_composition": cell_type_composition,
        }

    def list_obs_plate_vars(self):
        return {"name": "obs_plate", "input": [], "input_transform": [], "sites": {}}

    def create_plates(self, u_data, s_data, idx, batch_index, cell_type, spatial_coords=None, spatial_adj=None, cell_type_composition=None):
        del u_data, s_data, batch_index, cell_type, spatial_coords, spatial_adj, cell_type_composition
        return pyro.plate("obs_plate", size=self.n_obs, dim=-2, subsample=idx)

    def _subsample_optional_input(
        self,
        tensor: Optional[torch.Tensor],
        global_tensor: torch.Tensor,
        idx: torch.Tensor,
        expected_second_dim: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if tensor is not None:
            out = tensor
        elif global_tensor.numel() > 0:
            out = global_tensor.index_select(0, idx)
        else:
            return None
        if expected_second_dim is not None and out.dim() == 2 and out.size(1) != expected_second_dim:
            raise ValueError(f"Expected shape [B, {expected_second_dim}] but got {tuple(out.shape)}.")
        return out

    @staticmethod
    def _normalize_adjacency(adj, eps=1e-8):
        if adj is None:
            return None
        if adj.dim() != 2 or adj.size(0) != adj.size(1):
            raise ValueError("spatial_adj must be square.")
        coo = adj.to_sparse_coo().coalesce()
        ij, weights = coo.indices(), coo.values()
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Spatial weights must be finite and non-negative.")
        keep = (ij[0] != ij[1]) & (weights > 0)
        ij, weights = ij[:, keep], weights[keep]
        row_sum = torch.zeros(adj.size(0), device=weights.device, dtype=weights.dtype)
        row_sum.scatter_add_(0, ij[0], weights)
        return torch.sparse_coo_tensor(ij, weights / row_sum[ij[0]].clamp_min(eps), adj.shape).coalesce()

    def _resolve_spatial_inputs(self, idx, spatial_coords, spatial_adj, cell_type_composition):
        coords = self._subsample_optional_input(spatial_coords, self.spatial_coords_global, idx)
        comp = self._subsample_optional_input(cell_type_composition, self.cell_type_composition_global, idx)
        if not self.spatial_enabled:
            return coords, None, comp
        if idx.numel() != self.n_obs or idx.unique().numel() != self.n_obs:
            raise ValueError("Spatial regularization requires all cells in one batch; induced minibatch graphs are biased.")
        adj = spatial_adj if spatial_adj is not None else self.spatial_adj_global
        if adj.numel() == 0:
            raise ValueError("Spatial mode requires a full adjacency matrix.")
        coo = adj.to_sparse_coo().coalesce()
        inverse = torch.empty_like(idx)
        inverse[idx] = torch.arange(idx.numel(), device=idx.device)
        adj = torch.sparse_coo_tensor(inverse[coo.indices()], coo.values(), coo.shape).coalesce()
        return coords, self._normalize_adjacency(adj), comp

    @staticmethod
    def _spatial_smooth(values, adj_norm):
        if adj_norm is None:
            return values
        flat = values.reshape(values.size(0), -1)
        smooth = torch.sparse.mm(adj_norm, flat)
        degree = torch.sparse.sum(adj_norm, dim=1).to_dense()
        return torch.where(degree[:, None] > 0, smooth, flat).view_as(values)

    @staticmethod
    def _spatial_mse(values, adj_norm):
        """Normalized weighted edge Dirichlet energy; isolated nodes contribute zero."""
        if adj_norm is None:
            return values.sum() * 0.0
        coo = adj_norm.coalesce()
        ij, weights = coo.indices(), coo.values()
        flat = values.reshape(values.size(0), -1)
        total = values.sum() * 0.0
        for start in range(0, weights.numel(), 4096):
            edge = ij[:, start:start + 4096]
            delta = flat[edge[0]] - flat[edge[1]]
            total = total + (weights[start:start + 4096] * delta.square().mean(-1)).sum()
        return total / weights.sum().clamp_min(1e-8)

    def _resolve_celltype_effects(
        self,
        ct: torch.Tensor,
        amp_ctk: torch.Tensor,
        T_mON_ct: torch.Tensor,
        T_mOFF_ct: torch.Tensor,
        baseline_ctg_s: torch.Tensor,
        baseline_ctg_u: torch.Tensor,
        cell_type_composition: Optional[torch.Tensor],
    ):
        if self.spatial_composition_mode == "soft" and cell_type_composition is None:
            raise ValueError("`composition_mode='soft'` requires `cell_type_composition` with shape [B, n_cell_types].")
        if self.spatial_composition_mode == "soft" and cell_type_composition is not None:
            pi_ic = cell_type_composition.float()
            pi_ic = pi_ic / pi_ic.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            amp_ik = pi_ic @ amp_ctk
            t_on_ik = pi_ic @ T_mON_ct
            t_off_ik = pi_ic @ T_mOFF_ct
            baseline_ig_s = pi_ic @ baseline_ctg_s
            baseline_ig_u = pi_ic @ baseline_ctg_u
            return amp_ik, t_on_ik, t_off_ik, baseline_ig_s, baseline_ig_u, pi_ic

        amp_ik = amp_ctk[ct, :]
        t_on_ik = T_mON_ct[ct, :]
        t_off_ik = T_mOFF_ct[ct, :]
        baseline_ig_s = baseline_ctg_s[ct, :]
        baseline_ig_u = baseline_ctg_u[ct, :]
        pi_ic = F.one_hot(ct, num_classes=self.n_cell_types).to(dtype=amp_ctk.dtype)
        return amp_ik, t_on_ik, t_off_ik, baseline_ig_s, baseline_ig_u, pi_ic

    @staticmethod
    def _mu_alpha(alpha_new, alpha_old, tau, lam):
        return (alpha_new - alpha_old) * (1.0 - torch.exp(-lam * tau)) + alpha_old

    def _mu_mrna_cont_alpha(self, alpha, beta, gamma, tau, u0, s0, delta_alpha, lam):
        return propagate(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam)

    def _hard_state_probs(self, t_c, t_on, t_off):
        return soft_states(t_c, t_on, t_off, self.state_temperature)

    def _piecewise_mu(self, t_c, t_on, t_off, alpha, beta, gamma, lam_on, lam_off):
        # This implementation mirrors the cell2fate-style
        # mu_mRNA_continousAlpha_globalTime_twoStates reference:
        # tau/t0 boolean split with continuous alpha transition across ON/OFF states.
        eps = self.eps.to(device=alpha.device, dtype=alpha.dtype)
        max_dim = max(t_c.dim(), t_on.dim(), t_off.dim(), alpha.dim(), beta.dim(), gamma.dim(), lam_on.dim(), lam_off.dim())
        while t_c.dim() < max_dim:
            t_c = t_c.unsqueeze(-1)
        while t_on.dim() < max_dim:
            t_on = t_on.unsqueeze(-1)
        while t_off.dim() < max_dim:
            t_off = t_off.unsqueeze(-1)
        while alpha.dim() < max_dim:
            alpha = alpha.unsqueeze(-1)
        while beta.dim() < max_dim:
            beta = beta.unsqueeze(-1)
        while gamma.dim() < max_dim:
            gamma = gamma.unsqueeze(-1)
        while lam_on.dim() < max_dim:
            lam_on = lam_on.unsqueeze(-1)
        while lam_off.dim() < max_dim:
            lam_off = lam_off.unsqueeze(-1)

        t, t_on, t_off, alpha, beta, gamma, lam_on, lam_off = torch.broadcast_tensors(
            t_c, t_on, t_off, alpha, beta, gamma, lam_on, lam_off
        )

        alpha_off = self.alpha_off.to(device=alpha.device, dtype=alpha.dtype).expand_as(alpha)
        tau = torch.clamp(t - t_on, min=0.0)
        t0 = torch.clamp(t_off - t_on, min=0.0)
        is_on = tau < t0

        alpha_cg = torch.where(is_on, alpha, alpha_off)
        tau_cg = torch.where(is_on, tau, tau - t0)
        lam = torch.where(is_on, lam_on, lam_off)

        u_init, s_init = self._mu_mrna_cont_alpha(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            tau=t0,
            u0=torch.zeros_like(alpha),
            s0=torch.zeros_like(alpha),
            delta_alpha=alpha - alpha_off,
            lam=lam_on,
        )
        alpha_init = self._mu_alpha(alpha, alpha_off, t0, lam_on)
        u0 = torch.where(is_on, torch.zeros_like(alpha), u_init)
        s0 = torch.where(is_on, torch.zeros_like(alpha), s_init)
        delta_alpha = torch.where(is_on, alpha - alpha_off, alpha_off - alpha_init)

        mu_u, mu_s = self._mu_mrna_cont_alpha(
            alpha=alpha_cg,
            beta=beta,
            gamma=gamma,
            tau=tau_cg,
            u0=u0,
            s0=s0,
            delta_alpha=delta_alpha,
            lam=lam,
        )

        state_probs = self._hard_state_probs(t, t_on, t_off)
        return mu_u.clamp_min(0.0), mu_s.clamp_min(0.0), state_probs

    def forward(
        self,
        u_data,
        s_data,
        idx,
        batch_index,
        cell_type,
        spatial_coords=None,
        spatial_adj=None,
        cell_type_composition=None,
    ):
        if not torch.isfinite(u_data).all() or not torch.isfinite(s_data).all() or (u_data < 0).any() or (s_data < 0).any():
            raise ValueError("RNA counts must be finite and non-negative.")
        batch_size = len(idx)
        device = u_data.device
        dtype = u_data.dtype

        obs2sample = F.one_hot(batch_index.long().squeeze(-1), num_classes=self.n_batch).to(dtype=dtype)
        obs_plate = self.create_plates(
            u_data,
            s_data,
            idx,
            batch_index,
            cell_type,
            spatial_coords=spatial_coords,
            spatial_adj=spatial_adj,
            cell_type_composition=cell_type_composition,
        )

        # Gene-level kinetics (cell2fate-style hierarchical priors)
        splicing_alpha = pyro.sample(
            "splicing_alpha",
            dist.Gamma(
                self.splicing_alpha_hyp_alpha.to(device=device, dtype=dtype),
                self.splicing_alpha_hyp_alpha.to(device=device, dtype=dtype) / self.splicing_alpha_hyp_mean.to(device=device, dtype=dtype),
            ),
        )
        splicing_mean = pyro.sample(
            "splicing_mean",
            dist.Gamma(
                self.splicing_mean_hyp_alpha.to(device=device, dtype=dtype),
                self.splicing_mean_hyp_alpha.to(device=device, dtype=dtype) / self.splicing_mean_hyp_mean.to(device=device, dtype=dtype),
            ),
        )
        beta_g = pyro.sample("beta_g", dist.Gamma(splicing_alpha, splicing_alpha / splicing_mean).expand([1, self.n_vars]).to_event(2))

        degr_alpha = pyro.sample(
            "degradation_alpha",
            dist.Gamma(
                self.degr_alpha_hyp_alpha.to(device=device, dtype=dtype),
                self.degr_alpha_hyp_alpha.to(device=device, dtype=dtype) / self.degr_alpha_hyp_mean.to(device=device, dtype=dtype),
            ),
        )
        degr_alpha = degr_alpha + 1e-3
        degr_mean = pyro.sample(
            "degradation_mean",
            dist.Gamma(
                self.degr_mean_hyp_alpha.to(device=device, dtype=dtype),
                self.degr_mean_hyp_alpha.to(device=device, dtype=dtype) / self.degr_mean_hyp_mean.to(device=device, dtype=dtype),
            ),
        )
        gamma_g = pyro.sample("gamma_g", dist.Gamma(degr_alpha, degr_alpha / degr_mean).expand([1, self.n_vars]).to_event(2))

        if self.beta_gamma_spatial_residual_enabled and self.n_cell_types > 1:
            residual_sigma = self.beta_gamma_spatial_residual_sigma_scale.to(device=device, dtype=dtype)
            delta_beta_ctg_raw = pyro.sample(
                "delta_beta_ctg_raw",
                dist.Normal(
                    torch.zeros((self.n_cell_types, self.n_vars), device=device, dtype=dtype),
                    residual_sigma,
                ).to_event(2),
            )
            delta_gamma_ctg_raw = pyro.sample(
                "delta_gamma_ctg_raw",
                dist.Normal(
                    torch.zeros((self.n_cell_types, self.n_vars), device=device, dtype=dtype),
                    residual_sigma,
                ).to_event(2),
            )
            delta_beta_ctg = pyro.deterministic(
                "delta_beta_ctg",
                delta_beta_ctg_raw - delta_beta_ctg_raw.mean(dim=0, keepdim=True),
            )
            delta_gamma_ctg = pyro.deterministic(
                "delta_gamma_ctg",
                delta_gamma_ctg_raw - delta_gamma_ctg_raw.mean(dim=0, keepdim=True),
            )
        else:
            delta_beta_ctg = pyro.deterministic(
                "delta_beta_ctg",
                torch.zeros((self.n_cell_types, self.n_vars), device=device, dtype=dtype),
            )
            delta_gamma_ctg = pyro.deterministic(
                "delta_gamma_ctg",
                torch.zeros((self.n_cell_types, self.n_vars), device=device, dtype=dtype),
            )

        # Module-gene loadings (cell2fate-like), shared across cell types.
        factor_level_g = pyro.sample(
            "factor_level_g",
            dist.Gamma(
                self.factor_prior_alpha.to(device=device, dtype=dtype),
                self.factor_prior_beta.to(device=device, dtype=dtype),
            ).expand([1, self.n_vars]).to_event(2),
        )
        if self.use_hard_assignment:
            active_loading = pyro.sample("g_active", dist.Gamma(
                self.factor_states_per_gene.to(device=device, dtype=dtype),
                1.0 / factor_level_g.clamp_min(1e-6)).to_event(2))
            mask = F.one_hot(self.module_ids, self.n_modules).T.to(dtype)
            g_fg = pyro.deterministic("g_fg", active_loading * mask)
        else:
            g_fg = pyro.sample("g_fg", dist.Gamma(
                (self.factor_states_per_gene.to(device=device, dtype=dtype) / self.n_modules).clamp_min(1e-4),
                1.0 / factor_level_g.clamp_min(1e-6)).expand([self.n_modules, self.n_vars]).to_event(2))
        A_mgON = pyro.deterministic("A_mgON", g_fg * gamma_g)

        # Shared module kinetics: module identity is primarily temporal/state-structural, not cell-identity-specific.
        # Main module differences should come from switch timing and activation/repression rates.
        lam_mu = pyro.sample(
            "lam_mu",
            dist.Gamma(
                G_a(self.activation_mean_hyp_prior_mean.to(device=device, dtype=dtype), self.activation_mean_hyp_prior_sd.to(device=device, dtype=dtype)),
                G_b(self.activation_mean_hyp_prior_mean.to(device=device, dtype=dtype), self.activation_mean_hyp_prior_sd.to(device=device, dtype=dtype)),
            ),
        )
        lam_sd = pyro.sample(
            "lam_sd",
            dist.Gamma(
                G_a(self.activation_sd_hyp_prior_mean.to(device=device, dtype=dtype), self.activation_sd_hyp_prior_sd.to(device=device, dtype=dtype)),
                G_b(self.activation_sd_hyp_prior_mean.to(device=device, dtype=dtype), self.activation_sd_hyp_prior_sd.to(device=device, dtype=dtype)),
            ),
        )
        lam_m_mu = pyro.sample(
            "lam_m_mu",
            dist.Gamma(G_a(lam_mu, lam_sd), G_b(lam_mu, lam_sd)).expand([self.n_modules, 1]).to_event(2),
        )
        lam_mi = pyro.sample(
            "lam_mi",
            dist.Gamma(G_a(lam_m_mu, lam_m_mu * 0.05), G_b(lam_m_mu, lam_m_mu * 0.05)).expand([self.n_modules, 2]).to_event(2),
        )
        lambda_on_k = lam_mi[:, 0]
        lambda_off_k = lam_mi[:, 1]

        # Time
        # Fixed numerical time convention, not experimentally identified physical time.
        t_max = pyro.deterministic("Tmax", self.tmax_mean.to(device=device, dtype=dtype))
        t_c_loc = pyro.sample("t_c_loc", dist.Normal(torch.tensor(-0.7, device=device, dtype=dtype), torch.tensor(0.5, device=device, dtype=dtype)))
        t_c_scale = pyro.sample("t_c_scale", dist.HalfNormal(torch.tensor(0.5, device=device, dtype=dtype)))

        if self.n_modules > 1:
            t_delta = pyro.sample(
                "t_delta",
                dist.Gamma(
                    torch.ones(max(self.n_modules - 1, 0), device=device, dtype=dtype) * 20.0,
                    torch.ones(max(self.n_modules - 1, 0), device=device, dtype=dtype) * (20.0 * float(self.n_modules)),
                ).to_event(1),
            )
        else:
            t_delta = pyro.deterministic("t_delta", torch.empty(0, device=device, dtype=dtype))
        t_on_base = torch.cumsum(torch.cat([torch.zeros(1, device=device, dtype=dtype), t_delta], dim=0), dim=0)
        t_on_frac = t_on_base / (1.0 + t_on_base.max())
        t_on_k = pyro.deterministic("T_mON", t_max * t_on_frac)

        t_moff = pyro.sample(
            "t_mOFF",
            dist.Exponential(torch.ones(self.n_modules, device=device, dtype=dtype) * float(self.n_modules)).to_event(1),
        )
        t_off_k = pyro.deterministic("T_mOFF", t_on_k + t_max * t_moff)

        # Cell-type residual on module amplitude (centered).
        # Centering removes a common log-amplitude offset, not biological non-identifiability.
        sigma_amp_k = pyro.sample(
            "sigma_amp_k",
            dist.HalfNormal(torch.ones(self.n_modules, device=device, dtype=dtype) * self.amp_sigma_scale.to(device=device, dtype=dtype)).to_event(1),
        )
        delta_amp_ctk_raw = pyro.sample(
            "delta_amp_ctk_raw",
            dist.Normal(
                torch.zeros((self.n_cell_types, self.n_modules), device=device, dtype=dtype),
                sigma_amp_k.unsqueeze(0).expand(self.n_cell_types, -1),
            ).to_event(2),
        )
        delta_amp_ctk = pyro.deterministic("delta_amp_ctk", delta_amp_ctk_raw - delta_amp_ctk_raw.mean(dim=0, keepdim=True))
        amp_ctk = pyro.deterministic("amp_ctk", torch.exp(delta_amp_ctk))
        pyro.deterministic("amp_centered_mean_check", delta_amp_ctk.mean(dim=0))

        # Cell-type residual on module phase (centered small shift, not cell-type-specific module redefinition).
        if self.time_shift_enabled:
            ton_sigma_base = self.time_shift_ton_sigma_scale.to(device=device, dtype=dtype)
            toff_sigma_base = self.time_shift_toff_sigma_scale.to(device=device, dtype=dtype)
            if self.time_shift_scale_by_tmax:
                ton_sigma_base = ton_sigma_base * t_max
                toff_sigma_base = toff_sigma_base * t_max
            sigma_ton_k = pyro.sample(
                "sigma_ton_k",
                dist.HalfNormal(torch.ones(self.n_modules, device=device, dtype=dtype) * ton_sigma_base).to_event(1),
            )
            sigma_toff_k = pyro.sample(
                "sigma_toff_k",
                dist.HalfNormal(torch.ones(self.n_modules, device=device, dtype=dtype) * toff_sigma_base).to_event(1),
            )
            delta_ton_ctk_raw = pyro.sample(
                "delta_ton_ctk_raw",
                dist.Normal(
                    torch.zeros((self.n_cell_types, self.n_modules), device=device, dtype=dtype),
                    sigma_ton_k.unsqueeze(0).expand(self.n_cell_types, -1),
                ).to_event(2),
            )
            delta_toff_ctk_raw = pyro.sample(
                "delta_toff_ctk_raw",
                dist.Normal(
                    torch.zeros((self.n_cell_types, self.n_modules), device=device, dtype=dtype),
                    sigma_toff_k.unsqueeze(0).expand(self.n_cell_types, -1),
                ).to_event(2),
            )
            delta_ton_ctk = pyro.deterministic("delta_ton_ctk", delta_ton_ctk_raw - delta_ton_ctk_raw.mean(dim=0, keepdim=True))
            delta_toff_ctk = pyro.deterministic("delta_toff_ctk", delta_toff_ctk_raw - delta_toff_ctk_raw.mean(dim=0, keepdim=True))
            T_mON_ct_raw = t_on_k.unsqueeze(0) + delta_ton_ctk
            T_mOFF_ct_raw = t_off_k.unsqueeze(0) + delta_toff_ctk
            min_gap = (self.time_shift_min_gap_frac.to(device=device, dtype=dtype) * t_max).clamp_min(1e-4)
            T_mON_ct = torch.clamp(T_mON_ct_raw, min=0.0)
            T_mOFF_ct = torch.maximum(T_mOFF_ct_raw, T_mON_ct + min_gap)
        else:
            delta_ton_ctk = pyro.deterministic("delta_ton_ctk", torch.zeros((self.n_cell_types, self.n_modules), device=device, dtype=dtype))
            delta_toff_ctk = pyro.deterministic("delta_toff_ctk", torch.zeros((self.n_cell_types, self.n_modules), device=device, dtype=dtype))
            T_mON_ct = t_on_k.unsqueeze(0).expand(self.n_cell_types, -1)
            T_mOFF_ct = t_off_k.unsqueeze(0).expand(self.n_cell_types, -1)

        pyro.deterministic("time_shift_centered_mean_check_on", delta_ton_ctk.mean(dim=0))
        pyro.deterministic("time_shift_centered_mean_check_off", delta_toff_ctk.mean(dim=0))
        pyro.deterministic("T_mON_effective", T_mON_ct)
        pyro.deterministic("T_mOFF_effective", T_mOFF_ct)

        # Optional constitutive steady-state channel, conditional on fixed cell type.
        if self.baseline_s_enabled:
            baseline_ctg_s = pyro.sample(
                "baseline_ctg_s",
                dist.Gamma(
                    self.baseline_s_alpha.to(device=device, dtype=dtype),
                    self.baseline_s_beta.to(device=device, dtype=dtype),
                ).expand([self.n_cell_types, self.n_vars]).to_event(2),
            )
        else:
            baseline_ctg_s = pyro.deterministic(
                "baseline_ctg_s",
                torch.zeros((self.n_cell_types, self.n_vars), device=device, dtype=dtype),
            )
        pyro.deterministic("baseline_s_ct_mean", baseline_ctg_s.mean(dim=-1))
        baseline_ctg_u = pyro.deterministic(
            "baseline_ctg_u",
            (gamma_g * torch.exp(delta_gamma_ctg)) / (beta_g * torch.exp(delta_beta_ctg)) * baseline_ctg_s,
        )

        # Detection hierarchy + additive soup
        detection_mean_y_e = pyro.sample(
            "detection_mean_y_e",
            dist.Beta(
                self.det_mean_alpha.to(device=device, dtype=dtype),
                self.det_mean_beta.to(device=device, dtype=dtype),
            ).expand([self.n_batch, 1]).to_event(2),
        )
        detection_beta_c = self.det_alpha.to(device=device, dtype=dtype) / (obs2sample @ detection_mean_y_e).clamp_min(1e-4)

        detection_y_i = pyro.sample(
            "detection_y_i",
            dist.Gamma(
                self.det_i_alpha.to(device=device, dtype=dtype),
                self.det_i_alpha.to(device=device, dtype=dtype),
            ).expand([1, 1, 2]).to_event(3),
        )
        detection_gene = pyro.sample(
            "detection_y_gi" if self.gene_detection_mode == "independent" else "detection_y_g_shared",
            dist.Gamma(
                self.det_gi_alpha.to(device=device, dtype=dtype),
                self.det_gi_alpha.to(device=device, dtype=dtype),
            ).expand([1, self.n_vars, 2 if self.gene_detection_mode == "independent" else 1]).to_event(3),
        )
        detection_y_gi = (detection_gene if self.gene_detection_mode == "independent" else
                          pyro.deterministic("detection_y_gi", detection_gene.expand(1,self.n_vars,2)))

        if self.ambient_enabled:
            gene_add_alpha_e = pyro.sample(
                "gene_add_alpha_e",
                dist.Gamma(self.gene_add_alpha_hyp_alpha.to(device=device,dtype=dtype),
                           self.gene_add_alpha_hyp_beta.to(device=device,dtype=dtype)).expand([2]).to_event(1))
            gene_add_mean = pyro.sample(
                "gene_add_mean",
                dist.Gamma(self.gene_add_mean_hyp_alpha.to(device=device,dtype=dtype),
                           self.gene_add_mean_hyp_beta.to(device=device,dtype=dtype)).expand([self.n_batch,2]).to_event(2))
            s_g_gene_add = pyro.sample(
                "s_g_gene_add",
                dist.Gamma(
                    gene_add_alpha_e.reshape(1,1,2).expand(self.n_batch,self.n_vars,2),
                    gene_add_alpha_e.reshape(1,1,2).expand(self.n_batch,self.n_vars,2)
                    / gene_add_mean.reshape(self.n_batch,1,2).expand(self.n_batch,self.n_vars,2).clamp_min(1e-6)
                ).to_event(3))
        else:
            s_g_gene_add = pyro.deterministic("s_g_gene_add",torch.zeros(
                (self.n_batch,self.n_vars,2),device=device,dtype=dtype))

        # Overdispersion (cell2fate-style)
        stochastic_v_ag_hyp = pyro.sample(
            "stochastic_v_ag_hyp",
            dist.Gamma(
                self.stochastic_v_ag_hyp_alpha.to(device=device, dtype=dtype),
                self.stochastic_v_ag_hyp_beta.to(device=device, dtype=dtype),
            ).expand([1, 2]).to_event(2),
        )
        stochastic_v_ag_inv = pyro.sample(
            "stochastic_v_ag_inv",
            dist.Exponential((stochastic_v_ag_hyp + 1e-3)).expand([1, self.n_vars, 2]).to_event(3),
        )
        stochastic_v_ag = pyro.deterministic("stochastic_v_ag", 1.0 / stochastic_v_ag_inv.pow(2))

        with obs_plate:
            ct = cell_type.long().squeeze(-1)
            _, spatial_adj_batch, composition_batch = self._resolve_spatial_inputs(
                idx=idx,
                spatial_coords=spatial_coords,
                spatial_adj=spatial_adj,
                cell_type_composition=cell_type_composition,
            )
            if self.spatial_enabled and spatial_adj_batch is None:
                raise ValueError("Spatial mode requires `spatial_adj` (or global `spatial_adj` provided at init).")
            if self.time_parameterization == "noncentered":
                z_time = pyro.sample("z_time", dist.Normal(
                    torch.zeros((), device=device, dtype=dtype),
                    torch.ones((), device=device, dtype=dtype)).expand([batch_size, 1]))
                t_c = pyro.deterministic("t_c", torch.exp(t_c_loc + t_c_scale * z_time))
            else:
                t_c = pyro.sample("t_c", dist.LogNormal(t_c_loc, t_c_scale).expand([batch_size, 1]))
            T_c = pyro.deterministic("T_c", t_c * t_max)

            amp_ik, t_on_ik, t_off_ik, baseline_ig_s, baseline_ig_u, pi_ic = self._resolve_celltype_effects(
                ct=ct,
                amp_ctk=amp_ctk,
                T_mON_ct=T_mON_ct,
                T_mOFF_ct=T_mOFF_ct,
                baseline_ctg_s=baseline_ctg_s,
                baseline_ctg_u=baseline_ctg_u,
                cell_type_composition=composition_batch,
            )
            pyro.deterministic("cell_type_composition_effective", pi_ic)

            # alpha per cell-module-gene
            alpha_eff_cmg = A_mgON.unsqueeze(0) * amp_ik.unsqueeze(-1)
            alpha_eff_module = pyro.deterministic("alpha_eff_module", alpha_eff_cmg.mean(dim=-1))

            # Cell-type phase shifts are small residual timing adjustments on shared modules.
            t_on = t_on_ik.unsqueeze(-1)
            t_off = t_off_ik.unsqueeze(-1)
            beta_eff_gene = pyro.deterministic("beta_eff_gene", beta_g.expand(batch_size, -1) * torch.exp(delta_beta_ctg[ct]))
            gamma_eff_gene = pyro.deterministic("gamma_eff_gene", gamma_g.expand(batch_size, -1) * torch.exp(delta_gamma_ctg[ct]))

            mu_u_cmg, mu_s_cmg, state_probs_cmg = self._piecewise_mu(
                t_c=T_c,
                t_on=t_on,
                t_off=t_off,
                alpha=alpha_eff_cmg,
                beta=beta_eff_gene.unsqueeze(1),
                gamma=gamma_eff_gene.unsqueeze(1),
                lam_on=lambda_on_k.view(1, self.n_modules, 1),
                lam_off=lambda_off_k.view(1, self.n_modules, 1),
            )
            state_probs_cm = soft_states(T_c, t_on_ik, t_off_ik, self.state_temperature * t_max)
            kinetic_mu_u_gene = mu_u_cmg.sum(dim=1)
            kinetic_mu_s_gene = mu_s_cmg.sum(dim=1)
            # A constitutive baseline obeys alpha0 = beta*u0 = gamma*s0.
            baseline_u_gene = baseline_ig_u
            baseline_s_gene = baseline_ig_s
            mu_u_gene = (kinetic_mu_u_gene + baseline_u_gene).clamp_min(1e-8)
            mu_s_gene = (kinetic_mu_s_gene + baseline_s_gene).clamp_min(1e-8)
            velocity_module = beta_eff_gene.unsqueeze(1) * mu_u_cmg - gamma_eff_gene.unsqueeze(1) * mu_s_cmg
            velocity = velocity_module.sum(dim=1)
            if self.direction_target is not None and self.direction_weight:
                target = self.direction_target[idx]
                cosine = F.cosine_similarity(velocity,target,dim=-1,eps=1e-8)
                mask = self.direction_mask[idx] * (target.norm(dim=-1)>1e-8)
                # A generalized-Bayes regularizer derived from the chosen time
                # ordering; not an additional independent observation likelihood.
                pyro.factor("time_tangent_regularization",
                    (-self.direction_weight * mask * (1-cosine)).unsqueeze(-1))
            kinetic_vs_baseline_ratio = pyro.deterministic(
                "kinetic_vs_baseline_ratio",
                kinetic_mu_s_gene.mean(dim=-1, keepdim=True)
                / (baseline_s_gene.mean(dim=-1, keepdim=True) + self.eps.to(device=device, dtype=dtype)),
            )

            detection_y_c = pyro.sample(
                "detection_y_c",
                dist.Gamma(
                    self.det_alpha.to(device=device, dtype=dtype).expand(batch_size, 1),
                    detection_beta_c,
                ),
            )  # (B, 1)

            mu_stack = torch.stack([mu_u_gene, mu_s_gene], dim=-1)
            batch_gene_add = torch.einsum("cb,bgi->cgi", obs2sample, s_g_gene_add)
            mu = (mu_stack + batch_gene_add) * detection_y_c.unsqueeze(-1) * detection_y_i * detection_y_gi
            mu_u_obs = mu[..., 0].clamp_min(1e-8)
            mu_s_obs = mu[..., 1].clamp_min(1e-8)

            obs_scale_s = detection_y_c * detection_y_i[..., 1] * detection_y_gi[..., 1]
            pyro.deterministic("velocity_observed", velocity * obs_scale_s)
            pyro.deterministic("spliced_corrected", s_data / obs_scale_s - batch_gene_add[..., 1])
            pyro.deterministic("expression_biological_s", mu_s_gene)
            pyro.deterministic("expression_biological_u", mu_u_gene)
            pyro.deterministic("mu_u", mu_u_obs)
            pyro.deterministic("mu_s", mu_s_obs)
            pyro.deterministic("state_probs_module", state_probs_cm)
            pyro.deterministic("state_probs", state_probs_cm)
            pyro.deterministic("velocity", velocity)
            pyro.deterministic("kinetic_mu_s", kinetic_mu_s_gene.clamp_min(1e-8))
            pyro.deterministic("baseline_mu_s", baseline_s_gene)
            pyro.deterministic("kinetic_mu_u", kinetic_mu_u_gene.clamp_min(1e-8))
            pyro.deterministic("baseline_mu_u", baseline_u_gene)

            u_is_count = torch.allclose(u_data, u_data.round(), atol=1e-6, rtol=0.0)
            s_is_count = torch.allclose(s_data, s_data.round(), atol=1e-6, rtol=0.0)
            if not bool(u_is_count and s_is_count):
                raise ValueError(
                    "This model uses GammaPoisson likelihood only (cell2fate-compatible). "
                    "Please pass integer count layers."
                )

            conc_u = stochastic_v_ag[..., 0].expand(batch_size, self.n_vars)
            conc_s = stochastic_v_ag[..., 1].expand(batch_size, self.n_vars)
            with pyro.plate("genes", self.n_vars, dim=-1):
                pyro.sample("u", dist.GammaPoisson(concentration=conc_u, rate=conc_u / mu_u_obs), obs=u_data)
                pyro.sample("s", dist.GammaPoisson(concentration=conc_s, rate=conc_s / mu_s_obs), obs=s_data)

        # Graph couples cells: one explicit global factor, outside the independent-cell plate.
        if self.spatial_enabled:
            reg_time = self._spatial_mse(T_c / t_max, spatial_adj_batch)
            reg_state = self._spatial_mse(state_probs_cm, spatial_adj_batch)
            reg_velocity = self._spatial_mse(velocity, spatial_adj_batch)
            reg_total = (self.spatial_lambda_time * reg_time + self.spatial_lambda_state * reg_state
                         + self.spatial_lambda_velocity * reg_velocity)
            pyro.deterministic("spatial_reg_time", reg_time)
            pyro.deterministic("spatial_reg_state", reg_state)
            pyro.deterministic("spatial_reg_velocity", reg_velocity)
            pyro.deterministic("spatial_reg_total", reg_total)
            pyro.factor("spatial_regularization", -self.n_obs * reg_total)
        else:
            for name in ("time", "state", "velocity", "total"):
                pyro.deterministic("spatial_reg_" + name, velocity.new_zeros(()))



HVGKPyroModule = HVGKModuleKineticsPyroModule

