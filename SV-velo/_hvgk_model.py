"""High-level API for the non-VAE hierarchical Bayesian HVGK model."""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from pyro import clear_param_store, poutine
from scvi import REGISTRY_KEYS as SCVI_REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField, NumericalObsField
from scvi.data.fields import ObsmField
from scipy import sparse
from ._validation import validate_counts
from scvi.model.base import BaseModelClass, PyroSampleMixin, PyroSviTrainMixin
from scvi.utils import setup_anndata_dsp

from ._constants import REGISTRY_KEYS
from ._hvgk_pyro_module import HVGKPyroModule
from ._pyro_infra import LocalPyroBaseModule


class HVGK(PyroSampleMixin, PyroSviTrainMixin, BaseModelClass):
    """Hierarchical Velocity Generative Kinetics model.

    This is a non-VAE probabilistic model where kinetics and state-time structure
    are the model core and neural networks are only optional guide utilities.
    """

    def __init__(
        self,
        adata: AnnData,
        model_class=None,
        spatial_coords_key: Optional[str] = None,
        spatial_adj_key: Optional[str] = None,
        cell_type_composition_key: Optional[str] = None,
        initialization: str = "prior",
        initialization_seed: int = 0,
        initialization_reverse: bool = False,
        **model_kwargs,
    ):
        clear_param_store()
        super().__init__(adata)

        if model_class is None:
            model_class = HVGKPyroModule

        ct_state = self.adata_manager.get_state_registry(REGISTRY_KEYS.CELL_TYPE_KEY)
        n_cell_types = len(ct_state.categorical_mapping)
        n_obs = int(self.summary_stats.get("n_cells", self.summary_stats.get("n_obs", adata.n_obs)))
        n_vars = int(self.summary_stats.get("n_vars", adata.n_vars))
        n_batch = int(self.summary_stats.get("n_batch", 1))
        if "module_ids" not in model_kwargs:
            # Priority:
            # 1) If n_modules is explicitly provided, use cell2fate-like soft module loadings.
            # 2) Else fallback to hard assignment from adata.var["module_id"] if available.
            if "n_modules" in model_kwargs:
                pass
            elif "module_id" in adata.var.columns:
                model_kwargs["module_ids"] = adata.var["module_id"].to_numpy().astype("int64")
            else:
                raise ValueError(
                    "Please provide either `module_ids` (hard assignment) or `n_modules` "
                    "(cell2fate-like soft module loadings)."
                )

        # Optional global spatial inputs. These are indexed by minibatch `idx` inside the Pyro module.
        if "spatial_coords" not in model_kwargs and spatial_coords_key is not None:
            if spatial_coords_key not in adata.obsm:
                raise KeyError(f"`spatial_coords_key='{spatial_coords_key}'` not found in `adata.obsm`.")
            model_kwargs["spatial_coords"] = torch.as_tensor(np.asarray(adata.obsm[spatial_coords_key]), dtype=torch.float32)
        if "cell_type_composition" not in model_kwargs and cell_type_composition_key is not None:
            if cell_type_composition_key not in adata.obsm:
                raise KeyError(f"`cell_type_composition_key='{cell_type_composition_key}'` not found in `adata.obsm`.")
            model_kwargs["cell_type_composition"] = torch.as_tensor(np.asarray(adata.obsm[cell_type_composition_key]), dtype=torch.float32)
        if "spatial_adj" not in model_kwargs and spatial_adj_key is not None:
            if spatial_adj_key not in adata.obsp:
                raise KeyError(f"`spatial_adj_key='{spatial_adj_key}'` not found in `adata.obsp`.")
            spatial_adj = adata.obsp[spatial_adj_key]
            if sparse.issparse(spatial_adj):
                coo = spatial_adj.tocoo()
                model_kwargs["spatial_adj"] = torch.sparse_coo_tensor(
                    np.vstack([coo.row, coo.col]), coo.data.astype(np.float32), coo.shape).coalesce()
            else:
                model_kwargs["spatial_adj"] = torch.as_tensor(np.asarray(spatial_adj), dtype=torch.float32)

        init_vals = model_kwargs.pop("init_vals", None)
        if init_vals is not None and initialization != "prior":
            raise ValueError("Use explicit init_vals OR an initialization strategy, not both.")
        if initialization != "prior":
            if "module_ids" in model_kwargs:
                raise ValueError("Label-free loading initialization currently requires soft modules (n_modules).")
            from ._initialization import initial_values
            init_vals = initial_values(
                self.adata_manager.get_from_registry("unspliced"),
                self.adata_manager.get_from_registry("spliced"),
                model_kwargs["n_modules"], initialization, initialization_seed, initialization_reverse)
        if model_kwargs.get("time_parameterization", "centered") == "noncentered":
            # Match the centered guide's initial physical time, including explicit
            # hyperparameter initial values. This changes coordinates, not the prior.
            init_vals = dict(init_vals or {})
            loc = torch.as_tensor(init_vals.get("t_c_loc", -0.7))
            scale = torch.as_tensor(init_vals.get("t_c_scale", 0.5 * np.sqrt(2 / np.pi)))
            if not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError("Initial t_c_scale must be finite and positive")
            if "z_time" in init_vals and "t_c" in init_vals:
                raise ValueError("Provide initial t_c or z_time, not both")
            if "z_time" not in init_vals:
                target = torch.as_tensor(init_vals.pop("t_c", torch.exp(loc + scale.square()/2).expand(n_obs, 1)))
                if not torch.isfinite(target).all() or (target <= 0).any():
                    raise ValueError("Initial t_c must be finite and positive")
                init_vals["z_time"] = (target.log() - loc) / scale
        self.module = LocalPyroBaseModule(
            model=model_class,
            init_vals=init_vals,
            n_obs=n_obs,
            n_vars=n_vars,
            n_batch=n_batch,
            n_cell_types=n_cell_types,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"HVGK hierarchical Bayesian model; n_batch={n_batch}, "
            f"n_cell_types={n_cell_types}"
        )
        self.init_params_ = self._get_init_params(locals())
        self._training_obs_names = adata.obs_names.copy()
        self._training_var_names = adata.var_names.copy()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer: str = "spliced",
        unspliced_layer: str = "unspliced",
        cell_type_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        spatial_coords_key: Optional[str] = None,
        cell_type_composition_key: Optional[str] = None,
        **kwargs,
    ):
        """Register AnnData for HVGK.

        Parameters
        ----------
        adata
            AnnData object.
        spliced_layer
            Layer name for spliced counts.
        unspliced_layer
            Layer name for unspliced counts.
        cell_type_key
            Categorical obs key for cell type.
        batch_key
            Optional batch key in obs.
        """
        for layer in (spliced_layer, unspliced_layer):
            if layer not in adata.layers:
                raise KeyError(f"Missing raw count layer {layer!r}.")
            validate_counts(adata.layers[layer], layer)
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs).astype("int64")
        anndata_fields = [
            LayerField("unspliced", unspliced_layer, is_count_data=True),
            LayerField("spliced", spliced_layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.CELL_TYPE_KEY, cell_type_key),
            NumericalObsField(SCVI_REGISTRY_KEYS.INDICES_KEY, "_indices"),
            CategoricalObsField(SCVI_REGISTRY_KEYS.BATCH_KEY, batch_key),
        ]
        if spatial_coords_key is not None:
            anndata_fields.append(ObsmField("spatial_coords", spatial_coords_key))
        if cell_type_composition_key is not None:
            anndata_fields.append(ObsmField("cell_type_composition", cell_type_composition_key))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: Optional[int] = None,
        accelerator: str = "auto",
        device: int | str = "auto",
        train_size: Optional[float] = None,
        validation_size: Optional[float] = None,
        shuffle_set_split: bool = True,
        batch_size: Optional[int] = None,
        early_stopping: bool = False,
        lr: Optional[float] = None,
        training_plan=None,
        datasplitter_kwargs: Optional[dict] = None,
        plan_config=None,
        plan_kwargs=None,
        trainer_config=None,
        **trainer_kwargs,
    ):
        if train_size not in (None, 1.0) or validation_size not in (None, 0.0):
            raise ValueError("Per-cell latent variables require fitting all registered cells; fit a separate dataset for external validation.")
        if early_stopping:
            raise ValueError("No independent validation objective is implemented; early_stopping is unsupported.")
        if self.module.model.spatial_enabled:
            if batch_size is not None and batch_size < self.adata.n_obs:
                raise ValueError("Spatial training requires batch_size >= n_obs (or None).")
            if train_size not in (None, 1.0) or validation_size not in (None, 0.0):
                raise ValueError("Full-graph fitting is transductive; use a separate dataset for independent validation.")
        train_size = 1.0 if train_size is None else train_size
        batch_size = self.adata.n_obs if batch_size is None else batch_size
        # Initialize local guide parameters in registered cell order, before a shuffled loader.
        if self.module.guide.prototype_trace is None:
            args, kwargs = self._joint_args()
            with torch.no_grad():
                self.module.guide(*args, **kwargs)
        return super().train(
            max_epochs=max_epochs,
            accelerator=accelerator,
            device=device,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            early_stopping=early_stopping,
            lr=lr,
            training_plan=training_plan,
            datasplitter_kwargs=datasplitter_kwargs,
            plan_config=plan_config,
            plan_kwargs=plan_kwargs,
            trainer_config=trainer_config,
            **trainer_kwargs,
        )

    def _joint_args(self):
        self._check_training_adata(None)
        device = next(self.module.model.buffers()).device
        loader = self._make_data_loader(adata=self.adata, batch_size=self.adata.n_obs, shuffle=False)
        tensors = {k: v.to(device) for k, v in next(iter(loader)).items()}
        return self.module._get_fn_args_from_batch(tensors)

    def time_information(self, num_samples=10, n_permutations=5, seed=0):
        """Paired conditional likelihood sensitivity; not a held-out score or causal test."""
        from ._inference import time_information
        return time_information(self, num_samples, n_permutations, seed)

    def _check_training_adata(self, adata):
        if adata is not None and adata is not self.adata:
            raise ValueError("HVGK has per-cell latent variables; new/subset/reordered AnnData requires a new fit.")
        if not self.adata.obs_names.equals(self._training_obs_names) or not self.adata.var_names.equals(self._training_var_names):
            raise ValueError("Training cell/gene identities or order have changed; refit the model.")
        return self.adata

    @torch.inference_mode()
    def sample_posterior(self, adata=None, num_samples=30, batch_size=None,
                         return_sites=None, return_samples=False, accelerator="auto",
                         device="auto", **kwargs):
        """Joint full-data draws preserve shared globals and arbitrary local tensor shapes.

        Explicitly avoids guessing cell axes from a single plate dimension.
        No out-of-sample or independent per-minibatch global draws are implied.
        """
        self._check_training_adata(adata)
        if not getattr(self, "is_trained_", False):
            raise RuntimeError("Train the model before requesting a posterior.")
        if kwargs:
            raise TypeError(f"Unsupported posterior options: {sorted(kwargs)}")
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        if batch_size is not None and batch_size < self.adata.n_obs:
            raise ValueError("Joint posterior export requires batch_size >= n_obs (or None).")
        if accelerator not in ("auto", "cpu", "gpu", "cuda"):
            raise ValueError("Supported accelerators: auto, cpu, gpu/cuda.")
        if accelerator == "cpu":
            self.module.cpu()
        elif accelerator in ("gpu", "cuda"):
            self.module.cuda(0 if device == "auto" else int(device))
        target = next(self.module.model.buffers()).device
        loader = self._make_data_loader(adata=self.adata, batch_size=self.adata.n_obs, shuffle=False)
        tensors = {k: v.to(target) for k, v in next(iter(loader)).items()}
        args, model_kwargs = self.module._get_fn_args_from_batch(tensors)
        draws = {}
        for _ in range(num_samples):
            guide_trace = poutine.trace(self.module.guide).get_trace(*args, **model_kwargs)
            trace = poutine.trace(poutine.replay(self.module.model, trace=guide_trace)).get_trace(*args, **model_kwargs)
            values = {name: site["value"].detach().cpu().numpy()
                      for name, site in trace.nodes.items()
                      if site["type"] == "sample" and
                      (site.get("infer", {}).get("_deterministic", False) or
                       (not site.get("is_observed", False) and name not in ("obs_plate", "genes")))}
            if return_sites is not None:
                missing = set(return_sites) - values.keys()
                if missing:
                    raise KeyError(f"Unknown/unavailable posterior sites: {sorted(missing)}")
                values = {key: values[key] for key in return_sites}
            for key, value in values.items():
                draws.setdefault(key, []).append(value)
        arrays = {key: np.stack(value) for key, value in draws.items()}
        result = {
            "post_sample_means": {k: v.mean(0) for k, v in arrays.items()},
            "post_sample_stds": {k: v.std(0) for k, v in arrays.items()},
            "post_sample_q05": {k: np.quantile(v, .05, axis=0) for k, v in arrays.items()},
            "post_sample_q95": {k: np.quantile(v, .95, axis=0) for k, v in arrays.items()},
        }
        if return_samples:
            result["posterior_samples"] = arrays
        return result

    def export_posterior(
        self,
        adata: AnnData,
        sample_kwargs: Optional[dict] = None,
        export_slot: str = "hvgk",
    ) -> AnnData:
        self._check_training_adata(adata)
        if sample_kwargs is None:
            sample_kwargs = {
                "num_samples": 30,
                "batch_size": adata.n_obs,
                "accelerator": "auto",
                "return_samples": True,
            }
        if sample_kwargs.get("batch_size") is None:
            sample_kwargs["batch_size"] = adata.n_obs

        samples = self.sample_posterior(**sample_kwargs)
        means = samples["post_sample_means"]
        adata.uns[export_slot] = {
            "model_name": "HVGK",
            "model_revision": "2026-09-27",
            "velocity_interpretation": "biological RNA derivative conditional on fixed cell type; arbitrary time unit; not physical migration",
            "state_interpretation": "smooth three-state memberships, not calibrated fate probabilities",
            "date": str(date.today()),
            "var_names": adata.var_names.tolist(),
            "obs_names": adata.obs_names.tolist(),
            "post_sample_means": means,
            "post_sample_stds": samples["post_sample_stds"],
            "post_sample_q05": samples["post_sample_q05"],
            "post_sample_q95": samples["post_sample_q95"],
        }
        if sample_kwargs.get("return_samples", False):
            adata.uns[export_slot]["post_samples"] = samples["posterior_samples"]

        if "T_c" in means:
            adata.obs["latent_time_hvgk"] = np.asarray(means["T_c"]).reshape(-1)
        if "velocity" in means:
            adata.layers["velocity"] = np.asarray(means["velocity"])
        for site in ("velocity_observed", "expression_biological_s", "expression_biological_u", "spliced_corrected"):
            if site in means:
                adata.layers[site] = np.asarray(means[site])
        if "mu_s" in means:
            adata.layers["spliced_fit"] = np.asarray(means["mu_s"])
        if "mu_u" in means:
            adata.layers["unspliced_fit"] = np.asarray(means["mu_u"])
        if "state_probs" in means:
            st = np.asarray(means["state_probs"]).mean(axis=1)
            adata.obsm["hvgk_state_probs"] = st
            state_names = ["pre_induction", "induction", "repression"]
            for i, n in enumerate(state_names):
                adata.obs[f"hvgk_state_{n}"] = st[:, i]
            adata.obs["hvgk_state"] = pd.Categorical(np.asarray(state_names)[st.argmax(1)], categories=state_names)

        return adata

    @torch.inference_mode()
    def get_velocity(self, adata: Optional[AnnData] = None, n_samples: int = 20):
        adata = self._check_training_adata(adata)
        s = self.sample_posterior(
            num_samples=n_samples,
            batch_size=adata.n_obs,
            accelerator="auto",
            return_sites=["velocity"],
            return_samples=False,
        )
        return s["post_sample_means"]["velocity"]

    @torch.inference_mode()
    def get_latent_time(self, adata: Optional[AnnData] = None, n_samples: int = 20):
        adata = self._check_training_adata(adata)
        s = self.sample_posterior(
            num_samples=n_samples,
            batch_size=adata.n_obs,
            accelerator="auto",
            return_sites=["T_c"],
            return_samples=False,
        )
        return s["post_sample_means"]["T_c"]

    @torch.inference_mode()
    def get_state_probs(self, adata: Optional[AnnData] = None, n_samples: int = 20):
        adata = self._check_training_adata(adata)
        s = self.sample_posterior(
            num_samples=n_samples,
            batch_size=adata.n_obs,
            accelerator="auto",
            return_sites=["state_probs"],
            return_samples=False,
        )
        return s["post_sample_means"]["state_probs"]

    @torch.inference_mode()
    def get_rates(self, adata: Optional[AnnData] = None, n_samples: int = 20):
        adata = self._check_training_adata(adata)
        s = self.sample_posterior(
            num_samples=n_samples,
            batch_size=adata.n_obs,
            accelerator="auto",
            return_sites=[
                "beta_g",
                "gamma_g",
                "splicing_alpha",
                "splicing_mean",
                "degradation_alpha",
                "degradation_mean",
                "A_mgON",
                "g_fg",
                "factor_level_g",
                "lam_mu",
                "lam_sd",
                "lam_m_mu",
                "lam_mi",
                "T_mON",
                "T_mOFF",
                "T_mON_effective",
                "T_mOFF_effective",
                "delta_amp_ctk_raw",
                "delta_amp_ctk",
                "amp_ctk",
                "amp_centered_mean_check",
                "delta_ton_ctk_raw",
                "delta_toff_ctk_raw",
                "delta_ton_ctk",
                "delta_toff_ctk",
                "time_shift_centered_mean_check_on",
                "time_shift_centered_mean_check_off",
                "baseline_ctg_s",
                "baseline_ctg_u",
                "baseline_s_ct_mean",
                "alpha_eff_module",
                "kinetic_vs_baseline_ratio",
                "detection_mean_y_e",
                "detection_y_i",
                "detection_y_gi",
                "gene_add_alpha_e",
                "gene_add_mean",
                "s_g_gene_add",
                "stochastic_v_ag",
                "Tmax",
            ],
            return_samples=False,
        )
        return s["post_sample_means"]

    def plot_velocity_flow(
        self,
        adata: Optional[AnnData] = None,
        **plot_kwargs,
    ):
        """Wrapper for unified RNA velocity flow plotting.

        Delegates to :func:`velovi.plot_rna_velocity_flow`.
        """
        adata = self._check_training_adata(adata)
        from ._utils import plot_rna_velocity_flow

        return plot_rna_velocity_flow(adata=adata, **plot_kwargs)
