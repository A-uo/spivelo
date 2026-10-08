"""Inference checks that preserve joint samples and keep evaluation labels out of fitting."""
import numpy as np
import pandas as pd
import torch
import pyro
from pyro import poutine
from pyro.infer import Trace_ELBO
from pyro.util import get_rng_state, set_rng_state


def log_time_summary(model):
    """Exact log-time moments of the fitted guide, without posterior MC noise.

    exp(E[log T]) is a geometric center, NOT E[T] or a credible interval.
    Noncentered moments integrate independent guide factors m, sigma and z.
    """
    guide = model.module.guide
    with torch.no_grad():
        if model.module.model.time_parameterization == "centered":
            mean = guide.locs.t_c
            variance = guide.scales.t_c.square()
            ordering_mean, ordering_sd = mean, guide.scales.t_c
        else:
            m, vm = guide.locs.t_c_loc, guide.scales.t_c_loc.square()
            a, b = guide.locs.t_c_scale, guide.scales.t_c_scale.square()
            es = torch.exp(a + b/2)
            es2 = torch.exp(2*a + 2*b)
            z, vz = guide.locs.z_time, guide.scales.z_time.square()
            ordering_mean, ordering_sd = z, guide.scales.z_time
            mean = m + es*z
            variance = vm + es2*vz + (es2-es.square()).clamp_min(0)*z.square()
        mean = mean + model.module.model.tmax_mean.log()
        return pd.DataFrame({
            "mean_log_time":mean.cpu().numpy().reshape(-1),
            "sd_log_time":variance.sqrt().cpu().numpy().reshape(-1),
            "geometric_time":mean.exp().cpu().numpy().reshape(-1),
            "ordering_coordinate":ordering_mean.cpu().numpy().reshape(-1),
            "ordering_sd":ordering_sd.cpu().numpy().reshape(-1)}, index=model.adata.obs_names)


def time_sensitivity(model, log_step=0.02):
    """Local NB Fisher information for log time at guide coordinate medians.

    Profiles the cell detection multiplier through its Schur complement. Other
    parameters are fixed: these are local diagnostics, not posterior precision.
    A central difference is used, so check step sensitivity near switch points.
    """
    if not np.isfinite(log_step) or log_step <= 0:
        raise ValueError("log_step must be finite and positive")
    args, kwargs = model._joint_args()
    with torch.no_grad():
        values = model.module.guide.median(*args, **kwargs)
        def evaluate(offset):
            changed = dict(values)
            if "z_time" in changed:
                changed["z_time"] = changed["z_time"] + offset/changed["t_c_scale"]
            else:
                changed["t_c"] = changed["t_c"] * np.exp(offset)
            tr = poutine.trace(poutine.condition(model.module.model, data=changed)).get_trace(*args, **kwargs)
            mu = torch.stack([tr.nodes["mu_u"]["value"],tr.nodes["mu_s"]["value"]],-1)
            theta = tr.nodes["stochastic_v_ag"]["value"]
            return mu, theta
        mu, theta = evaluate(0.)
        upper,_ = evaluate(log_step)
        lower,_ = evaluate(-log_step)
        derivative = (upper-lower)/(2*log_step)
        variance = (mu + mu.square()/theta).clamp_min(1e-12)
        information = (derivative.square()/variance).sum((1,2))
        cross = (derivative*mu/variance).sum((1,2))
        depth_information = (mu.square()/variance).sum((1,2)).clamp_min(1e-12)
        profiled = (information-cross.square()/depth_information).clamp_min(0)
        return pd.DataFrame({"conditional_log_time_information":information.cpu().numpy(),
            "depth_profiled_log_time_information":profiled.cpu().numpy(),
            "information_retained_after_depth":(profiled/information.clamp_min(1e-12)).cpu().numpy()},
            index=model.adata.obs_names)


def fit_restarts(factory, starts, train_kwargs, score_samples=10, score_seed=31415):
    """Fit equal-budget starts of ONE generative model; select by MC training ELBO.

    factory accepts one start dictionary. Scores are not held-out performance.
    The returned model is rebound to the Pyro store. Do not use other models concurrently.
    """
    if not starts:
        raise ValueError("At least one start is required")
    best, best_score, rows, times = None, np.inf, [], []
    for index, start in enumerate(starts):
        pyro.set_rng_seed(int(start.get("seed",0)))
        candidate = factory(start)
        candidate.train(**train_kwargs)
        score = elbo_score(candidate, score_samples, score_seed)
        times.append(candidate.sample_posterior(num_samples=score_samples,
                     return_sites=['T_c'])['post_sample_means']['T_c'].reshape(-1))
        row = {"start":index, **start, **score}
        if "elbo_train" in candidate.history:
            history=np.asarray(candidate.history['elbo_train'],dtype=float).reshape(-1)
            if not np.isfinite(history).all():
                raise FloatingPointError("Nonfinite training history")
            width=max(1,min(50,len(history)//4))
            row['tail_relative_change']=(history[-width:].mean()-history[-2*width:-width].mean())/max(abs(history[-2*width:-width].mean()),1.) if len(history)>=2*width else np.nan
        rows.append(row)
        print(f"Start {index}: {start}, negative ELBO={score['negative_elbo']:.3f}, MC SE={score['mc_se']:.3f}",flush=True)
        if score['negative_elbo'] < best_score:
            best, best_score = candidate, score['negative_elbo']
    pyro.clear_param_store()
    restored = elbo_score(best, score_samples, score_seed)
    if not np.isclose(restored['negative_elbo'],best_score,rtol=1e-5,atol=1e-4):
        raise RuntimeError("Selected model parameters changed when rebinding Pyro store")
    table=pd.DataFrame(rows)
    table['selected']=table.start==int(table.negative_elbo.idxmin())
    best_se=float(table.loc[table.selected,'mc_se'].iloc[0])
    table['within_2_mc_se']=(table.negative_elbo-best_score)<=2*np.sqrt(table.mc_se**2+best_se**2)
    from scipy.stats import spearmanr
    selected=int(table.negative_elbo.idxmin())
    table['time_rank_corr_to_selected']=[float(spearmanr(t,times[selected])[0])
        if np.std(t)>0 and np.std(times[selected])>0 else np.nan for t in times]
    best.restart_time_means_=np.stack(times)
    return best, table


def elbo_score(model, num_samples=5, seed=0):
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    rng = get_rng_state()
    try:
        pyro.set_rng_seed(seed)
        args, kwargs = model._joint_args()
        with torch.no_grad():
            values = np.array([Trace_ELBO().loss(model.module.model, model.module.guide, *args, **kwargs)
                               for _ in range(num_samples)])
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite ELBO score")
        return {"negative_elbo":float(values.mean()), "mc_sd":float(values.std()),
                "mc_se":float(values.std()/np.sqrt(num_samples))}
    finally:
        set_rng_state(rng)


def unregularized_elbo_score(model, num_samples=5, seed=0):
    """Common count-model ELBO for comparing direction-regularization arms.

    Temporarily disables only the new direction factor; restores it even on error.
    """
    module=model.module.model
    previous=module.direction_weight
    try:
        module.direction_weight=0.
        return elbo_score(model,num_samples,seed)
    finally:
        module.direction_weight=previous


def time_information(model, num_samples=10, n_permutations=5, seed=0):
    if num_samples < 1 or n_permutations < 1:
        raise ValueError("Sample and permutation counts must be positive")
    rng = get_rng_state()
    try:
        pyro.set_rng_seed(seed)
        args, kwargs = model._joint_args()
        n = model.adata.n_obs
        records, cell_deltas = [], []
        def likelihood(values):
            trace = poutine.trace(poutine.condition(model.module.model, data=values)).get_trace(*args, **kwargs)
            # Only count likelihood: explicitly exclude priors and spatial factors.
            score = sum(trace.nodes[key]["fn"].log_prob(trace.nodes[key]["value"]).reshape(n,-1).sum(-1)
                        for key in ("u", "s"))
            if not torch.isfinite(score).all():
                raise FloatingPointError("Nonfinite conditional count likelihood")
            return score
        with torch.no_grad():
            for draw in range(num_samples):
                guide_trace = poutine.trace(model.module.guide).get_trace(*args, **kwargs)
                trace = poutine.trace(poutine.replay(model.module.model,trace=guide_trace)).get_trace(*args, **kwargs)
                values = {name: node["value"] for name,node in trace.nodes.items()
                          if node["type"]=="sample" and not node.get("is_observed",False)
                          and name not in ("obs_plate","genes") and not node.get("infer",{}).get("_deterministic",False)}
                base = likelihood(values)
                for perm in range(n_permutations):
                    time_site = "z_time" if "z_time" in values else "t_c"
                    order = torch.randperm(n, device=values[time_site].device)
                    changed = dict(values)
                    changed[time_site] = values[time_site][order]
                    delta = base-likelihood(changed)
                    cell_deltas.append(delta.cpu().numpy())
                    records.append({"draw":draw,"permutation":perm,"loglik_gain":delta.sum().item(),
                                    "gain_per_count_entry":delta.sum().item()/(2*n*model.adata.n_vars)})
        return {"draws":pd.DataFrame(records),
                "cells":pd.DataFrame({"mean_loglik_gain":np.mean(cell_deltas,axis=0)},index=model.adata.obs_names)}
    finally:
        set_rng_state(rng)
