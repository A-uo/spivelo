"""Conditional kinetic refinement with a fixed first-stage time posterior."""
import numpy as np
import torch


def local_time_tangents(expression, log_time, graph, min_side=2):
    """Weighted local slope d expression/d log(time), without group labels.

    Require neighbors on both sides to avoid one-sided boundary extrapolation.
    Target coordinates must match biological velocity; keep observation mapping
    fixed during refinement. This is a data-derived regularizer, not new data.
    """
    x=np.asarray(expression,dtype=float); t=np.asarray(log_time,dtype=float).reshape(-1)
    if x.ndim!=2 or len(t)!=len(x) or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError('Expected finite cell-by-gene expression and matching log times')
    graph=graph.tocsr()
    if graph.shape!=(len(x),len(x)):
        raise ValueError('Graph shape does not match cells')
    if min_side<1:
        raise ValueError('min_side must be positive')
    target=np.zeros_like(x); usable=np.zeros(len(x),dtype=bool)
    for i in range(len(x)):
        js=graph.indices[graph.indptr[i]:graph.indptr[i+1]]
        dt=t[js]-t[i]
        if (dt>1e-6).sum()<min_side or (dt< -1e-6).sum()<min_side:
            continue
        bandwidth=max(np.median(np.abs(dt)),1e-6)
        w=np.exp(-.5*(dt/bandwidth)**2)
        # Fit a free intercept. Anchoring at the noisy central observation adds
        # -x[i] * sum(w*dt) to the numerator in asymmetric neighborhoods.
        centered_dt=dt-np.sum(w*dt)/w.sum()
        denom=np.sum(w*centered_dt**2)
        if denom>1e-12:
            neighbor_mean=np.sum(w[:,None]*x[js],axis=0)/w.sum()
            target[i]=np.sum((w*centered_dt)[:,None]*(x[js]-neighbor_mean),axis=0)/denom
            usable[i]=np.linalg.norm(target[i])>1e-8
    return target.astype(np.float32),usable


def configure_direction_stage(model, target, mask, weight):
    """Freeze both time-guide factors and the observation coordinate mapping.

    Freezing guide factors preserves the entire fitted time distribution, not
    just a plug-in ordering. Kinetic parameters remain learnable.
    """
    module=model.module.model
    target=torch.as_tensor(target,dtype=module.tmax_mean.dtype,device=module.tmax_mean.device)
    mask=torch.as_tensor(mask,dtype=target.dtype,device=target.device)
    if target.shape!=(model.adata.n_obs,model.adata.n_vars) or mask.shape!=(model.adata.n_obs,):
        raise ValueError('Direction targets/mask must match registered cells and genes')
    if not torch.isfinite(target).all() or not torch.isfinite(mask).all() or (mask<0).any():
        raise ValueError('Nonfinite targets or invalid mask')
    if not np.isfinite(weight) or weight<0:
        raise ValueError('Direction weight must be finite and nonnegative')
    module.direction_target=target
    module.direction_mask=mask
    module.direction_weight=float(weight)
    sites=('t_c','t_c_loc','t_c_scale','z_time','detection_',
           'gene_add_','s_g_gene_add','stochastic_')
    frozen=[]
    for name,p in model.module.guide.named_parameters():
        site=name.split('.',1)[-1].removesuffix('_unconstrained')
        freeze=site in sites or site.startswith(('detection_','gene_add_','s_g_gene_add','stochastic_'))
        p.requires_grad_(not freeze)
        if freeze: frozen.append(name)
    if not frozen:
        raise RuntimeError('Initialize the guide before configuring second-stage training')
    return frozen
