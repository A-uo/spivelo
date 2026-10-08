"""Coordinate variational update of original module loadings, preserving q(time).

Uses the linearity of the RNA ODE in transcription amplitudes to cache basis
responses. All other variational factors remain exactly as in the checkpoint.
"""
import numpy as np
import torch,pyro
from scipy.optimize import nnls
from ._trajectory_refinement import nb_log_prob


@torch.no_grad()
def loading_basis(model,values):
    mod=model.module.model
    if mod.use_hard_assignment or mod.n_cell_types!=1 or mod.baseline_s_enabled:
        raise ValueError('This coordinate update currently supports soft modules, one unlabeled population, and no baseline')
    args,kw=model._joint_args()
    tr=pyro.poutine.trace(pyro.poutine.condition(mod,data=values)).get_trace(*args,**kw)
    def v(k):return tr.nodes[k]['value'].detach()
    common=dict(t_c=v('T_c'),t_on=v('T_mON')[None,:,None],t_off=v('T_mOFF')[None,:,None],
        beta=v('beta_g')[:,None,:],gamma=v('gamma_g')[:,None,:],
        lam_on=v('lam_mi')[:,0][None,:,None],lam_off=v('lam_mi')[:,1][None,:,None])
    one=mod._piecewise_mu(alpha=v('gamma_g')[None,:,:].expand(1,mod.n_modules,-1),**common)
    zero=mod._piecewise_mu(alpha=torch.zeros(1,mod.n_modules,mod.n_vars),**common)
    basis=torch.stack([one[0]-zero[0],one[1]-zero[1]],-1)
    offset=torch.stack([zero[0].sum(1),zero[1].sum(1)],-1)
    scale=v('detection_y_c')[...,None]*v('detection_y_i')*v('detection_y_gi')
    # Verify the cached representation against the actual generative model.
    reconstructed=torch.einsum('nmgc,mg->ngc',basis,v('g_fg'))+offset
    actual=torch.stack([v('expression_biological_u'),v('expression_biological_s')],-1)
    torch.testing.assert_close(reconstructed,actual,atol=2e-5,rtol=2e-5)
    return dict(basis=basis,offset=offset,scale=scale,ambient=v('s_g_gene_add'),
        theta=v('stochastic_v_ag'),prior_rate=1/v('factor_level_g').clamp_min(1e-6))


def lognormal_gamma_kl(loc,sd,shape,rate):
    """Analytic KL(q LogNormal || p Gamma) in the constrained loading space."""
    return (-shape*loc-sd.log()-.5*np.log(2*np.pi*np.e)
        +rate*torch.exp(loc+sd.square()/2)-shape*rate.log()+torch.lgamma(shape)).sum()


def refine_loadings(model,counts,corrected,time_mean,steps=500,cache_draws=16,seed=42,callback=None):
    """Warm-start loadings from unlabeled time-bin NNLS; optimize original ELBO block.

    Frozen-factor expectation is approximated by 16 joint draws, not by replacing
    the time posterior with a point estimate. Two antithetic loading draws reduce
    gradient noise. Evaluation must use independent posterior seeds.
    """
    pyro.set_rng_seed(seed);args,kw=model._joint_args()
    with torch.no_grad():
        median=loading_basis(model,model.module.guide.median(*args,**kw))
    order=np.argsort(time_mean);groups=np.array_split(order,min(12,len(order)))
    x=np.stack([median['basis'][ids].mean(0).numpy() for ids in groups])
    y=np.stack([np.asarray(corrected)[ids].mean(0) for ids in groups])
    off=np.stack([median['offset'][ids].mean(0).numpy() for ids in groups])
    init=[]
    for g in range(counts.shape[1]):
        design=x[:,:,g,:].transpose(0,2,1).reshape(-1,x.shape[1])
        coef,_=nnls(design,(y[:,g,:]-off[:,g,:]).reshape(-1));init.append(coef)
    loc=torch.nn.Parameter(torch.tensor(np.maximum(np.array(init).T,.001),dtype=torch.float32).log())
    log_sd=torch.nn.Parameter(torch.full_like(loc,np.log(.1)))
    cache=[]
    with torch.no_grad():
        for d in range(cache_draws):
            vals=model.module.guide(*args,**kw)
            cache.append(loading_basis(model,vals))
            if callback:callback({'cache_draw':d+1})
    counts=torch.as_tensor(counts,dtype=torch.float32)
    shape=model.module.model.factor_states_per_gene/model.module.model.n_modules
    opt=torch.optim.Adam([loc,log_sd],lr=.02);history=[]
    for step in range(steps):
        sample=cache[step%len(cache)]
        sd=log_sd.exp();noise=torch.randn_like(loc)
        opt.zero_grad();nll=0.
        for sign in (-1,1):
            g=(loc+sign*sd*noise).exp()
            bio=torch.einsum('nmgc,mg->ngc',sample['basis'],g)+sample['offset']
            mu=(bio+sample['ambient'])*sample['scale']
            nll=nll-.5*nb_log_prob(counts,mu,sample['theta']).sum()
        kl=lognormal_gamma_kl(loc,sd,shape,sample['prior_rate'])
        ((nll+kl)/counts.shape[0]).backward();torch.nn.utils.clip_grad_norm_([loc,log_sd],100.)
        opt.step()
        row=dict(step=step+1,count_nll=float(nll.detach()),loading_KL=float(kl.detach()))
        if not np.isfinite(row['count_nll']):raise FloatingPointError('Nonfinite coordinate ELBO')
        history.append(row)
        if callback and (step==0 or (step+1)%50==0):callback(row)
    # AutoNormal stores scale through a softplus transform.
    from torch.distributions import transform_to
    with torch.no_grad():
        model.module.guide.locs.g_fg.copy_(loc)
        raw=model.module.guide.scales.g_fg.unconstrained()
        raw.copy_(transform_to(model.module.guide.scale_constraint).inv(log_sd.exp()))
    return history
