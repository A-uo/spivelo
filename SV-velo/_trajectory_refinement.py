"""Conditional MAP RNA dynamics with a fixed first-stage time posterior.

This is an explicit alternative decoder, not a posterior sample from the
original module model. Counts and velocities share the same exact ODE solution.
No cluster labels, CBDir, or spatial smoothing enter fitting.
"""
import numpy as np
import torch
from torch import nn


def constant_transcription(alpha, beta, gamma, duration, u0, s0):
    """Exact positive linear RNA system, including equal beta/gamma.

    The convolution is evaluated symmetrically, avoiding exponentially growing
    intermediate values. Double precision protects the short-time s increment.
    """
    dtype=alpha.dtype
    a,b,g,t,u,s=torch.broadcast_tensors(*[z.double() for z in
        (alpha,beta,gamma,duration,u0,s0)])
    gap=(b-g).abs(); z=gap*t
    safe=z.clamp_min(1e-12)
    phi=torch.where(z<1e-4,1-z/2+z*z/6-z*z*z/24,-torch.expm1(-z)/safe)
    conv=torch.exp(-torch.minimum(b,g)*t)*t*phi
    fb=-torch.expm1(-b*t)/b;fg=-torch.expm1(-g*t)/g
    un=u*torch.exp(-b*t)+a*fb
    sn=s*torch.exp(-g*t)+b*u*conv+a*(fg-conv)
    return un.to(dtype),sn.to(dtype)


class ConditionalTrajectory(nn.Module):
    """Gene-specific nonnegative transcription on fixed time intervals.

    Positive initial RNA permits tissue already expressing genes before the
    observed window. Piecewise constant transcription makes u,s continuous;
    ds/dt is continuous even at transcription knots.
    """
    def __init__(self,knots,alpha,beta,gamma,initial):
        super().__init__()
        knots=torch.as_tensor(knots,dtype=torch.float32)
        if knots.ndim!=1 or len(knots)<2 or knots[0]!=0 or not torch.all(knots[1:]>knots[:-1]):
            raise ValueError('Knots must increase strictly from zero')
        self.register_buffer('knots',knots)
        for name,value in [('alpha',alpha),('beta',beta),('gamma',gamma),('initial',initial)]:
            x=torch.as_tensor(value,dtype=torch.float32)
            if not torch.isfinite(x).all() or (x<=0).any():raise ValueError('Parameters must be positive and finite')
            setattr(self,'log_'+name,nn.Parameter(x.log()))
        if self.log_alpha.shape!=(len(knots)-1,self.log_beta.numel()) or self.log_initial.shape!=(self.log_beta.numel(),2):
            raise ValueError('Parameter dimensions do not agree')
        self.register_buffer('rate_anchor',torch.stack([self.log_beta.detach(),self.log_gamma.detach()]))

    def forward(self,time):
        time=torch.as_tensor(time,dtype=self.knots.dtype,device=self.knots.device)
        if (time<0).any():raise ValueError('Time must be nonnegative')
        beta,gamma=self.log_beta.exp(),self.log_gamma.exp()
        alpha=self.log_alpha.exp();u,s=self.log_initial.exp().unbind(-1)
        us=[u];ss=[s]
        for k,dt in enumerate(self.knots.diff()[:-1]):
            u,s=constant_transcription(alpha[k],beta,gamma,dt,u,s)
            us.append(u);ss.append(s)
        interval=torch.searchsorted(self.knots[1:],time.contiguous(),right=True).clamp_max(len(alpha)-1)
        u0=torch.stack(us)[interval];s0=torch.stack(ss)[interval]
        dt=(time-self.knots[interval])[...,None]
        u,s=constant_transcription(alpha[interval],beta,gamma,dt,u0,s0)
        return torch.stack([u,s],-1),beta*u-gamma*s

    def penalty(self):
        # Gaussian prior on neighboring log-transcription differences (sd=1)
        # and log rates around the selected first-stage rates (sd=1).
        return .5*self.log_alpha.diff(dim=0).square().sum()+.5*(
            torch.stack([self.log_beta,self.log_gamma])-self.rate_anchor).square().sum()


def initialize_trajectory(arrays,n_intervals=16,train_mask=None):
    t=np.exp(arrays['time_mean']); corrected=arrays['corrected']
    horizon=float(np.quantile(np.exp(arrays['time_mean']+2*arrays['time_sd']),.995))
    knots=np.linspace(0,horizon,n_intervals+1)
    # Neighbor averages initialize parameters only; raw counts remain the likelihood data.
    centers=(knots[:-1]+knots[1:])/2
    width=max(horizon/n_intervals,float(np.std(t)/4))
    w=np.exp(-.5*((centers[:,None]-t[None,:])/width)**2)
    if train_mask is not None:w=w*np.asarray(train_mask,dtype=bool)[None,:]
    avg=np.einsum('kn,ngc->kgc',w,corrected)/w.sum(1)[:,None,None]
    alpha=np.maximum(avg[...,0],.01)*arrays['beta'][None,:]
    initial=np.maximum(avg[0],.01)
    return ConditionalTrajectory(knots,alpha,arrays['beta'],arrays['gamma'],initial)


def nb_log_prob(counts,mu,theta):
    mu=mu.clamp_min(1e-8)
    return (torch.lgamma(counts+theta)-torch.lgamma(theta)-torch.lgamma(counts+1)
        +theta*(theta.log()-(theta+mu).log())+counts*(mu.log()-(theta+mu).log()))


def time_quadrature(arrays,nodes=5):
    x,w=np.polynomial.hermite.hermgauss(nodes)
    t=np.exp(arrays['time_mean'][None,:]+np.sqrt(2)*x[:,None]*arrays['time_sd'][None,:])
    return torch.tensor(t,dtype=torch.float32),torch.tensor(w/np.sqrt(np.pi),dtype=torch.float32)


def fit_conditional(arrays,epochs=300,seed=42,n_intervals=16,lr=.02,callback=None,train_mask=None):
    """Optimize expected count NLL under unchanged q(time), with weak priors.

    Observation parameters are fixed at first-stage guide medians, not integrated.
    This limitation is recorded explicitly in the report. Counts are not smoothed.
    """
    torch.manual_seed(seed)
    if train_mask is None:train_mask=np.ones(len(arrays['counts']),dtype=bool)
    train_mask=np.asarray(train_mask,dtype=bool)
    if train_mask.shape!=(len(arrays['counts']),) or not train_mask.any():
        raise ValueError('Training mask must select at least one cell')
    model=initialize_trajectory(arrays,n_intervals,train_mask)
    times,weights=time_quadrature(arrays)
    counts,scale,ambient,theta=[torch.as_tensor(arrays[k],dtype=torch.float32)
        for k in ('counts','scale','ambient','theta')]
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    history=[]
    for epoch in range(epochs):
        opt.zero_grad(); expected=0.
        # Separate backward passes bound memory; this still uses all quadrature nodes.
        for time,w in zip(times,weights):
            bio,_=model(time);mu=(bio+ambient)*scale
            loss=-nb_log_prob(counts,mu,theta)[train_mask].sum()*w
            (loss/train_mask.sum()).backward();expected+=float(loss.detach())
        prior=model.penalty();(prior/train_mask.sum()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),100.)
        opt.step()
        if not np.isfinite(expected):raise FloatingPointError('Nonfinite likelihood')
        row=dict(epoch=epoch+1,expected_count_nll=expected,prior_penalty=float(prior.detach()))
        history.append(row)
        if callback is not None:callback(model,row)
    return model,history


@torch.no_grad()
def predict_conditional(model,arrays,nodes=15):
    times,weights=time_quadrature(arrays,nodes)
    bio=0.;velocity=0.
    for t,w in zip(times,weights):
        b,v=model(t);bio=bio+w*b;velocity=velocity+w*v
    return bio.numpy(),velocity.numpy()
