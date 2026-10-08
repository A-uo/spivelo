"""Differentiable exact propagation of the linear RNA kinetic system.

Matrix exponentials remain defined when beta, gamma and lambda coincide.
No rate-difference division or artificial denominator epsilon is used.
"""
import torch


def propagate_matrix(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam, chunk_size=4096):
    """Solve a(t)=alpha-delta_alpha*exp(-lam*t), u'=a-beta*u, s'=beta*u-gamma*s."""
    inputs = torch.broadcast_tensors(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam)
    shape, dtype = inputs[0].shape, inputs[0].dtype
    flat = [x.reshape(-1).to(torch.float64) for x in inputs]
    result = []
    for start in range(0, flat[0].numel(), chunk_size):
        a, b, g, t, u, s, d, l = [x[start:start + chunk_size] for x in flat]
        zero, one = torch.zeros_like(a), torch.ones_like(a)
        # State is [u, s, instantaneous transcription, constant 1].
        matrix = torch.stack([
            torch.stack([-b, zero, one, zero], -1),
            torch.stack([b, -g, zero, zero], -1),
            torch.stack([zero, zero, -l, l * a], -1),
            torch.stack([zero, zero, zero, zero], -1),
        ], -2)
        initial = torch.stack([u, s, a - d, one], -1)
        final = (torch.linalg.matrix_exp(matrix * t[:, None, None]) @ initial[..., None])[..., 0]
        result.append(final[:, :2])
    out = torch.cat(result).reshape(*shape, 2).to(dtype)
    return out[..., 0], out[..., 1]


def propagate(alpha, beta, gamma, tau, u0, s0, delta_alpha, lam, chunk_size=4096):
    """Exact exponential solution; use matrix exponential in cancellation-prone cases.

    The fast path is algebraically identical, not a time discretization. Rate collisions,
    short intervals and tiny rates retain the original matrix solution and its gradients.
    """
    inputs=torch.broadcast_tensors(alpha,beta,gamma,tau,u0,s0,delta_alpha,lam)
    shape,dtype=inputs[0].shape,inputs[0].dtype
    flat=[x.reshape(-1).to(torch.float64) for x in inputs]
    a,b,g,t,u,s,d,l=flat
    scale=torch.maximum(torch.maximum(b,g),l).clamp_min(1.)
    difficult=((b-g).abs()<1e-3*scale)|((b-l).abs()<1e-3*scale)|((g-l).abs()<1e-3*scale)
    difficult=difficult|(torch.minimum(b,g)*t<1e-3)
    easy=~difficult
    out_u=torch.zeros_like(a); out_s=torch.zeros_like(a)
    if easy.any():
        aa,bb,gg,tt,uu,ss,dd,ll=[x[easy] for x in flat]
        def convolution(r,q):
            gap=(r-q).abs()
            return torch.exp(-torch.minimum(r,q)*tt)*(-torch.expm1(-gap*tt))/gap
        cb_l=convolution(bb,ll); cg_b=convolution(gg,bb); cg_l=convolution(gg,ll)
        fb=-torch.expm1(-bb*tt)/bb; fg=-torch.expm1(-gg*tt)/gg
        out_u[easy]=uu*torch.exp(-bb*tt)+aa*fb-dd*cb_l
        out_s[easy]=ss*torch.exp(-gg*tt)+bb*uu*cg_b+aa*(fg-cg_b)-bb*dd*(cg_l-cg_b)/(bb-ll)
    if difficult.any():
        exact=propagate_matrix(*[x[difficult] for x in flat],chunk_size=chunk_size)
        out_u[difficult]=exact[0]; out_s[difficult]=exact[1]
    return out_u.reshape(shape).to(dtype),out_s.reshape(shape).to(dtype)


def soft_states(t, on, off, temperature):
    """Three ordered smooth memberships; not calibrated fate probabilities."""
    a = torch.sigmoid((t - on) / temperature)
    b = torch.sigmoid((t - off) / temperature)
    return torch.stack([1 - a, a - b, b], dim=-1)
