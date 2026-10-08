"""Exact program decomposition and phase-portrait diagnostics for the synthetic decoder."""
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import pearsonr,spearmanr
from ._spatial_mixture_ablation import SpotAblationModel,Switches,unit_step_response
from ._trajectory_refinement import nb_log_prob

@torch.no_grad()
def kinetic_curves(model,times):
    t=torch.as_tensor(times,dtype=model.log_beta.dtype)
    b=model.log_beta.exp();g=model.log_gamma.exp();A=model.log_A.exp()
    if model.switches.gated:
        on=model.T*.85*model.on_raw.sigmoid();off=on+torch.nn.functional.softplus(model.duration_raw)+.2
        u1,s1=unit_step_response((t[:,None,None]-on).clamp_min(0),b,g)
        u0,s0=unit_step_response((t[:,None,None]-off).clamp_min(0),b,g)
        u=(u1-u0).clamp_min(0)*A;s=(s1-s0).clamp_min(0)*A
        alpha=((t[:,None,None]>=on)&(t[:,None,None]<off)).to(A.dtype)*A
    else:
        u,s=unit_step_response(t[:,None,None],b,g);u=u*A;s=s*A
        alpha=A.expand(len(t),*A.shape)
        on=off=None
    component=dict(u=u,s=s,alpha=alpha,v=b*u-g*s,du=alpha-b*u)
    if model.switches.shared:
        W=model.type_logits.softmax(-1)
        typed={k:torch.einsum('cm,nmg->ncg',W,v).numpy() for k,v in component.items()}
    else:typed={k:v.numpy() for k,v in component.items()}
    return {k:v.numpy() for k,v in component.items()},typed

@torch.no_grad()
def decompose_spots(model):
    if not model.switches.shared:raise ValueError('Program decomposition requires shared programs')
    components,typed=kinetic_curves(model,model.time_input)
    W=model.type_logits.softmax(-1).numpy()
    pi=model.compositions().numpy();q=pi@W
    contributions={k:q[:,:,None]*v for k,v in components.items()}
    direct=model()
    np.testing.assert_allclose(contributions['s'].sum(1),direct[0][...,1].numpy(),rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(contributions['u'].sum(1),direct[0][...,0].numpy(),rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(contributions['v'].sum(1),direct[1].numpy(),rtol=2e-5,atol=2e-6)
    return components,contributions,q,W

def conditional_mix(typed,pi):
    pi=np.asarray(pi,dtype=float)
    if pi.ndim!=1 or not np.isclose(pi.sum(),1):raise ValueError('Single fixed composition vector required')
    return {k:np.einsum('c,ncg->ng',pi,v) for k,v in typed.items()}

def phase_portrait(ax,g,c,obs,truth,grid,typed,models,names,colors,title=None,arrows=True):
    n=int(obs['n_spots']);ref=np.flatnonzero(obs['reference_type']==c)
    corrected=obs['counts']/obs['exposure']-obs['ambient']
    ax.scatter(corrected[ref,g,1],corrected[ref,g,0],s=15,c='#888888',alpha=.4,label='参考观测')
    order=np.argsort(truth['time'])
    ax.plot(truth['type_expression'][order,c,g,1],truth['type_expression'][order,c,g,0],c='black',lw=1.6,label='真轨迹')
    max_s=max(float(truth['type_expression'][:,c,g,1].max()),*(float(typed[a]['s'][:,c,g].max()) for a in names))
    max_u=max(float(truth['type_expression'][:,c,g,0].max()),*(float(typed[a]['u'][:,c,g].max()) for a in names))
    for a,color in zip(names,colors):
        d=typed[a];x=d['s'][:,c,g];y=d['u'][:,c,g]
        ax.plot(x,y,color=color,lw=1.8,label=names[a])
        with torch.no_grad():slope=float(models[a].log_gamma[g].exp()/models[a].log_beta[g].exp())
        xx=np.array([0,min(max_s,max_u/max(slope,1e-12))]);ax.plot(xx,slope*xx,color=color,ls=':',alpha=.5,lw=.9)
        if arrows:
            # Exact tangent (ds/dt,du/dt), NOT two copies of RNA velocity.
            ids=np.unique(np.linspace(15,len(grid)-15,6).astype(int))
            dx=d['v'][ids,c,g];dy=d['du'][ids,c,g]
            norm=np.sqrt((dx/max(max_s,1e-8))**2+(dy/max(max_u,1e-8))**2)
            valid=norm>1e-5;dt=np.divide(.055,norm,out=np.zeros_like(norm),where=valid)
            ax.quiver(x[ids][valid],y[ids][valid],(dx*dt)[valid],(dy*dt)[valid],
                      color=color,angles='xy',scale_units='xy',scale=1,width=.005,zorder=5)
    ax.set(xlabel='Spliced s',ylabel='Unspliced u',title=title or str(obs['gene_names'][g]))
    # All observed points retained; curves drawn in the same corrected model units.
    return ax

def gene_test_metrics(obs,truth,pred,arm):
    n=int(obs['n_spots']);mask=obs['test_mask'][:n];true=truth['velocity'][:n];v=pred['velocity'][:n]
    mu=(pred['biological_expression'][:n]+obs['ambient'])*obs['exposure'][:n]
    nll=-nb_log_prob(torch.tensor(obs['counts'][:n]),torch.tensor(mu),torch.tensor(obs['theta'])).numpy()
    rows=[]
    for g in range(v.shape[1]):
        ix=mask[:,g];x=true[ix,g];y=v[ix,g]
        valid=len(x)>2 and np.std(x)>1e-10 and np.std(y)>1e-10
        den=np.sum(x*x)
        rows.append(dict(arm=arm,gene=str(obs['gene_names'][g]),test_n=int(ix.sum()),
            velocity_rrmse=float(np.sqrt(np.sum((y-x)**2)/den)) if den>1e-20 else np.nan,
            velocity_pearson=float(pearsonr(x,y).statistic) if valid else np.nan,
            velocity_spearman=float(spearmanr(x,y).statistic) if valid else np.nan,
            count_nll=float(nll[ix,g].mean()) if ix.any() else np.nan))
    return pd.DataFrame(rows)

def all_gene_evaluation(run_dir,out):
    run_dir=Path(run_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(run_dir/'all_results.csv')
    tables=[]
    for (scenario,seed),sub in manifest.groupby(['scenario','seed']):
        folder=run_dir/f'{scenario}_seed{seed}'
        obs=dict(np.load(folder/'observations.npz'));truth=dict(np.load(folder/'truth.npz'))
        for arm in ['S1_G1_M1','S0_G1_M1']:
            p=dict(np.load(folder/arm/'prediction.npz'))
            tab=gene_test_metrics(obs,truth,p,arm);tab['scenario']=scenario;tab['seed']=seed;tables.append(tab)
    data=pd.concat(tables,ignore_index=True);data.to_csv(out/'all_genes_heldout_evaluation.csv',index=False)
    return data
