"""Controlled spatial-mixture mechanism experiment, separate from production HVGK.

S: shared temporal programs versus independent type/gene kinetics.
G: pulse gates versus constitutive transcription (time itself is retained).
M: spot-specific cell-type fractions versus a learned global composition.
Type/program participation W is distinct from spot composition pi.
"""
from dataclasses import dataclass,asdict
from itertools import product
from pathlib import Path
import json,time,copy,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.integrate import solve_ivp
from scipy.special import softmax,expit
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF,PCA
from sklearn.neighbors import NearestNeighbors
from ._trajectory_refinement import nb_log_prob
from ._benchmark import cbdir_topovelo

TYPE_NAMES=['RG_like','IPC_like','Deep_EX_like','Upper_EX_like','IN_like']


@dataclass(frozen=True)
class Switches:
    shared: bool=True
    gated: bool=True
    mixture: bool=True

    @property
    def name(self):return f'S{int(self.shared)}_G{int(self.gated)}_M{int(self.mixture)}'


def factorial_switches():return [Switches(*x) for x in product([False,True],repeat=3)]


def simulate_spots(seed=11,n_spots=720,n_genes=80,reference_per_type=24,scenario='shared_mismatch'):
    """Brain-inspired artificial tissue; no human samples or empirical calibration.

    Truth is numerically integrated from smooth, irregular transcription curves;
    fitted models instead use exact rectangular-pulse responses. Reference rows
    are independently sampled, type-labelled pure calibration measurements.
    """
    if n_spots<40 or n_genes<16 or reference_per_type<4:raise ValueError('Insufficient simulation dimensions')
    if scenario not in ('shared_mismatch','independent_dynamics'):raise ValueError('Unknown scenario')
    rng=np.random.default_rng(seed);C=5;K=8;T=12.
    gene_group=np.arange(n_genes)%K;rng.shuffle(gene_group)
    A=rng.lognormal(.9,.5,(K,n_genes))*(.10+2.7*(np.arange(K)[:,None]==gene_group))
    W=np.array([[3,.8,.2,.1,.1,.4,.1,.3],[.8,3,1,.3,.2,.1,.1,.2],
        [.2,.6,1.4,3,.25,.1,.1,.3],[.2,.6,1.2,.2,3,.1,.1,.4],
        [.3,.15,.5,.1,.1,2.5,3,.4]],float)
    W=W/W.sum(1,keepdims=True)
    centers=np.array([1.4,3.8,5.6,7.4,9.0,2.8,8.0,6.8])
    widths=np.array([1.6,1.3,1.5,1.7,1.4,1.5,2.,1.0])
    beta=rng.lognormal(np.log(.8),.20,n_genes);gamma=rng.lognormal(np.log(.42),.22,n_genes)
    # Small independent component prevents the shared case from exactly matching the model.
    amp=rng.lognormal(.5,.5,(C,n_genes))
    independent_center=rng.uniform(.8,10.5,(C,n_genes))
    independent_width=rng.uniform(.65,2.5,(C,n_genes))
    def transcription(t):
        h=np.exp(-.5*((t-centers)/widths)**2)*(1+.22*np.sin(2*t+np.arange(K)))
        h+=.16*np.exp(-.5*((t-centers-1.8)/(.7*widths))**2)
        shared=(W*h[None,:])@A
        independent=amp*(np.exp(-.5*((t-independent_center)/independent_width)**2)
            +.30*np.exp(-.5*((t-independent_center+2.)/(.6*independent_width))**2))
        return .82*shared+.18*independent if scenario=='shared_mismatch' else independent
    def rhs(t,y):
        u,s=y.reshape(2,C,n_genes)
        return np.stack([transcription(t)-beta*u,beta*u-gamma*s]).reshape(-1)
    sol=solve_ivp(rhs,[0,T],np.zeros(2*C*n_genes),rtol=2e-8,atol=1e-10,dense_output=True)
    if not sol.success:raise RuntimeError(sol.message)
    age=rng.choice([2.,5.5,9.4],n_spots,p=[.30,.30,.40])+rng.normal(0,1.05,n_spots)
    age=np.clip(age,.35,11.65)
    area=rng.integers(0,3,n_spots);column=rng.uniform(-1.15,1.15,n_spots)
    depth=.12+.75*expit((age-5.5)/1.9)+rng.normal(0,.08,n_spots)
    radius=2.1+1.8*depth+.18*np.sin(5*column)
    xy=np.stack([radius*np.sin(column)+.34*np.sin(4*depth)+area*.25,
                 radius*np.cos(column)+.26*np.cos(3*column)+area*.6],1)
    # Two curved tissue domains, crossing composition gradients and local niches.
    island=rng.random(n_spots)<.13;xy[island]+=np.array([2.0,-.9])
    niche=np.exp(-((column-.45)**2/.12+(depth-.55)**2/.045))
    logits=np.stack([2.2-.38*age-1.3*depth,
        1.9-((age-4.3)/2.6)**2+.7*np.cos(4*column),
        -.4+.22*age-1.3*column+.5*(area==0),
        -1.1+.29*age+1.1*column+.5*(area==1),
        -.1+1.4*np.sin(3*column)+2*niche+.5*(area==2)],1)
    pi0=softmax(logits,axis=1)
    pi=np.stack([rng.dirichlet(.25+18*p) for p in pi0])
    pure=rng.random(n_spots)<.12
    pi[pure]=.03/C+.97*np.eye(C)[rng.integers(C,size=pure.sum())]
    donor=rng.integers(0,3,n_spots)
    ref_type=np.repeat(np.arange(C),reference_per_type)
    ref_age=np.tile(np.linspace(.6,11.4,reference_per_type),C)+rng.normal(0,.12,len(ref_type))
    all_age=np.r_[age,ref_age]
    type_state=sol.sol(all_age).T.reshape(len(all_age),2,C,n_genes).transpose(0,2,3,1)
    mixture=np.concatenate([pi,np.eye(C)[ref_type]])
    bio=np.einsum('nc,ncga->nga',mixture,type_state)
    velocity=np.einsum('nc,ncg->ng',mixture,beta*type_state[...,0]-gamma*type_state[...,1])
    exposure=rng.lognormal(-.55,.55,len(all_age))[:,None,None]*np.array([.7,1.])[None,None,:]
    exposure[:n_spots]*=np.array([.85,1.,1.2])[donor,None,None]
    ambient=rng.uniform(.01,.045,(1,n_genes,2))
    theta=np.full((1,n_genes,2),8.)
    mu=(bio+ambient)*exposure
    counts=rng.negative_binomial(theta,theta/(theta+mu)).astype(np.float32)
    proxy=np.clip(all_age+rng.normal(0,.40,len(all_age)),.15,T-.15)
    partition=rng.random((len(all_age),n_genes))
    train=partition<.80;val=(partition>=.80)&(partition<.90);test=partition>=.90
    train[n_spots:]=True;val[n_spots:]=False;test[n_spots:]=False
    obs=dict(counts=counts,exposure=exposure.astype(np.float32),ambient=ambient.astype(np.float32),
        theta=theta.astype(np.float32),time_proxy=proxy.astype(np.float32),
        reference_type=np.r_[np.full(n_spots,-1),ref_type],n_spots=np.array(n_spots),
        train_mask=train,val_mask=val,test_mask=test,horizon=np.array(T),
        xy=xy.astype(np.float32),gene_names=np.array([f'SYN_G{g:03d}' for g in range(n_genes)]))
    truth=dict(time=all_age.astype(np.float32),composition=mixture.astype(np.float32),
        biological_expression=bio.astype(np.float32),velocity=velocity.astype(np.float32),
        type_expression=type_state.astype(np.float32),beta=beta.astype(np.float32),gamma=gamma.astype(np.float32),
        program_loadings=A.astype(np.float32),type_program_weights=W.astype(np.float32),
        age_stage=np.where(age<3.5,'early',np.where(age<7.5,'middle','late')),
        donor=donor,area=area,scenario=np.array(scenario),seed=np.array(seed))
    return obs,truth


def unit_step_response(t,beta,gamma):
    """RNA response to unit constant transcription from zero RNA; stable equal rates."""
    dtype=t.dtype;t,b,g=torch.broadcast_tensors(t.double(),beta.double(),gamma.double())
    z=(b-g).abs()*t
    phi=torch.where(z<1e-4,1-z/2+z*z/6-z*z*z/24,-torch.expm1(-z)/z.clamp_min(1e-12))
    conv=torch.exp(-torch.minimum(b,g)*t)*t*phi
    u=-torch.expm1(-b*t)/b
    s=-torch.expm1(-g*t)/g-conv
    return u.to(dtype),s.clamp_min(0).to(dtype)


def masked_initialization(obs,C,K):
    """Only training entries initialize dynamics; validation/test counts are excluded."""
    mask=obs['train_mask'];counts=obs['counts']
    x=np.maximum(counts/obs['exposure']-obs['ambient'],0)
    x=x.mean(-1)
    means=(x*mask).sum(0)/np.maximum(mask.sum(0),1)
    filled=np.where(mask,x,means)
    labels=obs['reference_type'];profiles=np.stack([filled[labels==c].mean(0) for c in range(C)])
    # NMF sees calibration reference means only, not spot compositions or truth.
    nf=NMF(n_components=min(K,C),init='random',random_state=0,max_iter=500)
    w=nf.fit_transform(np.maximum(profiles,.01));a=nf.components_
    if K>C:
        a=np.vstack([a, np.repeat(a.mean(0)[None,:],K-C,axis=0)])
        w=np.c_[w,np.full((C,K-C),.05)]
    a=np.maximum(a,.01);w=np.maximum(w,.01);w/=w.sum(1,keepdims=True)
    return profiles,a,w


class SpotAblationModel(nn.Module):
    """Conditional-time penalized likelihood model for the factorial experiment.

    π_i,c mixes complete type-specific RNA solutions. It never averages switch
    times. W_c,m is a separate shared-program participation matrix.
    """
    def __init__(self,obs,switches,n_programs=8):
        super().__init__();self.switches=switches
        self.C=int(obs['reference_type'].max())+1;self.G=obs['counts'].shape[1]
        self.N=int(obs['n_spots']);self.K=n_programs;self.T=float(obs['horizon'])
        profiles,a,w=masked_initialization(obs,self.C,self.K)
        self.register_buffer('time_input',torch.tensor(obs['time_proxy']))
        self.register_buffer('ref_types',torch.tensor(obs['reference_type'][self.N:],dtype=torch.long))
        self.log_beta=nn.Parameter(torch.full((self.G,),np.log(.8)))
        self.log_gamma=nn.Parameter(torch.full((self.G,),np.log(.42)))
        if switches.shared:
            self.log_A=nn.Parameter(torch.tensor(np.log(a*.6),dtype=torch.float32))
            self.type_logits=nn.Parameter(torch.tensor(np.log(w),dtype=torch.float32))
            shape=(self.K,1)
        else:
            self.log_A=nn.Parameter(torch.tensor(np.log(np.maximum(profiles*.6,.01)),dtype=torch.float32))
            shape=(self.C,self.G)
        if switches.gated:
            # Same time coverage convention in every arm, without true switch times.
            centers=np.linspace(.015,.20,int(np.prod(shape))).reshape(shape)
            self.on_raw=nn.Parameter(torch.tensor(np.log(centers/(1-centers)),dtype=torch.float32))
            self.duration_raw=nn.Parameter(torch.full(shape,5.0))
        self.composition_logits=nn.Parameter(torch.zeros((self.N if switches.mixture else 1,self.C)))

    def compositions(self):
        pi=torch.softmax(self.composition_logits,-1)
        if not self.switches.mixture:pi=pi.expand(self.N,-1)
        ref=torch.nn.functional.one_hot(self.ref_types,num_classes=self.C).to(pi.dtype)
        return torch.cat([pi,ref])

    def forward(self,time_input=None):
        t=self.time_input if time_input is None else time_input
        beta=self.log_beta.exp();gamma=self.log_gamma.exp()
        if self.switches.gated:
            on=self.T*.85*torch.sigmoid(self.on_raw)
            off=on+torch.nn.functional.softplus(self.duration_raw)+.2
            up,sp=unit_step_response((t[:,None,None]-on).clamp_min(0),beta,gamma)
            um,sm=unit_step_response((t[:,None,None]-off).clamp_min(0),beta,gamma)
            u=(up-um).clamp_min(0);s=(sp-sm).clamp_min(0)
        else:
            u,s=unit_step_response(t[:,None,None],beta,gamma)
        u=u*self.log_A.exp();s=s*self.log_A.exp()
        if self.switches.shared:
            W=torch.softmax(self.type_logits,dim=-1)
            u=torch.einsum('cm,nmg->ncg',W,u);s=torch.einsum('cm,nmg->ncg',W,s)
        pi=self.compositions()
        bio=torch.stack([torch.einsum('nc,ncg->ng',pi,u),torch.einsum('nc,ncg->ng',pi,s)],-1)
        velocity=torch.einsum('nc,ncg->ng',pi,beta*u-gamma*s)
        return bio,velocity,pi,torch.stack([u,s],-1)

    def penalty(self):
        # Parameter shrinkage; no spatial or temporal neighbor smoothing.
        value=.5*((self.log_beta-np.log(.8)).square().sum()+(self.log_gamma-np.log(.42)).square().sum())
        value=value+.02*(self.composition_logits-self.composition_logits.mean(-1,keepdim=True)).square().sum()
        return value


def fit_ablation(obs,switches,steps=240,seed=101,callback=None):
    """Validation selects training step; the test mask and truth never enter fitting."""
    torch.manual_seed(seed);model=SpotAblationModel(obs,switches)
    y,scale,ambient,theta=[torch.tensor(obs[k],dtype=torch.float32) for k in ['counts','exposure','ambient','theta']]
    train=torch.tensor(obs['train_mask']);val=torch.tensor(obs['val_mask'])
    opt=torch.optim.Adam(model.parameters(),lr=.035)
    best=np.inf;state=None;history=[];best_step=0
    start=time.time()
    for step in range(1,steps+1):
        opt.zero_grad();bio,_,_,_=model()
        loss=-nb_log_prob(y,(bio+ambient)*scale,theta)[train].sum()
        objective=(loss+model.penalty())/train.sum()
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite objective')
        objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),20.);opt.step()
        if step==steps//2:
            for group in opt.param_groups:group['lr']=.012
        if step==1 or step%20==0 or step==steps:
            with torch.no_grad():
                bio,_,_,_=model()
                score=float(-nb_log_prob(y,(bio+ambient)*scale,theta)[val].mean())
            row=dict(step=step,train_nll_per_entry=float(loss.detach())/(int(train.sum())*2),val_nll=score)
            history.append(row)
            if score<best:best=score;state=copy.deepcopy(model.state_dict());best_step=step
            if callback:callback(row)
    model.load_state_dict(state)
    return model,pd.DataFrame(history),dict(best_step=best_step,val_nll=best,seconds=time.time()-start,
        parameters=sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def evaluate_ablation(model,obs,truth):
    bio,v,pi,type_bio=model();bio=bio.numpy();v=v.numpy();pi=pi.numpy();type_bio=type_bio.numpy()
    N=int(obs['n_spots']);test=obs['test_mask']
    mu=(bio+obs['ambient'])*obs['exposure']
    nll=-nb_log_prob(torch.tensor(obs['counts']),torch.tensor(mu),torch.tensor(obs['theta'])).numpy()
    denom=np.linalg.norm(v[:N],axis=1)*np.linalg.norm(truth['velocity'][:N],axis=1)
    valid=denom>1e-10
    cosine=np.divide((v[:N]*truth['velocity'][:N]).sum(1),denom,out=np.zeros(N),where=valid)
    ref=truth['biological_expression']
    error=(v-truth['velocity'])**2
    result=dict(test_nll=float(nll[test].mean()),velocity_cosine=float(cosine[valid].mean()),
        velocity_test_relative_rmse=float(np.sqrt(error[test].mean()/max((truth['velocity'][test]**2).mean(),1e-10))),
        expression_test_relative_rmse=float(np.sqrt(((bio-ref)**2)[test].mean()/max((ref[test]**2).mean(),1e-10))),
        composition_mae=float(np.abs(pi[:N]-truth['composition'][:N]).mean()),
        dominant_type_accuracy=float((pi[:N].argmax(1)==truth['composition'][:N].argmax(1)).mean()),
        fraction_valid_velocity=float(valid.mean()))
    # Same graph for every arm; true spot RNA velocity is also scored as a diagnostic.
    x=obs['counts'][:N,:,1]/obs['exposure'][:N,:,1]-obs['ambient'][...,1]
    embedding=PCA(n_components=min(15,obs['counts'].shape[1]-1),svd_solver='full').fit_transform(np.log1p(np.maximum(x,0)))
    ix=NearestNeighbors(n_neighbors=min(25,N)).fit(embedding).kneighbors(embedding,return_distance=False)
    edges=[('early','middle'),('middle','late')]
    cb,_=cbdir_topovelo(x,v[:N],ix,truth['age_stage'],edges)
    oracle,_=cbdir_topovelo(x,truth['velocity'][:N],ix,truth['age_stage'],edges)
    result['gene_CBDir']=float(cb.CBDir.mean());result['true_velocity_gene_CBDir']=float(oracle.CBDir.mean())
    result['covered_transitions']=int(cb.CBDir.notna().sum())
    entropy=-(truth['composition'][:N]*np.log(truth['composition'][:N].clip(1e-10))).sum(1)
    high=entropy>=np.quantile(entropy,.75)
    result['high_mixture_velocity_cosine']=float(cosine[high&valid].mean())
    return result,dict(biological_expression=bio,velocity=v,composition=pi,type_expression=type_bio),cb


def run_factorial(output,seeds=(11,29,47),scenarios=('shared_mismatch','independent_dynamics'),
    n_spots=720,n_genes=80,reference_per_type=24,steps=240,resume=True,callback=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    config=dict(seeds=list(seeds),scenarios=list(scenarios),n_spots=n_spots,n_genes=n_genes,
        reference_per_type=reference_per_type,steps=steps,time_input='noisy_age_metadata',
        estimator='conditional_penalized_likelihood',known_observation_offsets=True,
        spatial_smoothing=False,truth_used_for_fitting=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config_file=output/'config.json'
    if config_file.exists() and json.loads(config_file.read_text())!=config:raise ValueError('Use a new directory for changed configuration')
    config_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[]
    for scenario in scenarios:
        for seed in seeds:
            folder=output/f'{scenario}_seed{seed}';folder.mkdir(exist_ok=True)
            obs,truth=simulate_spots(seed,n_spots,n_genes,reference_per_type,scenario)
            np.savez_compressed(folder/'observations.npz',**obs);np.savez_compressed(folder/'truth.npz',**truth)
            for switches in factorial_switches():
                arm=folder/switches.name;arm.mkdir(exist_ok=True)
                if resume and (arm/'metrics.json').exists():
                    result=json.loads((arm/'metrics.json').read_text())
                else:
                    model,history,fitinfo=fit_ablation(obs,switches,steps=steps,seed=seed+1000)
                    result,pred,cb=evaluate_ablation(model,obs,truth)
                    result.update(**asdict(switches),**fitinfo,arm=switches.name,scenario=scenario,seed=seed)
                    torch.save(model.state_dict(),arm/'model_state.pt')
                    history.to_csv(arm/'training.csv',index=False);cb.to_csv(arm/'cbdir.csv',index=False)
                    np.savez_compressed(arm/'prediction.npz',**pred)
                    (arm/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                rows.append(result)
                pd.DataFrame(rows).to_csv(output/'all_results.csv',index=False)
                if callback:callback(result)
    table=pd.DataFrame(rows)
    table.groupby(['scenario','arm'])[['test_nll','velocity_cosine','composition_mae','expression_test_relative_rmse']].agg(['mean','std']).to_csv(output/'summary.csv')
    return table
