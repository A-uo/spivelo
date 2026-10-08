"""Spatial velocity regularization, output smoothing, and fixed-protocol evaluation."""
from pathlib import Path
import copy,json,time,hashlib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from ._spatial_mixture_ablation import SpotAblationModel,Switches
from ._trajectory_refinement import nb_log_prob
from ._hvgk_pyro_module import HVGKModuleKineticsPyroModule
from ._velocity_truth_evaluation import evaluate_velocity
from ._benchmark import cbdir_topovelo

def spatial_graph(xy,k=24):
    n=len(xy);k=min(k,n-1)
    dist,ix=NearestNeighbors(n_neighbors=k+1).fit(xy).kneighbors(xy)
    rows=[];cols=[];weights=[]
    for i in range(n):
        keep=ix[i]!=i;js=ix[i][keep][:k];d=dist[i][keep][:k]
        scale=max(float(d[-1]),1e-8)
        rows.extend([i]*len(js));cols.extend(js);weights.extend(np.exp(-.5*(d/scale)**2))
    a=sparse.csr_matrix((weights,(rows,cols)),shape=(n,n))
    a=(a+a.T)*.5;a.setdiag(0);a.eliminate_zeros()
    return sparse.diags(1/np.asarray(a.sum(1)).ravel().clip(1e-12))@a

def permute_graph(p,seed):
    ix=np.random.default_rng(seed).permutation(p.shape[0])
    return p[ix][:,ix].tocsr()

def smooth_velocity(v,p,strength=.5,passes=1):
    if not 0<=strength<=1 or passes<1:raise ValueError('Invalid smoothing')
    out=np.array(v,copy=True)
    for _ in range(passes):out=(1-strength)*out+strength*(p@out)
    return out

def torch_graph(p):
    coo=p.tocoo()
    return torch.sparse_coo_tensor(torch.tensor(np.stack([coo.row,coo.col]),dtype=torch.long),
           torch.tensor(coo.data,dtype=torch.float32),coo.shape,check_invariants=True).coalesce()

def energy(v,p):
    coo=p.tocoo()
    return float(np.sum(coo.data*np.mean((v[coo.row]-v[coo.col])**2,axis=1))/coo.data.sum())

def graph_coherence(v,p):
    coo=p.tocoo();a=v[coo.row];b=v[coo.col]
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    c=np.divide((a*b).sum(1),den,out=np.zeros(len(den)),where=den>1e-12)
    return float(np.average(c,weights=coo.data))

def refine_spatial(obs,base_state,p,weight=0.,steps=120):
    """Same checkpoint/optimizer/time/budget; exact production edge-energy function."""
    m=SpotAblationModel(obs,Switches());m.load_state_dict(base_state)
    n=int(obs['n_spots']);adj=torch_graph(p)
    with torch.no_grad():scale=m()[1][:n].square().mean().clamp_min(1e-8)
    y,ex,amb,theta=[torch.tensor(obs[k],dtype=torch.float32) for k in ['counts','exposure','ambient','theta']]
    train=torch.tensor(obs['train_mask']);val=torch.tensor(obs['val_mask'])
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    best=np.inf;state=None;history=[];start=time.time()
    for step in range(1,steps+1):
        opt.zero_grad();bio,v,_,_=m()
        nll=-nb_log_prob(y,(bio+amb)*ex,theta)
        data_loss=(nll[train].sum()+m.penalty())/train.sum()
        reg=HVGKModuleKineticsPyroModule._spatial_mse(v[:n],adj)/scale
        loss=data_loss+weight*reg
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),20);opt.step()
        if step==steps//2:
            for g in opt.param_groups:g['lr']=.005
        if step==1 or step%20==0 or step==steps:
            with torch.no_grad():
                bio,_,_,_=m();score=float(-nb_log_prob(y,(bio+amb)*ex,theta)[val].mean())
            history.append(dict(step=step,val_nll=score,edge_loss=float(reg.detach()),data_loss=float(data_loss.detach())))
            if score<best:best=score;best_step=step;state=copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    with torch.no_grad():v=m()[1][:n].numpy()
    return v,state,pd.DataFrame(history),dict(best_step=best_step,val_nll=best,seconds=time.time()-start,
                                              energy_normalization=float(scale))

def protocols(obs,truth):
    n=int(obs['n_spots'])
    x=obs['counts'][:n,:,1]/obs['exposure'][:n,:,1]-obs['ambient'][...,1]
    z=PCA(n_components=min(15,x.shape[1]-1),svd_solver='full').fit_transform(np.log1p(np.maximum(x,0)))
    indices={name:NearestNeighbors(n_neighbors=min(k,n)).fit(rep).kneighbors(rep,return_distance=False)
             for name,rep,k in [('expression25',z,25),('spatial50',obs['xy'],50)]}
    return x,indices

def measure(v,base,obs,truth,eval_graph,x,indices):
    n=int(obs['n_spots']);t=truth['velocity'][:n]
    stats,spots,_=evaluate_velocity(v,t,obs['test_mask'][:n])
    den=np.linalg.norm(v,axis=1)*np.linalg.norm(base,axis=1)
    stats['cosine_to_unsmoothed']=float(np.mean(np.divide((v*base).sum(1),den,out=np.zeros(n),where=den>1e-12)))
    stats['spatial_neighbor_coherence']=graph_coherence(v,eval_graph)
    stats['edge_energy_ratio']=energy(v,eval_graph)/max(energy(base,eval_graph),1e-12)
    stats['between_spot_variance_ratio']=float(v.var(0).sum()/max(base.var(0).sum(),1e-12))
    # Truth composition defines evaluation strata only; never informs graph or fitting.
    coo=eval_graph.tocoo();pi=truth['composition'][:n]
    gradient=np.bincount(coo.row,weights=coo.data*np.abs(pi[coo.row]-pi[coo.col]).sum(1),minlength=n)
    high=gradient>=np.quantile(gradient,.75)
    stats['high_composition_gradient_cosine']=float(spots.loc[high,'cosine'].mean())
    stats['high_composition_gradient_rmse']=float(np.sqrt(((v[high]-t[high])**2).sum()/(t[high]**2).sum()))
    cbrows=[]
    for name,ix in indices.items():
        cb,_=cbdir_topovelo(x,v,ix,truth['age_stage'],[('early','middle'),('middle','late')])
        stats['CBDir_'+name]=float(cb.CBDir.mean());stats['covered_'+name]=int(cb.CBDir.notna().sum())
        cb['protocol']=name;cbrows.append(cb)
    return stats,pd.concat(cbrows,ignore_index=True)

def run_smoothing_experiment(run_dir,out,steps=120,seeds=(11,29,47),scenarios=('shared_mismatch','independent_dynamics'),callback=None):
    run_dir=Path(run_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    config=dict(steps=steps,seeds=list(seeds),scenarios=list(scenarios),k_values=[8,24],strengths=[.25,.5,1.],
       training_weights=[0.,.1,1.],truth_used_for_fit=False,
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       metric_sha256=hashlib.sha256(Path(__file__).with_name('_velocity_truth_evaluation.py').read_bytes()).hexdigest())
    config['input_hashes']={str(f.relative_to(run_dir)):hashlib.sha256(f.read_bytes()).hexdigest()
        for scenario in scenarios for seed in seeds
        for f in [run_dir/f'{scenario}_seed{seed}'/rel for rel in
            ['observations.npz','truth.npz','S1_G1_M1/model_state.pt','S1_G1_M1/prediction.npz','S0_G1_M1/prediction.npz']]}
    cf=out/'config.json'
    if cf.exists() and json.loads(cf.read_text())!=config:raise ValueError('New config/code requires a new output directory')
    cf.write_text(json.dumps(config,indent=2),encoding='utf-8')
    records=[];transitions=[]
    for scenario in scenarios:
        for seed in seeds:
            src=run_dir/f'{scenario}_seed{seed}';dest=out/src.name;dest.mkdir(exist_ok=True)
            obs=dict(np.load(src/'observations.npz'));truth=dict(np.load(src/'truth.npz'));n=int(obs['n_spots'])
            graphs={k:spatial_graph(obs['xy'],k) for k in [8,24]}
            shuffled=permute_graph(graphs[24],seed+8000);x,indices=protocols(obs,truth)
            def record(v,base,arm,label,mode,strength=0.,k=24,passes=1,extra=None):
                stats,cb=measure(v,base,obs,truth,graphs[24],x,indices)
                row=dict(scenario=scenario,seed=seed,backbone=arm,condition=label,mode=mode,
                         strength=strength,k=k,passes=passes,**stats)
                if extra:row.update(extra)
                records.append(row)
                cb['scenario']=scenario;cb['seed']=seed;cb['backbone']=arm;cb['condition']=label;transitions.append(cb)
                np.save(dest/(arm+'__'+label+'.npy'),v)
                pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
            for arm in ['S1_G1_M1','S0_G1_M1']:
                base=dict(np.load(src/arm/'prediction.npz'))['velocity'][:n]
                record(base,base,arm,'raw','raw')
                for k,p in graphs.items():
                    for rho in [.25,.5,1.]:
                        record(smooth_velocity(base,p,rho),base,arm,f'post_k{k}_a{rho:g}','post',rho,k)
                    record(smooth_velocity(base,p,.5,4),base,arm,f'post_k{k}_a0.5_repeat4','post',.5,k,4)
                record(smooth_velocity(base,shuffled,.5),base,arm,'post_shuffled_a0.5','shuffled_post',.5)
            arm='S1_G1_M1';base=dict(np.load(src/arm/'prediction.npz'))['velocity'][:n]
            state=torch.load(src/arm/'model_state.pt',map_location='cpu',weights_only=True)
            for label,weight,p in [('train_l0',0.,graphs[24]),('train_l0.1',.1,graphs[24]),('train_l1',1.,graphs[24]),('train_shuffled_l1',1.,shuffled)]:
                model_file=dest/(label+'.pt');meta_file=dest/(label+'.json')
                if model_file.exists() and meta_file.exists():
                    saved=torch.load(model_file,map_location='cpu',weights_only=True)
                    m=SpotAblationModel(obs,Switches());m.load_state_dict(saved)
                    with torch.no_grad():v=m()[1][:n].numpy()
                    info=json.loads(meta_file.read_text())
                else:
                    v,saved,history,info=refine_spatial(obs,state,p,weight,steps)
                    torch.save(saved,model_file);meta_file.write_text(json.dumps(info),encoding='utf-8')
                    history.to_csv(dest/(label+'_history.csv'),index=False)
                record(v,base,arm,label,'shuffled_train' if 'shuffled' in label else 'train',weight,extra=info)
            record(truth['velocity'][:n],base,'truth','truth','truth')
            if callback:callback(dict(scenario=scenario,seed=seed,completed_rows=len(records)))
    table=pd.DataFrame(records);pd.concat(transitions,ignore_index=True).to_csv(out/'cbdir_transitions.csv',index=False)
    return table
