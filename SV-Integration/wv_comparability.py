"""Three-way masked H ablation and native W/V geometry diagnostics."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
if __package__:
    from .mapping_optimizer import NMFModel
    from .masked_programs import make_masks, evaluate_entries
else:
    from mapping_optimizer import NMFModel
    from masked_programs import make_masks, evaluate_entries

MODES=('shared','separate','none')


def masked_cosine_loss(pred,target,mask):
    y=torch.where(mask,target,torch.zeros_like(target))
    p=torch.where(mask,pred,torch.zeros_like(pred))
    scores=[]
    for dim in (0,1):
        valid=y.square().sum(dim)>0
        if valid.any():scores.append(F.cosine_similarity(p,y,dim=dim)[valid].mean())
    return 1-torch.stack(scores).mean() if scores else pred.sum()*0


class ThreeWayModel(NMFModel):
    def __init__(self,sc,sp,ms,mg,k,mode,device='cpu'):
        self.mode=mode
        s=np.where(ms,sc,0).astype(np.float32);g=np.where(mg,sp,0).astype(np.float32)
        if not np.isfinite(s).all() or not np.isfinite(g).all() or (s<0).any() or (g<0).any():
            raise ValueError('Invalid observed training counts')
        super().__init__(s,g,np.ones((len(s),1),np.float32),np.full(len(g),1/len(g),np.float32),
                         torch.device(device),reconstruction_mode=mode,n_programs=k,type_prior_weight=0.)
        self.register_buffer('ms',torch.as_tensor(ms,dtype=torch.bool,device=device))
        self.register_buffer('mg',torch.as_tensor(mg,dtype=torch.bool,device=device))

    def objective(self):
        w=self._factor_W();v=F.softplus(self._V)
        m=F.softmax(w@v.T,dim=1)
        # Observed-only weighted source mean times inferred spot mass. With no
        # masking this is exactly M.T @ S. Hidden values are never accessed.
        numerator=m.T@self.S;denominator=m.T@self.ms.float()
        aggregate=numerator/denominator.clamp_min(1e-8)*m.sum(0)[:,None]
        keep=self.mg&(denominator>1e-8)
        agg=masked_cosine_loss(aggregate,self.G,keep)
        density=F.kl_div(torch.log(m.sum(0)/len(m)+1e-8),self.d,reduction='sum')
        ls=lg=self.S.new_zeros(())
        if self.mode!='none':
            h=F.softplus(self._H)
            hs=F.softplus(self._H_sp) if self.mode=='separate' else h
            ls=masked_cosine_loss(w@h,self.S,self.ms)
            lg=masked_cosine_loss(v@hs,self.G,self.mg)
        return ls+lg+agg+density,m,dict(reconstruction_sc=ls,reconstruction_sp=lg,aggregation=agg,density=density)


def cross_predict(mapping,source,source_mask,target,target_mask):
    """Query-by-source weights; calibrate each gene using target TRAINING entries only."""
    obs=np.where(source_mask,source,0)
    denom=mapping@source_mask.astype(float)
    raw=np.divide(mapping@obs,denom,out=np.zeros((len(mapping),source.shape[1])),where=denom>1e-12)
    target_obs=np.where(target_mask,target,0)
    scale=(target_obs.sum(0)+1e-8)/((raw*target_mask).sum(0)+1e-8)
    return raw*scale,scale


def fit_three_way(sc,sp,ms,mg,*,k,mode,seed=0,epochs=150,learning_rate=.1,device='cpu'):
    if mode not in MODES or epochs<1:raise ValueError('Choose shared/separate/none and positive epochs')
    torch.manual_seed(seed)
    model=ThreeWayModel(sc,sp,ms,mg,k,mode,device)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    history=[]
    for epoch in range(epochs):
        optimizer.zero_grad();loss,_,terms=model.objective()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();optimizer.step()
        history.append(dict(epoch=epoch,total=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()}))
    with torch.no_grad():_,m,_=model.objective()
    m=m.cpu().numpy();f=model.export_factors()
    # This COMMON prediction route is available even when there is no decoder H.
    ps,cs=cross_predict(m,sp,mg,sc,ms)
    pg,cg=cross_predict(m.T,sc,ms,sp,mg)
    f.update(M=m,predicted_sc=ps,predicted_sp=pg,calibration_sc=cs,calibration_sp=cg)
    return f,pd.DataFrame(history)


def normalized_coefficients(w,v):
    return w/np.maximum(w.sum(1,keepdims=True),1e-12),v/np.maximum(v.sum(1,keepdims=True),1e-12)


def geometry_metrics(w,v,m,reference_types,true_composition,reference_state=None,spatial_state=None,k_neighbors=15):
    """Truth labels are used ONLY after fitting. No Procrustes or label-based alignment."""
    w,v=normalized_coefficients(w,v)
    def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    similarity=unit(v)@unit(w).T
    ids=np.argsort(-similarity,axis=1,kind='stable')[:,:min(k_neighbors,len(w))]
    t=np.eye(true_composition.shape[1])[reference_types]
    neighbors=t[ids].mean(1)
    mapped=m.T@t;mapped/=np.maximum(mapped.sum(1,keepdims=True),1e-12)
    wc=np.stack([w[reference_types==j].mean(0) for j in range(t.shape[1])])
    vc=true_composition.T@v/np.maximum(true_composition.sum(0)[:,None],1e-12)
    centroid=unit(wc)@unit(vc).T
    diag=np.diag(centroid).mean();off=centroid[~np.eye(len(centroid),dtype=bool)].mean()
    z=np.vstack([w,v]);singular=np.linalg.svd(z-z.mean(0),compute_uv=False);eig=singular**2
    metrics=dict(latent_neighbor_composition_rmse=float(np.sqrt(np.mean((neighbors-true_composition)**2))),
                 mapping_composition_rmse=float(np.sqrt(np.mean((mapped-true_composition)**2))),
                 centroid_same_minus_other=float(diag-off),
                 centroid_correct_type_fraction=float(np.mean(centroid.argmax(1)==np.arange(len(centroid)))),
                 effective_centered_dimension=float(eig.sum()**2/max(float((eig**2).sum()),1e-12)),
                 mean_W_row_norm=float(np.linalg.norm(w,axis=1).mean()),mean_V_row_norm=float(np.linalg.norm(v,axis=1).mean()))
    if reference_state is not None and spatial_state is not None:
        predicted=reference_state[ids].mean(1)
        metrics['latent_neighbor_state_pcc']=float(np.corrcoef(predicted,spatial_state)[0,1]) if predicted.std()>0 and spatial_state.std()>0 else np.nan
    return metrics,centroid,neighbors,mapped


def run_three_way(sc,sp,out,*,candidates=(2,4,6,8),mask_seeds=(31,),seeds=(0,1),epochs=150,
                  learning_rate=.1,validation_fraction=.15,test_fraction=.15,tolerance=.005,
                  device='cpu',rerun=False):
    sc,sp=np.asarray(sc,np.float32),np.asarray(sp,np.float32)
    if sc.ndim!=2 or sp.ndim!=2 or sc.shape[1]!=sp.shape[1]:raise ValueError('Aligned gene matrices required')
    candidates=sorted(set(int(k) for k in candidates))
    if not candidates or candidates[0]<1 or not seeds or not mask_seeds or tolerance<0:raise ValueError('Invalid configuration')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=dict(candidates=candidates,mask_seeds=list(mask_seeds),seeds=list(seeds),epochs=epochs,
             learning_rate=learning_rate,validation_fraction=validation_fraction,test_fraction=test_fraction,
             tolerance=tolerance,device=str(device),type_prior_weight=0.,density='uniform',
             selection='smallest K within tolerance of mean validation log-RMSE across all three modes, both modalities and seeds',
             data_sha256=hashlib.sha256(sc.tobytes()+sp.tobytes()).hexdigest(),
             code_sha256={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                          for p in ['wv_comparability.py','mapping_optimizer.py','masked_programs.py']})
    names=['validation_scores','selection','test_metrics','training_history']
    if (out/'run.json').exists() and not rerun:
        if json.loads((out/'run.json').read_text())!=cfg:raise ValueError('Cache differs; choose new OUT or RERUN=True')
        return {k:pd.read_csv(out/f'{k}.csv') for k in names}
    if (out/'run.json').exists():(out/'run.json').unlink()
    masks={};validation=[];selection=[]
    for mask_seed in mask_seeds:
        a=make_masks(sc.shape,mask_seed,validation_fraction,test_fraction)
        b=make_masks(sp.shape,mask_seed+100003,validation_fraction,test_fraction)
        masks[mask_seed]=(a,b)
        np.savez_compressed(out/f'masks_{mask_seed}.npz',**{f'sc_{k}':v for k,v in a.items()},**{f'sp_{k}':v for k,v in b.items()})
        for k in candidates:
            for mode in MODES:
                for seed in seeds:
                    f,_=fit_three_way(sc,sp,a['train'],b['train'],k=k,mode=mode,seed=seed,epochs=epochs,learning_rate=learning_rate,device=device)
                    for modality,x,p,mask in [('single_cell',sc,f['predicted_sc'],a['validation']),('spatial',sp,f['predicted_sp'],b['validation'])]:
                        score=evaluate_entries(x,p,mask)
                        validation.append(dict(mask_seed=mask_seed,K=k,mode=mode,seed=seed,modality=modality,log_rmse=score['log_rmse']))
            print(f'Mask {mask_seed}: all three modes validation K={k} finished',flush=True)
        frame=pd.DataFrame(validation);curve=frame[frame.mask_seed==mask_seed].groupby('K').log_rmse.mean()
        chosen=int(curve[curve<=curve.min()+tolerance].index.min())
        selection.append(dict(mask_seed=mask_seed,K=chosen,best_at_boundary=int(curve.idxmin()) in [min(candidates),max(candidates)]))
    selection=pd.DataFrame(selection);validation=pd.DataFrame(validation)
    selection.to_csv(out/'selection.csv',index=False);validation.to_csv(out/'validation_scores.csv',index=False)
    (out/'selection_frozen.json').write_text(json.dumps(dict(config=cfg,selected=selection.to_dict('records')),indent=2))
    rows=[];history=[]
    for mask_seed in mask_seeds:
        a,b=masks[mask_seed];k=int(selection.loc[selection.mask_seed==mask_seed,'K'].iloc[0])
        for mode in MODES:
            for seed in seeds:
                f,h=fit_three_way(sc,sp,~a['test'],~b['test'],k=k,mode=mode,seed=seed,epochs=epochs,learning_rate=learning_rate,device=device)
                folder=out/f'mask_{mask_seed}'/mode/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'fit.npz',**f)
                history.append(h.assign(mask_seed=mask_seed,mode=mode,seed=seed,K=k))
                for modality,x,p,mask in [('single_cell',sc,f['predicted_sc'],a['test']),('spatial',sp,f['predicted_sp'],b['test'])]:
                    rows.append(dict(mask_seed=mask_seed,mode=mode,seed=seed,K=k,modality=modality,**evaluate_entries(x,p,mask)))
        print(f'Mask {mask_seed}: three-way final fits finished',flush=True)
    tables=dict(validation_scores=validation,selection=selection,test_metrics=pd.DataFrame(rows),training_history=pd.concat(history,ignore_index=True))
    for key,frame in tables.items():frame.to_csv(out/f'{key}.csv',index=False)
    (out/'run.json').write_text(json.dumps(cfg,indent=2))
    return tables
