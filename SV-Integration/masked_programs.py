"""Entry-masked two-modality dictionary diagnostics, with observed-only training.

This is a count-likelihood diagnostic extension of the production W/V/H
parameterization, not a claim that the original cosine objective was unchanged.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
if __package__:
    from .mapping_optimizer import NMFModel
else:
    from mapping_optimizer import NMFModel

MODALITIES = ("single_cell", "spatial")


def make_masks(shape, seed, validation_fraction=.15, test_fraction=.15):
    if min(shape) < 4 or not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction+test_fraction >= .8:
        raise ValueError("Need >=4 rows/columns and positive validation/test fractions totaling <.8")
    rng=np.random.default_rng(seed)
    for _ in range(100):
        u=rng.random(shape)
        test=u<test_fraction
        validation=(u>=test_fraction)&(u<test_fraction+validation_fraction)
        train=~(test|validation)
        if train.sum(0).min()>=2 and train.sum(1).min()>=2 and test.any() and validation.any():
            return dict(train=train,validation=validation,test=test)
    raise ValueError("Cannot construct sufficient training coverage; increase data size")


def training_view(x, mask):
    x=np.asarray(x,dtype=np.float32);mask=np.asarray(mask,dtype=bool)
    if x.shape!=mask.shape or not mask.any() or not np.isfinite(x[mask]).all() or (x[mask]<0).any():
        raise ValueError("Training entries must be finite nonnegative counts and match the mask")
    # Held-out values, including NaN sentinels, are never used below.
    observed=np.where(mask,x,0).astype(np.float32)
    rate=observed.sum(1)/np.maximum(mask.sum(1),1)
    global_mean=max(float(observed.sum()/mask.sum()),1e-6)
    exposure=np.maximum(rate/global_mean,1e-3).astype(np.float32)
    return observed,exposure


def gene_mean_prediction(x,mask):
    observed,e=training_view(x,mask)
    denominator=(mask*e[:,None]).sum(0)
    # Small fixed pseudocount prevents zero-rate predictions for unseen positives.
    rate=(observed.sum(0)+.1)/(denominator+.1)
    return np.maximum(e[:,None]*rate[None,:],1e-8)


def masked_deviance(mu, target, mask):
    """Mask both prediction loss and target before any arithmetic."""
    y=torch.where(mask,target,torch.zeros_like(target))
    p=mu.clamp_min(1e-8)
    term=p-y+torch.where(y>0,y*torch.log(y.clamp_min(1e-8)/p),torch.zeros_like(y))
    return 2*term.masked_select(mask).mean()


class MaskedProgramModel(NMFModel):
    def __init__(self, sc, sp, sc_mask, sp_mask, k, mode, device="cpu", aggregation_weight=0.):
        sx,se=training_view(sc,sc_mask);gx,ge=training_view(sp,sp_mask)
        # No reference type prior: comparability cannot be attributed to type labels.
        super().__init__(sx,gx,np.ones((len(sx),1),dtype=np.float32),
                         np.full(len(gx),1/len(gx),dtype=np.float32),torch.device(device),
                         reconstruction_mode=mode,n_programs=k,type_prior_weight=0.)
        self.register_buffer('sc_mask',torch.as_tensor(sc_mask,dtype=torch.bool,device=device))
        self.register_buffer('sp_mask',torch.as_tensor(sp_mask,dtype=torch.bool,device=device))
        self.register_buffer('sc_exposure',torch.as_tensor(se[:,None],device=device))
        self.register_buffer('sp_exposure',torch.as_tensor(ge[:,None],device=device))
        self.aggregation_weight=aggregation_weight
        # A learned positive SC row gain removes the arbitrary scale obstacle
        # when a spatial-only dictionary is frozen and transferred back to SC.
        # Spatial V already has free row amplitude. Gains use training entries only.
        self._sc_gain=torch.nn.Parameter(torch.full((len(sx),1),float(np.log(np.expm1(1.))),device=device))
        self.sc_scale=max(float(sx.sum()/np.asarray(sc_mask).sum()),1e-3)
        self.sp_scale=max(float(gx.sum()/np.asarray(sp_mask).sum()),1e-3)

    def predictions(self):
        w=self._factor_W();v=F.softplus(self._V);h=F.softplus(self._H)
        hs=F.softplus(self._H_sp) if self.reconstruction_mode=='separate' else h
        return self.sc_exposure*F.softplus(self._sc_gain)*(w@h), self.sp_exposure*(v@hs)

    def objective(self, target_modality=None):
        sc,sp=self.predictions()
        ls=masked_deviance(sc,self.S,self.sc_mask)/self.sc_scale
        lg=masked_deviance(sp,self.G,self.sp_mask)/self.sp_scale
        if target_modality=='single_cell':return ls
        if target_modality=='spatial':return lg
        loss=.5*(ls+lg)
        if self.aggregation_weight:
            mapping=F.softmax(self._factor_W()@F.softplus(self._V).T,dim=1)
            # Horvitz-Thompson input estimate under random entry masking. The
            # hidden SC entries were zeroed before model construction.
            agg=mapping.T@(self.S/self.sc_mask.float().mean())
            a=agg*self.sp_mask;g=self.G*self.sp_mask
            keep=(g.square().sum(0)>0)
            if keep.any():
                loss=loss+self.aggregation_weight*(1-F.cosine_similarity(a[:,keep],g[:,keep],dim=0).mean())
        return loss


def _optimize(model,epochs,learning_rate,target=None):
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    history=[]
    for epoch in range(epochs):
        optimizer.zero_grad();loss=model.objective(target)
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite masked loss")
        loss.backward();optimizer.step();history.append(float(loss.detach().cpu()))
    return history


def fit_masked(sc,sp,sc_mask,sp_mask,*,k,method,seed=0,epochs=200,learning_rate=.05,device="cpu",aggregation_weight=0.):
    """Only masks marked True may affect the returned predictions/factors."""
    if method not in ('shared','separate','fixed_random','shuffled_shared','sc_to_sp','sp_to_sc'):
        raise ValueError('Unknown method')
    if epochs<1 or k<1 or aggregation_weight<0:raise ValueError('Invalid K, epochs or weight')
    if aggregation_weight and method in ('sc_to_sp','sp_to_sc'):
        raise ValueError('Directional frozen-dictionary controls require aggregation_weight=0')
    perm=np.arange(sp.shape[1])
    if method=='shuffled_shared':
        # Permute data AND masks; invert predictions before scoring. This is a
        # training correspondence intervention, not deliberately mislabelled evaluation.
        perm=np.random.default_rng(seed+7919).permutation(sp.shape[1])
    torch.manual_seed(seed)
    mode='separate' if method=='separate' else ('fixed_random' if method=='fixed_random' else 'shared')
    model=MaskedProgramModel(sc,sp[:,perm],sc_mask,sp_mask[:,perm],k,mode,device,aggregation_weight)
    if method in ('sc_to_sp','sp_to_sc'):
        source='single_cell' if method=='sc_to_sp' else 'spatial'
        target='spatial' if source=='single_cell' else 'single_cell'
        history=_optimize(model,epochs,learning_rate,source)
        # Freeze the learned dictionary and source coefficients; infer target
        # coefficients from target training entries only.
        for p in model.parameters():p.requires_grad_(False)
        (model._V if target=='spatial' else model._W).requires_grad_(True)
        if target=='single_cell':model._sc_gain.requires_grad_(True)
        history+=_optimize(model,epochs,learning_rate,target)
    else:
        history=_optimize(model,epochs,learning_rate)
    with torch.no_grad():a,b=model.predictions()
    predicted={'single_cell':a.cpu().numpy(),'spatial':b.cpu().numpy()[:,np.argsort(perm)]}
    factors=model.export_factors()
    factors['sc_exposure']=model.sc_exposure.cpu().numpy()
    factors['sp_exposure']=model.sp_exposure.cpu().numpy()
    factors['sc_gain']=F.softplus(model._sc_gain).detach().cpu().numpy()
    factors['spatial_gene_permutation']=perm
    return predicted,factors,np.array(history)


def evaluate_entries(actual,predicted,mask):
    y=np.asarray(actual,float)[mask];p=np.maximum(np.asarray(predicted,float)[mask],1e-8)
    if not len(y) or not np.isfinite(y).all() or (y<0).any():raise ValueError('Invalid evaluation targets')
    dev=2*(p-y+np.where(y>0,y*np.log(np.maximum(y,1e-8)/p),0))
    logy,logp=np.log1p(y),np.log1p(p)
    def corr(a,b):
        return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>0 and b.std()>0 else np.nan
    gene_pcc=[]
    for j in range(actual.shape[1]):
        m=mask[:,j]
        gene_pcc.append(corr(np.log1p(actual[m,j]),np.log1p(predicted[m,j])))
    valid=np.isfinite(gene_pcc)
    return dict(poisson_deviance=float(dev.mean()),log_rmse=float(np.sqrt(np.mean((logy-logp)**2))),
                raw_mae=float(np.abs(y-p).mean()),pooled_log_pcc=corr(logy,logp),
                median_gene_log_pcc=float(np.nanmedian(gene_pcc)) if valid.any() else np.nan,
                valid_gene_pcc=int(valid.sum()),n_test_entries=len(y),nonzero_fraction=float((y>0).mean()),
                nonzero_log_rmse=float(np.sqrt(np.mean((logy[y>0]-logp[y>0])**2))) if (y>0).any() else np.nan,
                zero_mae=float(p[y==0].mean()) if (y==0).any() else np.nan)


def run_masked_experiment(sc,sp,out,*,genes=None,candidates=(2,4,6,8),mask_seeds=(31,47),seeds=(0,1),
                          epochs=200,learning_rate=.05,validation_fraction=.15,test_fraction=.15,
                          tolerance=.01,device='cpu',aggregation_weight=0.,rerun=False):
    sc,sp=np.asarray(sc,np.float32),np.asarray(sp,np.float32)
    if sc.ndim!=2 or sp.ndim!=2 or sc.shape[1]!=sp.shape[1]:raise ValueError('Aligned cells/spots by genes required')
    genes=np.asarray(genes if genes is not None else [f'Gene_{i}' for i in range(sc.shape[1])],dtype=str)
    if len(genes)!=sc.shape[1] or len(set(genes))!=len(genes):raise ValueError('Unique aligned gene IDs required')
    candidates=sorted(set(int(k) for k in candidates))
    if not candidates or candidates[0]<1 or not seeds or not mask_seeds or tolerance<0:raise ValueError('Invalid experiment settings')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config=dict(candidates=candidates,mask_seeds=list(mask_seeds),seeds=list(seeds),epochs=epochs,
                learning_rate=learning_rate,validation_fraction=validation_fraction,test_fraction=test_fraction,
                tolerance=tolerance,device=str(device),aggregation_weight=aggregation_weight,
                data_sha256=hashlib.sha256(sc.tobytes()+sp.tobytes()+genes.tobytes()).hexdigest(),
                code_sha256={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                             for p in ['masked_programs.py','mapping_optimizer.py']},
                objective='observed-entry mean-scaled Poisson deviance',type_prior_weight=0.,density_weight=0.)
    keys=['validation_scores','test_metrics','selection']
    if (out/'run.json').exists() and not rerun:
        if json.loads((out/'run.json').read_text())!=config:raise ValueError('Cache differs; use new OUT or RERUN=True')
        return {k:pd.read_csv(out/f'{k}.csv') for k in keys}
    if (out/'run.json').exists():(out/'run.json').unlink()
    masks={};validation=[];choices=[]
    # Select K for shared and independently tuned separate dictionaries. No test scoring here.
    for mask_seed in mask_seeds:
        ms=make_masks(sc.shape,mask_seed,validation_fraction,test_fraction)
        mg=make_masks(sp.shape,mask_seed+100003,validation_fraction,test_fraction)
        masks[mask_seed]=(ms,mg)
        np.savez_compressed(out/f'masks_{mask_seed}.npz',**{f'sc_{k}':v for k,v in ms.items()},**{f'sp_{k}':v for k,v in mg.items()})
        base={c:evaluate_entries(x,gene_mean_prediction(x,m['train']),m['validation'])['poisson_deviance']
              for c,x,m in zip(MODALITIES,(sc,sp),(ms,mg))}
        for method in ('shared','separate'):
            for k in candidates:
                for seed in seeds:
                    pred,_,_=fit_masked(sc,sp,ms['train'],mg['train'],k=k,method=method,seed=seed,
                        epochs=epochs,learning_rate=learning_rate,device=device,aggregation_weight=aggregation_weight)
                    vals={c:evaluate_entries(x,pred[c],m['validation'])['poisson_deviance']/max(base[c],1e-8)
                          for c,x,m in zip(MODALITIES,(sc,sp),(ms,mg))}
                    validation.append(dict(mask_seed=mask_seed,method=method,K=k,seed=seed,
                        sc_relative_deviance=vals['single_cell'],sp_relative_deviance=vals['spatial'],
                        worst_relative_deviance=max(vals.values())))
                print(f'Mask {mask_seed} | validation {method} | K={k} finished',flush=True)
            table=pd.DataFrame(validation)
            curve=table[(table.mask_seed==mask_seed)&(table.method==method)].groupby('K').worst_relative_deviance.mean()
            chosen=int(curve[curve<=curve.min()+tolerance].index.min())
            choices.append(dict(mask_seed=mask_seed,method=method,selected_K=chosen,
                                best_at_boundary=bool(int(curve.idxmin()) in (min(candidates),max(candidates)))))
        print(f'Mask {mask_seed}: K selection finished using validation entries only',flush=True)
    validation=pd.DataFrame(validation);choices=pd.DataFrame(choices)
    validation.to_csv(out/'validation_scores.csv',index=False);choices.to_csv(out/'selection.csv',index=False)
    (out/'selection_frozen.json').write_text(json.dumps(dict(config=config,choices=choices.to_dict('records')),indent=2))
    # Only now refit on train+validation, and read test targets for evaluation.
    rows=[]
    for mask_seed in mask_seeds:
        ms,mg=masks[mask_seed];ts=~ms['test'];tg=~mg['test']
        k=int(choices[(choices.mask_seed==mask_seed)&(choices.method=='shared')].selected_K.iloc[0])
        separate_k=int(choices[(choices.mask_seed==mask_seed)&(choices.method=='separate')].selected_K.iloc[0])
        jobs=[(method,seed) for method in ['shared','separate','separate_tuned','fixed_random','shuffled_shared'] for seed in seeds]
        if aggregation_weight==0:jobs += [(method,seed) for method in ['sc_to_sp','sp_to_sc'] for seed in seeds]
        jobs += [('gene_mean',-1),('zero',-1)]
        for label,seed in jobs:
            use_k=separate_k if label=='separate_tuned' else k
            if label in ('gene_mean','zero'):
                predictions={c:(gene_mean_prediction(x,m) if label=='gene_mean' else np.zeros_like(x))
                             for c,x,m in zip(MODALITIES,(sc,sp),(ts,tg))}
                factors={};history=np.array([])
            else:
                predictions,factors,history=fit_masked(sc,sp,ts,tg,k=use_k,method='separate' if label=='separate_tuned' else label,
                    seed=seed,epochs=epochs,learning_rate=learning_rate,device=device,aggregation_weight=aggregation_weight)
            folder=out/f'mask_{mask_seed}'/label/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'fit.npz',predicted_sc=predictions['single_cell'],predicted_sp=predictions['spatial'],
                                genes=genes,loss=history,**factors)
            for c,x,m in zip(MODALITIES,(sc,sp),(ms,mg)):
                metric=evaluate_entries(x,predictions[c],m['test'])
                rows.append(dict(mask_seed=mask_seed,seed=seed,method=label,K=use_k if factors else np.nan,
                    modality=c,is_cross_domain_target=(label=='sc_to_sp' and c=='spatial') or (label=='sp_to_sc' and c=='single_cell'),**metric))
        print(f'Mask {mask_seed}: held-out evaluation finished',flush=True)
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'test_metrics.csv',index=False)
    (out/'run.json').write_text(json.dumps(config,indent=2))
    return dict(validation_scores=validation,test_metrics=metrics,selection=choices)
