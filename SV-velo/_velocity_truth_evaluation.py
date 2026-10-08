"""Ground-truth RNA-velocity evaluation; no training or model selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

def correlations(a,b):
    """Column-wise Pearson; constant columns are undefined, not zero."""
    a=a-a.mean(0);b=b-b.mean(0)
    den=np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0)
    return np.divide((a*b).sum(0),den,out=np.full(a.shape[1],np.nan),where=den>1e-12)

def evaluate_velocity(pred,true,mask=None,sign_fraction=.05):
    p=np.asarray(pred,dtype=float);t=np.asarray(true,dtype=float)
    if p.shape!=t.shape or p.ndim!=2:raise ValueError('Aligned spot x gene arrays required')
    if not np.isfinite(p).all() or not np.isfinite(t).all():raise ValueError('Nonfinite velocities')
    nt=np.linalg.norm(t,axis=1);npred=np.linalg.norm(p,axis=1)
    active=nt>1e-10;directed=active&(npred>1e-10)
    cos=np.full(len(t),np.nan);cos[active]=0.
    cos[directed]=(p[directed]*t[directed]).sum(1)/(nt[directed]*npred[directed])
    cos=np.clip(cos,-1,1)
    angle=np.full(len(t),np.nan);angle[directed]=np.degrees(np.arccos(cos[directed]))
    ratio=np.divide(npred,nt,out=np.full(len(t),np.nan),where=active)
    rms=np.sqrt((t*t).mean(0))
    threshold=sign_fraction*rms
    numerical_zero=np.maximum(1e-6*rms,1e-12)
    pos=t>np.maximum(threshold,1e-12);neg=t < -np.maximum(threshold,1e-12)
    predicted_pos=p>numerical_zero;predicted_neg=p < -numerical_zero
    covered=pos|neg;correct=(pos&predicted_pos)|(neg&predicted_neg)
    rec_pos=float(predicted_pos[pos].mean()) if pos.any() else np.nan
    rec_neg=float(predicted_neg[neg].mean()) if neg.any() else np.nan
    pearson=correlations(t,p)
    spearman=correlations(rankdata(t,axis=0),rankdata(p,axis=0))
    error=(p-t)**2
    def rel_rmse(m):
        den=(t[m]**2).sum()
        return float(np.sqrt(error[m].sum()/den)) if den>1e-20 else np.nan
    speed_r=correlations(rankdata(nt[:,None],axis=0),rankdata(npred[:,None],axis=0))[0]
    summary=dict(direction_cosine=float(np.nanmean(cos)) if active.any() else np.nan,
        angle_median_deg=float(np.nanmedian(angle)) if directed.any() else np.nan,
        opposite_fraction=float((cos[directed]<0).mean()) if directed.any() else np.nan,
        directed_fraction=float(directed[active].mean()) if active.any() else np.nan,
        active_truth_fraction=float(active.mean()),relative_rmse=rel_rmse(np.ones(t.shape,dtype=bool)),
        heldout_relative_rmse=rel_rmse(mask) if mask is not None else np.nan,
        norm_spearman=float(speed_r),norm_ratio_median=float(np.nanmedian(ratio)) if active.any() else np.nan,
        gene_pearson_mean=float(np.nanmean(pearson)) if np.isfinite(pearson).any() else np.nan,
        gene_spearman_mean=float(np.nanmean(spearman)) if np.isfinite(spearman).any() else np.nan,
        correlated_gene_fraction=float(np.isfinite(spearman).mean()),
        positive_recall=rec_pos,negative_recall=rec_neg,
        sign_balanced_accuracy=float((rec_pos+rec_neg)/2) if pos.any() and neg.any() else np.nan,
        sign_coverage=float(covered.mean()))
    spot=pd.DataFrame(dict(cosine=cos,angle_deg=angle,norm_true=nt,norm_pred=npred,norm_ratio=ratio,
        relative_error=np.divide(np.sqrt(error.sum(1)),nt,out=np.full(len(t),np.nan),where=active)))
    gene=pd.DataFrame(dict(pearson=pearson,spearman=spearman,
        relative_rmse=np.sqrt(np.divide(error.sum(0),(t*t).sum(0),out=np.full(t.shape[1],np.nan),where=(t*t).sum(0)>1e-20)),
        sign_accuracy=np.divide(correct.sum(0),covered.sum(0),out=np.full(t.shape[1],np.nan),where=covered.sum(0)>0),
        sign_entries=covered.sum(0)))
    return summary,spot,gene

def evaluate_saved_runs(run_dir,output):
    run_dir=Path(run_dir);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(run_dir/'all_results.csv');rows=[];groups=[];sens=[]
    for (scenario,seed),batch in manifest.groupby(['scenario','seed']):
        folder=run_dir/f'{scenario}_seed{seed}'
        obs=dict(np.load(folder/'observations.npz'));truth=dict(np.load(folder/'truth.npz'));N=int(obs['n_spots'])
        entropy=-(truth['composition'][:N]*np.log(truth['composition'][:N].clip(1e-10))).sum(1)
        norm=np.linalg.norm(truth['velocity'][:N],axis=1)
        # Bins are defined from truth once, shared by all models; evaluation only.
        labels=dict(stage=truth['age_stage'],dominant_type=truth['composition'][:N].argmax(1).astype(str),
            mixing_quartile=np.searchsorted(np.quantile(entropy,[.25,.5,.75]),entropy,side='right').astype(str),
            speed_quartile=np.searchsorted(np.quantile(norm,[.25,.5,.75]),norm,side='right').astype(str))
        for _,r in batch.iterrows():
            p=dict(np.load(folder/r.arm/'prediction.npz'))['velocity'][:N];t=truth['velocity'][:N]
            stats,spot,gene=evaluate_velocity(p,t,obs['test_mask'][:N])
            meta=dict(scenario=scenario,seed=int(seed),arm=r.arm)
            rows.append(dict(**meta,**stats))
            gene['gene']=obs['gene_names']
            stem=f'{scenario}_seed{seed}_{r.arm}'
            spot.to_csv(output/(stem+'_spots.csv'),index=False);gene.to_csv(output/(stem+'_genes.csv'),index=False)
            for kind,lab in labels.items():
                for group in np.unique(lab):
                    ix=lab==group
                    groups.append(dict(**meta,grouping=kind,group=group,n=int(ix.sum()),
                        direction_cosine=float(spot.loc[ix,'cosine'].mean()),
                        relative_rmse=float(np.sqrt(((p[ix]-t[ix])**2).sum()/(t[ix]**2).sum()))))
            for fraction in [0.,.05,.10]:
                if fraction==.05:result=stats
                else:result=evaluate_velocity(p,t,sign_fraction=fraction)[0]
                sens.append(dict(**meta,threshold_fraction=fraction,sign_balanced_accuracy=result['sign_balanced_accuracy'],
                                 sign_coverage=result['sign_coverage']))
    rows=pd.DataFrame(rows);rows.to_csv(output/'velocity_metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'stratified_metrics.csv',index=False)
    pd.DataFrame(sens).to_csv(output/'sign_sensitivity.csv',index=False)
    return rows,pd.DataFrame(groups),pd.DataFrame(sens)
