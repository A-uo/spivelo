"""Mechanism ablation evaluation; labels never enter training or select a winner."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr
from scipy.special import ndtr
from sklearn.decomposition import PCA
from ._inference import log_time_summary, time_sensitivity, time_information, elbo_score


def fixed_expression_graph(adata, k=15):
    x=adata.layers['spliced']
    x=x.toarray() if sparse.issparse(x) else np.asarray(x)
    scale=x.std(0)
    x=(x-x.mean(0))/np.maximum(scale,1e-8)
    p=PCA(n_components=min(30,*[d-1 for d in x.shape]),svd_solver='full').fit_transform(x)
    neighbors=cKDTree(p).query(p,k=min(k+1,len(x)))[1]
    ii=np.repeat(np.arange(len(x)),neighbors.shape[1]); jj=neighbors.reshape(-1)
    valid=ii!=jj
    graph=sparse.csr_matrix((np.ones(valid.sum()),(ii[valid],jj[valid])),shape=(len(x),len(x)))
    return graph.maximum(graph.T)


def evaluate_candidate(model, graph, output, config, posterior_samples=10):
    output.mkdir(parents=True,exist_ok=True)
    torch.save(model.module.state_dict(),output/'model_state.pt')
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    saved_history=model.history or {}
    history=np.asarray(saved_history.get('elbo_train',[]),dtype=float).reshape(-1)
    pd.DataFrame({'negative_elbo':history}).to_csv(output/'training.csv',index=False)
    times=log_time_summary(model)
    info=time_sensitivity(model)
    shuffled=time_information(model,num_samples=3,n_permutations=2,seed=42)
    post=model.sample_posterior(num_samples=posterior_samples,return_sites=[
        'velocity','spliced_corrected','mu_u','mu_s','T_mON','T_mOFF'])['post_sample_means']
    times.to_csv(output/'time_summary.csv')
    info.to_csv(output/'time_sensitivity.csv')
    shuffled['draws'].to_csv(output/'time_permutation.csv',index=False)
    np.savez_compressed(output/'posterior_means.npz',**post)
    states=model.adata.obs['clusters'].astype(str).to_numpy()
    order=['VZ','SVZ','IZ','CP','L5/6']
    rank=pd.Series(states).map({g:i for i,g in enumerate(order)}).to_numpy(float)
    valid=np.isfinite(rank)
    rows=[]
    src,dst=graph.nonzero()
    velocity=post['velocity']; expression=post['spliced_corrected']
    for a,b in zip(order[:-1],order[1:]):
        use=(states[src]==a)&(states[dst]==b)
        ii,jj=src[use],dst[use]
        delta=expression[jj]-expression[ii]
        denom=np.linalg.norm(velocity[ii],axis=1)*np.linalg.norm(delta,axis=1)
        keep=np.isfinite(denom)&(denom>1e-12)
        score=np.clip((velocity[ii[keep]]*delta[keep]).sum(1)/denom[keep],-1,1)
        source=pd.DataFrame({'source':ii[keep],'score':score}).groupby('source').score.mean()
        rows.append({'transition':a+' -> '+b,'valid_edges':len(score),
                     'source_mean':source.mean(),'edge_mean':score.mean() if len(score) else np.nan})
    cbdir=pd.DataFrame(rows)
    cbdir.to_csv(output/'cbdir.csv',index=False)
    correlations=[]
    for site,layer in [('mu_u','unspliced'),('mu_s','spliced')]:
        y=model.adata.layers[layer]
        y=(y.toarray() if sparse.issparse(y) else np.asarray(y)).reshape(-1)
        pred=post[site].reshape(-1)
        correlations.append({'layer':layer,'Pearson':pearsonr(y,pred)[0],
                             'Spearman':spearmanr(y,pred)[0]})
    pd.DataFrame(correlations).to_csv(output/'reconstruction.csv',index=False)
    between=float(times.mean_log_time.std(ddof=0))
    uncertainty=float(np.sqrt(np.mean(times.sd_log_time**2)))
    # Shared global location cancels in a time contrast; positive global scale
    # cannot change ordering. Do not mistake global uncertainty for uncertain ranks.
    rng=np.random.default_rng(42)
    i=rng.integers(0,len(times),10000)
    j=rng.integers(0,len(times),10000)
    i,j=i[i!=j],j[i!=j]
    rank_mu=times.ordering_coordinate.to_numpy()
    rank_sd=times.ordering_sd.to_numpy()
    certainty=ndtr(np.abs(rank_mu[i]-rank_mu[j])/np.sqrt(rank_sd[i]**2+rank_sd[j]**2))
    width=max(1,min(20,len(history)//4))
    result={**config,**elbo_score(model,num_samples=5,seed=42),
        'between_log_time_sd':between,'posterior_log_time_rms_sd':uncertainty,
        'between_over_uncertainty':between/max(uncertainty,1e-12),
        'pairwise_order_confidence_median':float(np.median(certainty)),
        'reference_Spearman':spearmanr(rank[valid],times.mean_log_time.to_numpy()[valid])[0],
        'CBDir_macro':cbdir.source_mean.mean(),
        'covered_transitions':int(cbdir.source_mean.notna().sum()),
        'conditional_information_median':info.conditional_log_time_information.median(),
        'depth_profiled_information_median':info.depth_profiled_log_time_information.median(),
        'time_permutation_gain_per_entry':shuffled['draws'].gain_per_count_entry.mean(),
        'history_available':bool(len(history)),
        'tail_relative_change':float((history[-width:].mean()-history[-2*width:-width].mean())/max(abs(history[-2*width:-width].mean()),1.)) if len(history)>=2*width else np.nan}
    # Reporting flag only; never alters fitted time or selects by reference labels.
    result['time_resolution_flag']='weak' if result['pairwise_order_confidence_median']<.75 else 'requires_validation'
    result={k:v.item() if isinstance(v,np.generic) else v for k,v in result.items()}
    (output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
