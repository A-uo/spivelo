"""Explicit CBDir protocols and native scVelo projection, shared across methods.

Metric follows TopoVelo evaluation_util.cross_boundary_correctness: mean over
target neighbors, then source cells, then covered directed cluster transitions.
Zero velocity contributes zero, as sklearn cosine_similarity does upstream.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


def cbdir_topovelo(positions,velocity,indices,labels,edges):
    x=np.asarray(positions);v=np.asarray(velocity);labels=np.asarray(labels,dtype=str)
    indices=np.asarray(indices)
    if x.ndim!=2 or v.shape!=x.shape or len(labels)!=len(x) or indices.shape[0]!=len(x):
        raise ValueError('Coordinates, velocities, labels and neighbors must align')
    if not np.isfinite(x).all() or not np.isfinite(v).all():raise ValueError('Nonfinite coordinates/velocities')
    if not np.issubdtype(indices.dtype,np.integer) or (indices<0).any() or (indices>=len(x)).any():
        raise ValueError('Neighbor indices must be valid integers')
    transitions=[];sources=[]
    for start,end in edges:
        scores=[];n_edges=0;zero_sources=0
        for i in np.flatnonzero(labels==start):
            js=indices[i];js=js[labels[js]==end]
            if not len(js):continue
            score=float(cosine_similarity(x[js]-x[i],v[i:i+1]).mean())
            scores.append(score);n_edges+=len(js);zero_sources+=int(np.linalg.norm(v[i])==0)
            sources.append(dict(transition=f'{start} -> {end}',source=int(i),CBDir=score,neighbors=len(js)))
        transitions.append(dict(transition=f'{start} -> {end}',CBDir=float(np.mean(scores)) if scores else np.nan,
            sources=len(scores),edges=n_edges,zero_velocity_sources=zero_sources))
    return pd.DataFrame(transitions),pd.DataFrame(sources)


def prepare_shared_embedding(adata,seed=42,n_pcs=30):
    """Expression PCA for shared expression neighbors; preserve supplied UMAP."""
    import scanpy as sc
    from sklearn.decomposition import PCA
    s=adata.layers['spliced'];s=s.toarray() if sparse.issparse(s) else np.asarray(s)
    depth=s.sum(1);target=np.median(depth[depth>0])
    lognorm=np.log1p(s/np.maximum(depth[:,None],1)*target)
    adata.obsm['X_benchmark_pca']=PCA(n_components=min(n_pcs,adata.n_vars-1,adata.n_obs-1),
        svd_solver='full').fit_transform(lognorm).astype(np.float32)
    if 'X_umap' in adata.obsm:
        origin='input_file_X_umap'
    else:
        sc.pp.neighbors(adata,n_neighbors=min(30,adata.n_obs-1),use_rep='X_benchmark_pca',random_state=seed)
        sc.tl.umap(adata,random_state=seed);origin='new_expression_UMAP_seed_'+str(seed)
    if 'X_spatial' not in adata.obsm:
        if 'spatial' in adata.obsm:adata.obsm['X_spatial']=adata.obsm['spatial'].copy()
        else:raise ValueError('Spatial coordinates are required')
    adata.obsm['X_umap']=np.asarray(adata.obsm['X_umap'])[:,:2].copy()
    adata.obsm['X_spatial']=np.asarray(adata.obsm['X_spatial'])[:,:2].copy()
    return origin


def set_shared_neighbors(adata,protocol='spatial',n_neighbors=50,seed=42):
    """TopoVelo outer spatial protocol uses Euclidean KNN including self in indices."""
    import scanpy as sc
    if protocol not in ('spatial','expression'):raise ValueError('Unknown neighbor protocol')
    rep='X_spatial' if protocol=='spatial' else 'X_benchmark_pca'
    k=min(n_neighbors,adata.n_obs-1)
    sc.pp.neighbors(adata,n_neighbors=k,use_rep=rep,random_state=seed,method='umap',metric='euclidean')
    indices=NearestNeighbors(n_neighbors=k,metric='euclidean').fit(adata.obsm[rep]).kneighbors(
        adata.obsm[rep],return_distance=False)
    adata.uns['neighbors']['indices']=indices
    return indices


def project_scvelo(adata,vkey,xkey):
    """Use scVelo itself; do not approximate its graph with hand-drawn arrows."""
    import scvelo as scv
    # All model genes, no method-specific velocity-gene thresholding.
    scv.tl.velocity_graph(adata,vkey=vkey,xkey=xkey,gene_subset=adata.var_names,
        n_jobs=1,approx=False,sqrt_transform=False,n_recurse_neighbors=2)
    for basis in ('umap','spatial'):
        scv.tl.velocity_embedding(adata,vkey=vkey,basis=basis,autoscale=True,retain_scale=False)
        if not np.isfinite(adata.obsm[f'{vkey}_{basis}']).all():
            raise FloatingPointError(f'Nonfinite scVelo projection: {vkey}/{basis}')


def import_external_method(adata,path,velocity_key,expression_key,name):
    """Align actual external predictions to shared cells/genes; never invent a baseline."""
    import anndata as ad
    other=ad.read_h5ad(path)
    if not other.obs_names.is_unique or not other.var_names.is_unique:raise ValueError('External names must be unique')
    if not adata.obs_names.isin(other.obs_names).all() or not adata.var_names.isin(other.var_names).all():
        raise ValueError('External method does not cover all benchmark cells and genes')
    other=other[adata.obs_names,adata.var_names]
    for source,target in [(velocity_key,name),(expression_key,name+'_expression')]:
        a=other.layers[source];a=a.toarray() if sparse.issparse(a) else np.asarray(a)
        if not np.isfinite(a).all():raise ValueError(f'Nonfinite external layer {source}')
        adata.layers[target]=a.copy()
    return name,name+'_expression'
