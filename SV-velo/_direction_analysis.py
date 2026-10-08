"""Repeatable CBDir evaluation for fixed-time kinetic refinement."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch,pyro
from . import HVGK
from ._direction import configure_direction_stage
from ._inference import log_time_summary


def cbdir_table(velocity, expression, graph, states, source_mask):
    src,dst=graph.nonzero()
    states=np.asarray(states,dtype=str)
    rows=[]
    order=['VZ','SVZ','IZ','CP','L5/6']
    for a,b in zip(order[:-1],order[1:]):
        keep=source_mask[src] & (states[src]==a) & (states[dst]==b)
        i,j=src[keep],dst[keep]
        delta=expression[j]-expression[i]
        denom=np.linalg.norm(velocity[i],axis=1)*np.linalg.norm(delta,axis=1)
        valid=np.isfinite(denom)&(denom>1e-12)
        score=np.clip((velocity[i[valid]]*delta[valid]).sum(1)/denom[valid],-1,1)
        sources=pd.DataFrame({'source':i[valid],'score':score}).groupby('source').score.mean()
        rows.append({'transition':a+' -> '+b,'sources':len(sources),'CBDir':sources.mean()})
    return pd.DataFrame(rows)


def repeated_direction_evaluation(adata, graph, output, seeds=(700,701,702), samples=20):
    """Same seeds/coordinates in all arms; dispersion is MC error, not biological CI."""
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    baseline_dir=Path(manifest['baseline'])
    config=json.loads((baseline_dir/'config.json').read_text(encoding='utf-8'))
    state=torch.load(baseline_dir/'model_state.pt',weights_only=True,map_location='cpu')
    arrays=np.load(output/'direction_targets.npz')
    np.testing.assert_array_equal(arrays['obs_names'],adata.obs_names.to_numpy(dtype=str))
    np.testing.assert_array_equal(arrays['var_names'],adata.var_names.to_numpy(dtype=str))
    def load_baseline():
        pyro.set_rng_seed(manifest['seed'])
        m=HVGK(adata,n_modules=5,initialization='expression',initialization_reverse=True,
            time_parameterization='centered',spatial_config={'enabled':False},
            observation_config={'gene_detection':config['gene_detection'],'ambient_enabled':config['ambient_enabled']})
        args,kwargs=m._joint_args()
        with torch.no_grad(): m.module.guide(*args,**kwargs)
        m.module.load_state_dict(state); m.is_trained_=True
        return m
    baseline=load_baseline()
    # Reproduce the original 20-draw coordinate estimate before any new seeds.
    with torch.no_grad():
        initial=baseline.sample_posterior(num_samples=20,return_sites=['spliced_corrected','velocity'])['post_sample_means']
    fixed_expression=initial['spliced_corrected']
    states=adata.obs['clusters'].astype(str).to_numpy()
    baseline_check=cbdir_table(initial['velocity'],fixed_expression,graph,states,np.ones(adata.n_obs,dtype=bool))
    np.testing.assert_allclose(baseline_check.CBDir,pd.read_csv(output/'baseline_cbdir.csv').CBDir,rtol=1e-5,atol=1e-6)
    np.save(output/'fixed_expression_for_cbdir.npy',fixed_expression)
    reference=pd.read_csv(output/'baseline_time.csv',index_col=0)
    masks={'all':np.ones(adata.n_obs,dtype=bool),'withheld_sources':arrays['holdout']}
    replicate_rows=[]; pooled_rows=[]; transition_rows=[]; freeze_audit=[]
    for name in ('baseline','kinetics_only','time_tangent'):
        model=load_baseline()
        if name!='baseline':
            arm=json.loads((output/name/'config.json').read_text(encoding='utf-8'))
            configure_direction_stage(model,arrays['target'],arrays['train_mask'],arm['direction_weight'])
            arm_state=torch.load(output/name/'model_state.pt',weights_only=True,map_location='cpu')
            model.module.load_state_dict(arm_state)
            frozen=json.loads((output/name/'frozen_parameters.json').read_text(encoding='utf-8'))
            changed=[k for k in frozen if not torch.equal(state['_guide.'+k],arm_state['_guide.'+k])]
            if changed:
                raise AssertionError(f'Frozen parameters changed in {name}: {changed}')
            freeze_audit.append({'name':name,'frozen_tensors':len(frozen),'changed_tensors':len(changed)})
        np.testing.assert_allclose(log_time_summary(model).to_numpy(),reference.to_numpy(),rtol=1e-6,atol=1e-6)
        vectors=[]
        for seed in seeds:
            pyro.set_rng_seed(seed)
            with torch.no_grad():
                velocity=model.sample_posterior(num_samples=samples,return_sites=['velocity'])['post_sample_means']['velocity']
            vectors.append(velocity)
            for scope,mask in masks.items():
                scores=cbdir_table(velocity,fixed_expression,graph,states,mask)
                replicate_rows.append({'name':name,'seed':seed,'scope':scope,'samples':samples,
                    'CBDir_macro':scores.CBDir.mean()})
        pooled=np.mean(vectors,axis=0)
        np.save(output/('pooled_velocity_'+name+'.npy'),pooled)
        for scope,mask in masks.items():
            scores=cbdir_table(pooled,fixed_expression,graph,states,mask)
            pooled_rows.append({'name':name,'scope':scope,'samples':samples*len(seeds),
                'CBDir_macro':scores.CBDir.mean(),'covered_transitions':int(scores.CBDir.notna().sum())})
            transition_rows.extend([{'name':name,'scope':scope,**r} for r in scores.to_dict('records')])
        pd.DataFrame(replicate_rows).to_csv(output/'cbdir_mc_repeats.csv',index=False)
        print('Posterior evaluation completed:',name,flush=True)
    repeats=pd.DataFrame(replicate_rows)
    pooled=pd.DataFrame(pooled_rows)
    transitions=pd.DataFrame(transition_rows)
    pooled.to_csv(output/'cbdir_pooled.csv',index=False)
    transitions.to_csv(output/'cbdir_pooled_by_transition.csv',index=False)
    pd.DataFrame(freeze_audit).to_csv(output/'frozen_parameter_audit.csv',index=False)
    return repeats,pooled,transitions
