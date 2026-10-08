"""Training-only velocity spatial-regularization experiment. No output filters."""
from pathlib import Path
import copy,json,time,hashlib
import numpy as np,pandas as pd,torch
from ._spatial_mixture_ablation import SpotAblationModel,Switches
from ._trajectory_refinement import nb_log_prob
from ._hvgk_pyro_module import HVGKModuleKineticsPyroModule
from ._spatial_smoothing_experiment import spatial_graph,torch_graph,protocols,measure

def fit_training_regularizer(obs,state,p,weight,steps):
    if not np.isfinite(weight) or weight<0 or steps<1:raise ValueError('Nonnegative weight and positive steps required')
    m=SpotAblationModel(obs,Switches());m.load_state_dict(state)
    n=int(obs['n_spots']);adj=torch_graph(p)
    y,ex,amb,theta=[torch.tensor(obs[k],dtype=torch.float32) for k in ['counts','exposure','ambient','theta']]
    train=torch.tensor(obs['train_mask']);val=torch.tensor(obs['val_mask'])
    with torch.no_grad():scale=m()[1][:n].square().mean().clamp_min(1e-8)
    history=[]
    def loss_parts():
        bio,v,_,_=m()
        nll=-nb_log_prob(y,(bio+amb)*ex,theta)
        data=(nll[train].sum()+m.penalty())/train.sum()
        reg=HVGKModuleKineticsPyroModule._spatial_mse(v[:n],adj)/scale
        return data,reg,nll,val
    with torch.no_grad():
        data,reg,nll,_=loss_parts();best=float(nll[val].mean())
        history.append(dict(step=0,data_loss=float(data),spatial_penalty=float(reg),weighted_penalty=weight*float(reg),val_nll=best))
    best_step=0;best_state=copy.deepcopy(m.state_dict())
    opt=torch.optim.Adam(m.parameters(),lr=.01);start=time.time()
    for step in range(1,steps+1):
        opt.zero_grad();data,reg,_,_=loss_parts();loss=data+weight*reg
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),20);opt.step()
        if step==steps//2:
            for g in opt.param_groups:g['lr']=.005
        if step==1 or step%20==0 or step==steps:
            with torch.no_grad():
                data,reg,nll,_=loss_parts();score=float(nll[val].mean())
            history.append(dict(step=step,data_loss=float(data),spatial_penalty=float(reg),weighted_penalty=weight*float(reg),val_nll=score))
            if score<best:best=score;best_step=step;best_state=copy.deepcopy(m.state_dict())
    final_state=copy.deepcopy(m.state_dict())
    with torch.no_grad():final_velocity=m()[1][:n].numpy()
    m.load_state_dict(best_state)
    with torch.no_grad():selected_velocity=m()[1][:n].numpy()
    assert torch.equal(state['time_input'],final_state['time_input'])
    return dict(final=final_velocity,selected=selected_velocity),dict(final=final_state,selected=best_state),pd.DataFrame(history),dict(
        best_step=best_step,best_val_nll=best,steps=steps,seconds=time.time()-start,energy_normalization=float(scale))

def run_training_only(run_dir,out,weights=(0.,.001,.01,.1),steps=120,seeds=(11,29,47),
                      scenarios=('shared_mismatch','independent_dynamics'),callback=None):
    run_dir=Path(run_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    config=dict(weights=list(weights),steps=steps,seeds=list(seeds),scenarios=list(scenarios),k=24,
        primary_checkpoint='fixed_final_step',secondary_checkpoint='validation_selected_including_step0',
        regularizer='production_velocity_edge_energy_divided_by_frozen_initial_velocity_mean_square',
        postprocessing=False,truth_in_fit=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config['inputs']={str(f.relative_to(run_dir)):hashlib.sha256(f.read_bytes()).hexdigest()
        for scenario in scenarios for seed in seeds
        for f in [run_dir/f'{scenario}_seed{seed}'/rel for rel in ['observations.npz','truth.npz','S1_G1_M1/model_state.pt']]}
    path=out/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Changed experiment: choose a new output directory')
    path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[];cbrows=[]
    for scenario in scenarios:
        for seed in seeds:
            src=run_dir/f'{scenario}_seed{seed}';dest=out/src.name;dest.mkdir(exist_ok=True)
            obs=dict(np.load(src/'observations.npz'));truth=dict(np.load(src/'truth.npz'))
            n=int(obs['n_spots']);p=spatial_graph(obs['xy'],24);x,indices=protocols(obs,truth)
            initial=torch.load(src/'S1_G1_M1/model_state.pt',map_location='cpu',weights_only=True)
            m=SpotAblationModel(obs,Switches());m.load_state_dict(initial)
            with torch.no_grad():base=m()[1][:n].numpy()
            def evaluate(v,weight,stage,info):
                metrics,cb=measure(v,base,obs,truth,p,x,indices)
                metrics['cosine_to_initial']=metrics.pop('cosine_to_unsmoothed')
                rows.append(dict(scenario=scenario,seed=seed,weight=weight,checkpoint=stage,**metrics,**info))
                cb['scenario']=scenario;cb['seed']=seed;cb['weight']=weight;cb['checkpoint']=stage;cbrows.append(cb)
                pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
            evaluate(base,0.,'initial',dict(evaluated_step=0))
            for weight in weights:
                arm=dest/f'lambda_{weight:g}';arm.mkdir(exist_ok=True)
                if (arm/'metadata.json').exists() and all((arm/(name+'_velocity.npy')).exists() for name in ['final','selected']):
                    info=json.loads((arm/'metadata.json').read_text())
                    vectors={name:np.load(arm/(name+'_velocity.npy')) for name in ['final','selected']}
                else:
                    vectors,states,hist,info=fit_training_regularizer(obs,initial,p,weight,steps)
                    for name in vectors:
                        np.save(arm/(name+'_velocity.npy'),vectors[name]);torch.save(states[name],arm/(name+'_state.pt'))
                    hist.to_csv(arm/'history.csv',index=False)
                    (arm/'metadata.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
                for stage in ['final','selected']:
                    evaluate(vectors[stage],weight,stage,dict(**info,evaluated_step=steps if stage=='final' else info['best_step']))
                if callback:callback(dict(scenario=scenario,seed=seed,weight=weight,best_step=info['best_step']))
    pd.concat(cbrows,ignore_index=True).to_csv(out/'cbdir_transitions.csv',index=False)
    return pd.DataFrame(rows)
