"""Select predictive program dimension using observed validation genes only.

Selection and truth evaluation are deliberately separate functions. The selector
receives only observed/reference counts, genes and reference type labels.
"""
from pathlib import Path
import json
import hashlib
import inspect
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
if __package__:
    from .mapping_optimizer import NMFModel
    from .integration_analysis import evaluate_transfer, split_genes
else:
    from mapping_optimizer import NMFModel
    from integration_analysis import evaluate_transfer, split_genes

INPUT_KEYS = ("genes", "reference_spliced", "reference_unspliced", "observed_spliced", "observed_unspliced", "reference_types")


def observed_inputs(data):
    return {key: np.asarray(data[key]) for key in INPUT_KEYS}


def input_fingerprint(data):
    h = hashlib.sha256()
    for key in INPUT_KEYS:
        x = np.ascontiguousarray(data[key])
        h.update(key.encode()); h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(x.tobytes())
    return h.hexdigest()


def _arrays(inputs):
    s = (inputs["reference_spliced"] + inputs["reference_unspliced"]).astype(np.float32)
    g = (inputs["observed_spliced"] + inputs["observed_unspliced"]).astype(np.float32)
    labels = np.asarray(inputs["reference_types"])
    _, codes = np.unique(labels, return_inverse=True)
    t = np.eye(len(np.unique(labels)), dtype=np.float32)[codes]
    return s, g, t


def _gene_pcc(actual, predicted):
    a = actual - actual.mean(0)
    b = predicted - predicted.mean(0)
    denom = np.linalg.norm(a, axis=0)*np.linalg.norm(b, axis=0)
    return np.divide((a*b).sum(0), denom, out=np.full(len(denom), np.nan), where=denom>1e-12)


def make_split(inputs, *, test_fraction=.2, n_folds=2, split_seed=42):
    s, g, _ = _arrays(inputs)
    genes = inputs["genes"].tolist()
    if len(set(genes)) != len(genes): raise ValueError("Gene names must be unique")
    eligible = [name for name, valid in zip(genes, (s.sum(0)>0)&(g.sum(0)>0)) if valid]
    pool, test = split_genes(eligible, test_fraction, split_seed)
    if n_folds < 2 or len(pool)//n_folds < 3: raise ValueError("Need at least two folds and three validation genes per fold")
    rng = np.random.default_rng(split_seed+1)
    folds = [x.tolist() for x in np.array_split(rng.permutation(pool), n_folds)]
    return dict(outer_training_genes=pool, test_genes=test, validation_folds=folds)


def choose_k(scores, tolerance=.01):
    if tolerance < 0: raise ValueError("Tolerance must be nonnegative")
    curve = scores.groupby("K").validation_pcc.mean().sort_index()
    if not np.isfinite(curve).all(): raise ValueError("All candidate K values need finite validation scores")
    best = float(curve.max())
    chosen = int(curve[curve >= best-tolerance].index.min())
    return chosen, curve


def _fit(s, g, t, indices, k, seed, epochs, learning_rate, device, type_prior_weight):
    torch.manual_seed(seed)
    m = NMFModel(s[:, indices], g[:, indices], t, np.full(len(g), 1/len(g), dtype=np.float32),
                 torch.device(device), n_programs=k, type_prior_weight=type_prior_weight)
    mapping, loss = m.train_model(epochs-1, learning_rate=learning_rate, print_each=None)
    return mapping, m.export_factors(), loss, sum(p.numel() for p in m.parameters() if p.requires_grad)


def select_program_number(inputs, *, candidates=(2,4,6,8,10,12), seeds=(0,1), epochs=150,
                          learning_rate=.1, n_folds=2, split_seed=42, test_fraction=.2,
                          tolerance=.01, device="cpu", type_prior_weight=.5):
    """Never reads H, K_true, latent state, capture rates or test expression values.

    Filtering uses detection before splitting; all fitting/scoring then uses the
    outer training pool. Tolerance is a predeclared predictive parsimony rule,
    not a confidence interval or a claim to recover the generating rank.
    """
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError("Pass observed_inputs(data), without hidden truth keys")
    if "n_programs" not in inspect.signature(NMFModel).parameters:
        raise RuntimeError("Update mapping_optimizer.py and restart the kernel")
    candidates = sorted(set(int(k) for k in candidates))
    if not candidates or candidates[0] < 1 or not seeds or epochs < 1:
        raise ValueError("Provide candidate ranks, seeds and positive epochs")
    split = make_split(inputs, test_fraction=test_fraction, n_folds=n_folds, split_seed=split_seed)
    s, g, t = _arrays(inputs)
    genes = inputs["genes"].tolist()
    records = []
    for fold, val in enumerate(split["validation_folds"]):
        train = [x for x in split["outer_training_genes"] if x not in set(val)]
        tr, va = [genes.index(x) for x in train], [genes.index(x) for x in val]
        for k in candidates:
            for seed in seeds:
                mapping, _, _, n_params = _fit(s, g, t, tr, k, seed, epochs, learning_rate, device, type_prior_weight)
                pcc = _gene_pcc(g[:, va], mapping.T @ s[:, va])
                if np.isfinite(pcc).sum() < max(3, len(val)//2):
                    raise ValueError("Too few evaluable validation genes; revise QC before selection")
                records.append(dict(K=k, fold=fold, seed=seed, validation_pcc=float(np.nanmedian(pcc)),
                                    valid_genes=int(np.isfinite(pcc).sum()), n_parameters=n_params))
    scores = pd.DataFrame(records)
    selected, curve = choose_k(scores, tolerance)
    return dict(selected_K=selected, split=split,
                selection_rule="smallest K within fixed tolerance of best mean inner-fold/seed median gene PCC",
                tolerance=tolerance, candidates=candidates, validation_curve={str(k):float(v) for k,v in curve.items()},
                best_on_grid_boundary=bool(int(curve.idxmax()) in (candidates[0], candidates[-1])),
                input_sha256=input_fingerprint(inputs)), scores


def recovery_metrics(learned, truth):
    """Rectangular matching: unmatched true/learned programs receive zero credit."""
    def norm(x): return x/np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = norm(learned) @ norm(truth).T
    row, col = linear_sum_assignment(-sim)
    total = float(sim[row,col].sum())
    precision, recall = total/len(learned), total/len(truth)
    within = norm(learned) @ norm(learned).T
    np.fill_diagonal(within, np.nan)
    return dict(recovery_precision=precision, recovery_recall=recall,
                recovery_f1=2*precision*recall/max(precision+recall, 1e-12),
                max_program_cosine=float(np.nanmax(within)) if len(learned)>1 else np.nan), row, col


def evaluate_frozen_selection(data, decision, out, *, seeds=(0,1), epochs=150,
                              learning_rate=.1, device="cpu", type_prior_weight=.5):
    """Refit every prespecified candidate for sensitivity; selection is already frozen.

    All refits use the entire outer training pool. Test curves are diagnostic and
    must not be used to revise K, tolerance, candidate grid or training budget.
    """
    inputs = observed_inputs(data)
    if input_fingerprint(inputs) != decision["input_sha256"]:
        raise ValueError("Inputs differ from frozen selection")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    s, g, t = _arrays(inputs)
    genes = inputs["genes"].tolist()
    train, test = decision["split"]["outer_training_genes"], decision["split"]["test_genes"]
    tr, te = [genes.index(x) for x in train], [genes.index(x) for x in test]
    truth = {c: pd.DataFrame(data[f"true_{c}"][:,te], columns=test) for c in ("spliced","unspliced")}
    rows, gene_rows, history = [], [], []
    saved_h = {}
    for k in decision["candidates"]:
        for seed in seeds:
            m, f, loss, params = _fit(s,g,t,tr,k,seed,epochs,learning_rate,device,type_prior_weight)
            predictions = {c:pd.DataFrame(m.T @ data[f"reference_{c}"][:,te],columns=test) for c in truth}
            scores = evaluate_transfer(truth,{"shared":predictions},training_genes=train,test_genes=test)
            gene_rows.append(scores.assign(K=k,seed=seed))
            comp = m.T @ t; comp /= comp.sum(1,keepdims=True)
            true_comp = data["composition_counts"]/data["n_cells"][:,None]
            rec, matched, true_ids = recovery_metrics(f["H"],data["program_H"][:,tr])
            row = dict(K=k,seed=seed,selected=(k==decision["selected_K"]), true_K=data["program_H"].shape[0],
                       observed_test_pcc=float(np.nanmedian(_gene_pcc(g[:,te],m.T@s[:,te]))),
                       composition_rmse=float(np.sqrt(np.mean((comp-true_comp)**2))),n_parameters=params,**rec)
            for c in truth: row[c+"_pcc"] = scores.loc[scores.channel==c,"pcc"].median()
            row["fraction_mae"] = scores.loc[scores.channel=="joint","fraction_mae"].median()
            contribution = f["V"].sum(0)*f["H"].sum(1)
            contribution /= max(float(contribution.sum()),1e-12)
            row["active_programs_1pct"] = int((contribution>=.01).sum())
            rows.append(row); saved_h[(k,seed)] = f["H"]
            folder = out/f"K_{k}"/f"seed_{seed}"; folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'fit.npz',M=m,composition=comp,training_genes=np.array(train),testing_genes=np.array(test),**f)
            history.append(pd.DataFrame(loss).assign(K=k,seed=seed,epoch=np.arange(epochs)))
    stability=[]
    from itertools import combinations
    for k in decision["candidates"]:
        for a,b in combinations(seeds,2):
            r,_,_=recovery_metrics(saved_h[k,a],saved_h[k,b])
            stability.append(dict(K=k,seed_a=a,seed_b=b,matched_cosine=r["recovery_f1"]))
    tables=dict(test_summary=pd.DataFrame(rows),test_gene_metrics=pd.concat(gene_rows,ignore_index=True),
                training_history=pd.concat(history,ignore_index=True),stability=pd.DataFrame(stability))
    for key,frame in tables.items(): frame.to_csv(out/f'{key}.csv',index=False)
    return tables


def run_k_study(data_dir, out, *, scenarios, candidates=(2,4,6,8,10,12), seeds=(0,1), epochs=150,
                learning_rate=.1,n_folds=2,split_seed=42,test_fraction=.2,tolerance=.01,
                device="cpu",type_prior_weight=.5,rerun=False):
    data_dir,out=Path(data_dir),Path(out)
    config=dict(scenarios=list(scenarios),candidates=list(candidates),seeds=list(seeds),epochs=epochs,
                learning_rate=learning_rate,n_folds=n_folds,split_seed=split_seed,test_fraction=test_fraction,
                tolerance=tolerance,device=str(device),type_prior_weight=type_prior_weight,density_prior="uniform")
    config['data_sha256']={s:hashlib.sha256((data_dir/s/'simulation_arrays.npz').read_bytes()).hexdigest() for s in scenarios}
    config['code_sha256']={p:hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest() for p in
                           ['program_selection.py','mapping_optimizer.py','integration_analysis.py']}
    if (out/'run.json').exists() and not rerun:
        if json.loads((out/'run.json').read_text())!=config: raise ValueError('Cache differs; use new OUT or RERUN=True')
        return {k:pd.read_csv(out/f'{k}.csv') for k in ['validation_scores','test_summary','stability']}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run.json').exists(): (out/'run.json').unlink()
    decisions={}; validations=[]
    # Freeze every decision before evaluating ANY hidden truth or outer test score.
    for scenario in scenarios:
        with np.load(data_dir/scenario/'simulation_arrays.npz') as z:
            inputs={k:z[k] for k in INPUT_KEYS}
        decision,scores=select_program_number(inputs,candidates=candidates,seeds=seeds,epochs=epochs,
            learning_rate=learning_rate,n_folds=n_folds,split_seed=split_seed,test_fraction=test_fraction,
            tolerance=tolerance,device=device,type_prior_weight=type_prior_weight)
        decisions[scenario]=decision; validations.append(scores.assign(scenario=scenario))
        print(f'{scenario}: selected K={decision["selected_K"]} using observed validation only',flush=True)
    selection=dict(parameterization="learned_type_program",density_prior="uniform",type_prior_weight=type_prior_weight,
                   cases=decisions,config=config)
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    v=pd.concat(validations,ignore_index=True); v.to_csv(out/'validation_scores.csv',index=False)
    tests=[]; stability=[]
    for scenario in scenarios:
        with np.load(data_dir/scenario/'simulation_arrays.npz') as z: data={k:z[k] for k in z.files}
        tables=evaluate_frozen_selection(data,decisions[scenario],out/scenario,seeds=seeds,epochs=epochs,
                                        learning_rate=learning_rate,device=device,type_prior_weight=type_prior_weight)
        tests.append(tables['test_summary'].assign(scenario=scenario))
        if not tables['stability'].empty: stability.append(tables['stability'].assign(scenario=scenario))
        print(f'{scenario}: independent test evaluation finished',flush=True)
    tables=dict(validation_scores=v,test_summary=pd.concat(tests,ignore_index=True),
                stability=pd.concat(stability,ignore_index=True) if stability else pd.DataFrame(columns=['K','seed_a','seed_b','matched_cosine','scenario']))
    for key,frame in tables.items(): frame.to_csv(out/f'{key}.csv',index=False)
    (out/'run.json').write_text(json.dumps(config,indent=2))
    return tables
