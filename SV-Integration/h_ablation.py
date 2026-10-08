"""Controlled tests of shared-H training, using the production NMFModel.

May be imported directly from this directory for a lightweight notebook kernel.
Ground truth is only accessed by evaluation functions, never by the optimizer.
"""
from pathlib import Path
import json
import time
import inspect

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

if __package__:
    from .mapping_optimizer import NMFModel
    from .integration_analysis import evaluate_transfer, split_genes, knn_transfer
else:
    from mapping_optimizer import NMFModel
    from integration_analysis import evaluate_transfer, split_genes, knn_transfer


MODES = ("shared", "separate", "none", "fixed_random")


def validate_optimizer_api():
    """Fail before reading data or writing results when an old module is loaded."""
    if "reconstruction_mode" not in inspect.signature(NMFModel.__init__).parameters:
        source = inspect.getfile(NMFModel)
        raise RuntimeError(
            f"Incompatible NMFModel loaded from {source}. "
            "Update spnmf/mapping_optimizer.py together with spnmf/h_ablation.py, "
            "then restart the notebook kernel and run all cells. "
            "Do not remove reconstruction_mode: it defines the ablation controls."
        )


def match_programs(learned, true, *, top_n=15, n_permutations=100, seed=0):
    """Match programs by cosine with Hungarian assignment; recompute null matching.

    Gene permutations preserve within-H loading distributions. Null scores are
    descriptive controls, not independent biological-replicate p-values.
    """
    learned, true = np.asarray(learned, float), np.asarray(true, float)
    if learned.ndim != 2 or true.ndim != 2 or learned.shape[1] != true.shape[1] or not len(learned) or not len(true):
        raise ValueError("Recovery requires nonempty programs and the same ordered training genes")
    if not np.isfinite(learned).all() or not np.isfinite(true).all():
        raise ValueError("Program loadings must be finite")
    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    x, y = norm(learned), norm(true)
    similarity = x @ y.T
    rows, cols = linear_sum_assignment(-similarity)
    k = min(top_n, learned.shape[1])
    table = []
    for i, j in zip(rows, cols):
        a, b = set(np.argsort(learned[i])[-k:]), set(np.argsort(true[j])[-k:])
        table.append(dict(learned_program=int(i), true_program=int(j), cosine=float(similarity[i, j]),
                          top_gene_jaccard=len(a & b) / len(a | b), learned_K=len(learned), true_K=len(true),
                          matched_fraction_learned=len(rows)/len(learned), matched_fraction_true=len(rows)/len(true),
                          penalized_precision=float(similarity[rows, cols].sum()/len(learned)),
                          penalized_recall=float(similarity[rows, cols].sum()/len(true))))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        score = x[:, rng.permutation(x.shape[1])] @ y.T
        r, c = linear_sum_assignment(-score)
        null.append(float(score[r, c].mean()))
    return pd.DataFrame(table), np.array(null)


def _pcc(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _composition(M, T):
    x = M.T @ T
    return x / np.maximum(x.sum(axis=1, keepdims=True), 1e-12)


def run_ablation_suite(data_directory, output_directory, *, scenarios=None, seeds=(0, 1, 2),
                       epochs=1000, learning_rate=.1, split_seed=42, test_fraction=.2,
                       modes=MODES, device="cpu", n_neighbors=15, n_permutations=100):
    """Same training genes, W/V initialization, type prior and optimizer for all modes.

    All scenarios use the same gene split after a common detection filter, keeping
    paired 02/04 comparable. Fixed training budget; never select on test metrics.
    This benchmark assumes the generated shared gene IDs/reference type encoding.
    """
    from itertools import combinations
    validate_optimizer_api()
    data_directory, out = Path(data_directory), Path(output_directory)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = list(scenarios or [p.name for p in sorted(data_directory.iterdir()) if (p / "simulation_arrays.npz").is_file()])
    if not scenarios or not seeds or any(m not in MODES for m in modes):
        raise ValueError("Provide scenarios, seeds and valid modes")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    datasets = {}
    eligible = None
    for name in scenarios:
        with np.load(data_directory / name / "simulation_arrays.npz", allow_pickle=False) as f:
            d = {k: f[k] for k in f.files}
        datasets[name] = d
        Xref, Xsp = d["reference_spliced"] + d["reference_unspliced"], d["observed_spliced"] + d["observed_unspliced"]
        detected = set(d["genes"][(Xref.sum(axis=0) > 0) & (Xsp.sum(axis=0) > 0)])
        eligible = detected if eligible is None else eligible & detected
    order = datasets[scenarios[0]]["genes"].tolist()
    if any(d["genes"].tolist() != order for d in datasets.values()):
        raise ValueError("This suite requires consistent gene ordering")
    training, testing = split_genes([g for g in order if g in eligible], test_fraction, split_seed)
    tr, te = [order.index(g) for g in training], [order.index(g) for g in testing]
    (out / "gene_split.json").write_text(json.dumps(dict(training_genes=training, test_genes=testing), indent=2))
    metrics, summary, recovery, history, null_records, stability, intervention = [], [], [], [], [], [], []
    for name, data in datasets.items():
        print(f"Scenario: {name}", flush=True)
        S = (data["reference_spliced"] + data["reference_unspliced"])[:, tr].astype(np.float32)
        G = (data["observed_spliced"] + data["observed_unspliced"])[:, tr].astype(np.float32)
        T = np.eye(6, dtype=np.float32)[data["reference_types"]]
        density = G.sum(axis=1)
        if (density <= 0).any() or (S.sum(axis=1) <= 0).any():
            raise ValueError("Zero training library; change QC/split explicitly, do not silently drop spots")
        density = density / density.sum()
        spot_ids = [f"Spot_{i+1:05d}" for i in range(len(G))]
        cell_ids = [f"Reference_{i+1:05d}" for i in range(len(S))]
        channels = ("spliced", "unspliced")
        sources = {c: pd.DataFrame(data[f"reference_{c}"][:, te], index=cell_ids, columns=testing) for c in channels}
        truth = {c: pd.DataFrame(data[f"true_{c}"][:, te], index=spot_ids, columns=testing) for c in channels}
        true_comp = data["composition_counts"] / data["n_cells"][:, None]
        all_H, all_predictions = {}, {}
        for seed in seeds:
            for mode in modes:
                torch.manual_seed(seed)
                model = NMFModel(S, G, T, density, torch.device(device), reconstruction_mode=mode)
                start = time.perf_counter()
                M, losses = model.train_model(epochs - 1, learning_rate=learning_rate, print_each=None)
                runtime = time.perf_counter() - start
                factors = model.export_factors()
                pred = {c: pd.DataFrame(M.T @ sources[c].to_numpy(), index=spot_ids, columns=testing) for c in channels}
                scores = evaluate_transfer(truth, {mode: pred}, training_genes=training, test_genes=testing)
                scores["scenario"], scores["seed"] = name, seed
                metrics.append(scores)
                comp = _composition(M, T)
                base = dict(scenario=name, seed=seed, method=mode, runtime_seconds=runtime,
                            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                            composition_rmse=float(np.sqrt(np.mean((comp - true_comp) ** 2))),
                            mean_composition_pcc=float(np.nanmean([_pcc(comp[:, i], true_comp[:, i]) for i in range(6)])),
                            rare_type_rmse=float(np.sqrt(np.mean((comp[:, 5] - true_comp[:, 5]) ** 2))))
                for c in channels:
                    sub = scores[scores.channel == c]
                    base[f"{c}_pcc"] = sub.pcc.median()
                    base[f"{c}_spearman"] = sub.spearman.median()
                    base[f"{c}_pattern_rmse"] = sub.pattern_rmse.median()
                base["fraction_mae"] = scores[scores.channel == "joint"].fraction_mae.median()
                summary.append(base)
                all_predictions[(seed, mode)] = np.concatenate([pred[c].to_numpy().ravel() for c in channels])
                losses = pd.DataFrame(losses).assign(epoch=np.arange(epochs), scenario=name, seed=seed, method=mode)
                history.append(losses)
                folder = out / name / f"seed_{seed}" / mode
                folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(folder / "fit.npz", M=M, composition=comp, **factors,
                                    training_genes=np.array(training), testing_genes=np.array(testing))
                for key in ("H", "H_sc", "H_sp"):
                    if key not in factors:
                        continue
                    all_H[(seed, mode, key)] = factors[key]
                    matched, null = match_programs(factors[key], data["program_H"][:, tr],
                                                   n_permutations=n_permutations, seed=split_seed)
                    matched = matched.assign(scenario=name, seed=seed, method=mode, dictionary=key)
                    recovery.append(matched)
                    null_records.append(dict(scenario=name, seed=seed, method=mode, dictionary=key,
                                             observed_mean_cosine=matched.cosine.mean(),
                                             permutation_mean=float(null.mean()) if len(null) else np.nan,
                                             permutation_q95=float(np.quantile(null, .95)) if len(null) else np.nan))
                if mode == "shared":
                    with torch.no_grad():
                        model._H.add_(10.0)
                        _, changed = model._loss_fn(verbose=False)
                        delta = float(np.max(np.abs(changed.cpu().numpy() - M)))
                    intervention.append(dict(scenario=name, seed=seed, intervention="change_H_with_WV_fixed",
                                             max_mapping_change=delta))
                print(f"  {mode}, seed={seed}: S PCC={base['spliced_pcc']:.3f}, U PCC={base['unspliced_pcc']:.3f}", flush=True)
        # kNN controls use exactly the same training features and held-out counts.
        for weighted, label in ((False, "kNN"), (True, "Weighted-kNN")):
            pred = knn_transfer(pd.DataFrame(S, index=cell_ids, columns=training),
                                pd.DataFrame(G, index=spot_ids, columns=training), sources, training,
                                n_neighbors=n_neighbors, weighted=weighted)
            scores = evaluate_transfer(truth, {label: pred}, training_genes=training, test_genes=testing)
            scores["scenario"], scores["seed"] = name, -1
            metrics.append(scores)
            row = dict(scenario=name, seed=-1, method=label)
            for c in channels:
                sub = scores[scores.channel == c]
                row[f"{c}_pcc"] = sub.pcc.median()
                row[f"{c}_spearman"] = sub.spearman.median()
                row[f"{c}_pattern_rmse"] = sub.pattern_rmse.median()
            row["fraction_mae"] = scores[scores.channel == "joint"].fraction_mae.median()
            summary.append(row)
        for mode in modes:
            for a, b in combinations(seeds, 2):
                stability.append(dict(scenario=name, method=mode, seed_a=a, seed_b=b,
                                      prediction_correlation=_pcc(all_predictions[(a, mode)], all_predictions[(b, mode)])))
                for key in ("H", "H_sc", "H_sp"):
                    if (a, mode, key) in all_H:
                        matches, _ = match_programs(all_H[(a, mode, key)], all_H[(b, mode, key)], n_permutations=0)
                        stability[-1][f"{key}_matched_cosine"] = matches.cosine.mean()
    tables = dict(gene_metrics=pd.concat(metrics, ignore_index=True), summary=pd.DataFrame(summary),
                  program_recovery=pd.concat(recovery, ignore_index=True) if recovery else pd.DataFrame(),
                  program_null=pd.DataFrame(null_records), training_history=pd.concat(history, ignore_index=True),
                  stability=pd.DataFrame(stability), intervention=pd.DataFrame(intervention))
    for key, frame in tables.items():
        frame.to_csv(out / f"{key}.csv", index=False)
    config = dict(scenarios=scenarios, seeds=list(seeds), epochs=epochs, learning_rate=learning_rate,
                  split_seed=split_seed, test_fraction=test_fraction, modes=list(modes), device=device,
                  n_neighbors=n_neighbors, n_permutations=n_permutations,
                  evaluation_target="unthinned_molecular_counts", torch_version=str(torch.__version__),
                  design="fixed budget, common split, matched W/V starts; seeds are initialization replicates")
    (out / "run.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return tables


def plot_ablation_results(tables):
    """Show conditional contribution, not only the appearance of H."""
    import matplotlib.pyplot as plt
    summary = tables["summary"]
    scenarios = summary.scenario.unique()
    modes = [m for m in MODES if m in set(summary.method)] + [m for m in ("kNN", "Weighted-kNN") if m in set(summary.method)]
    fig, axes = plt.subplots(2, len(scenarios), figsize=(5 * len(scenarios), 8), squeeze=False, layout="constrained")
    for j, scenario in enumerate(scenarios):
        for i, metric in enumerate(("spliced_pcc", "unspliced_pcc")):
            ax = axes[i, j]
            for k, mode in enumerate(modes):
                values = summary.loc[(summary.scenario == scenario) & (summary.method == mode), metric].dropna().to_numpy()
                ax.scatter(np.full(len(values), k), values, s=28, alpha=.7)
                if len(values):
                    ax.plot([k - .25, k + .25], [values.mean()] * 2, color="black")
            ax.set_xticks(range(len(modes)), modes, rotation=45, ha="right")
            ax.set(title=scenario, ylabel=metric, ylim=(-1, 1))
    fig.suptitle("Held-out S/U transfer | dots = initialization seeds, bars = mean")
    delta_fig, delta_axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for ax, metric in zip(delta_axes, ("spliced_pcc", "unspliced_pcc")):
        pivot = summary[summary.seed >= 0].pivot(index=["scenario", "seed"], columns="method", values=metric)
        alternatives = [m for m in ("separate", "none", "fixed_random") if m in pivot]
        for j, mode in enumerate(alternatives):
            for scenario_index, scenario in enumerate(scenarios):
                values = (pivot.shared - pivot[mode]).loc[scenario].to_numpy()
                ax.scatter(np.full(len(values), j), values, color=plt.get_cmap("tab10")(scenario_index),
                           label=scenario if j == 0 else None)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xticks(range(len(alternatives)), [f"shared - {m}" for m in alternatives], rotation=15)
        ax.set_ylabel(f"Paired delta {metric}")
    delta_axes[0].legend(fontsize=7)
    delta_fig.suptitle("Positive supports shared H under this setting; negative does not")
    return {"ablation_transfer": fig, "paired_H_contribution": delta_fig}


def plot_program_recovery(data_directory, fit_directory, *, n_top=12):
    """Matched true/learned H plus true/inferred spatial activity per program."""
    import matplotlib.pyplot as plt
    with np.load(Path(data_directory) / "simulation_arrays.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    with np.load(Path(fit_directory) / "fit.npz", allow_pickle=False) as f:
        fit = {k: f[k] for k in f.files}
    if "H" not in fit:
        raise ValueError("Choose a shared or fixed_random fit for this paired H plot")
    indices = [data["genes"].tolist().index(g) for g in fit["training_genes"]]
    true = data["program_H"][:, indices]
    learned = fit["H"]
    match, _ = match_programs(learned, true)
    match = match.sort_values("true_program")
    order = match.learned_program.tolist()
    order += [i for i in range(len(learned)) if i not in order]
    learned = learned[order]
    selected = list(dict.fromkeys(np.argsort(true, axis=1)[:, -n_top:].ravel().tolist()))
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(selected) * .18), 6), layout="constrained")
    norm_true = true / np.maximum(true.sum(axis=1, keepdims=True), 1e-12)
    norm_fit = learned / np.maximum(learned.sum(axis=1, keepdims=True), 1e-12)
    vmax = max(norm_true[:, selected].max(), norm_fit[:, selected].max())
    learned_labels = []
    for i in order:
        hit = match[match.learned_program == i]
        learned_labels.append(f"L{i+1}->T{int(hit.true_program.iloc[0])+1}" if len(hit) else f"L{i+1} unmatched")
    for ax, values, title, labels in zip(axes, (norm_true, norm_fit),
            ("Generator H (all programs)", "Learned H (all programs, including unmatched)"),
            ([f"T{i+1}" for i in range(len(true))], learned_labels)):
        im = ax.imshow(values[:, selected], aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(selected)), np.array(fit["training_genes"])[selected], rotation=90, fontsize=6)
        ax.set_yticks(range(len(values)), labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"True K={len(true)}, fitted K={len(learned)}; matched pairs={len(match)}")
    spatial, axes = plt.subplots(len(match), 2, figsize=(8, max(3, 3*len(match))), squeeze=False, layout="constrained")
    xy = data["coordinates"]
    for row, (_, item) in enumerate(match.iterrows()):
        a = data["program_activity"][:, int(item.true_program)]
        b = fit["V"][:, int(item.learned_program)]
        for col, (values, title) in enumerate(((a, "Generator activity"), (b, "Inferred V activity"))):
            scaled = values / max(values.max(), 1e-12)
            axes[row, col].scatter(xy[:, 0], xy[:, 1], c=scaled, vmin=0, vmax=1, cmap="viridis", s=10)
            axes[row, col].set(title=f"P{row+1}: {title}", aspect="equal")
            axes[row, col].axis("off")
        axes[row, 1].set_title(f"P{row+1}: V; spatial PCC={_pcc(a,b):.2f}")
    spatial.suptitle(f"Matched activity pairs only ({len(match)}/{len(true)} true, {len(match)}/{len(learned)} fitted); individually scaled")
    return {"matched_H_recovery": fig, "matched_program_spatial": spatial}


def plot_spatial_ablation(data_directory, scenario_results, *, seed=0):
    """Spatial held-out pattern errors; shows both benefits and damage of H."""
    import matplotlib.pyplot as plt
    with np.load(Path(data_directory) / "simulation_arrays.npz", allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    errors = {}
    for mode in MODES:
        with np.load(Path(scenario_results) / f"seed_{seed}" / mode / "fit.npz", allow_pickle=False) as f:
            M = f["M"]
            genes = f["testing_genes"].tolist()
        ix = [data["genes"].tolist().index(g) for g in genes]
        errors[mode] = {}
        for channel in ("spliced", "unspliced"):
            pred = M.T @ data[f"reference_{channel}"][:, ix]
            actual = data[f"true_{channel}"][:, ix].astype(float)
            valid = (actual.mean(axis=0) > 0) & (pred.mean(axis=0) > 0)
            p, t = pred[:, valid], actual[:, valid]
            errors[mode][channel] = np.sqrt(np.mean((p / p.mean(axis=0) - t / t.mean(axis=0)) ** 2, axis=1))
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), layout="constrained")
    xy = data["coordinates"]
    for row, channel in enumerate(("spliced", "unspliced")):
        maximum = max(errors[m][channel].max() for m in MODES)
        for col, mode in enumerate(MODES):
            im = axes[row, col].scatter(xy[:, 0], xy[:, 1], c=errors[mode][channel],
                                        vmin=0, vmax=maximum, s=12, cmap="magma")
            axes[row, col].set_title(f"{mode}: {channel} error")
            fig.colorbar(im, ax=axes[row, col], shrink=.7)
        delta = errors["none"][channel] - errors["shared"][channel]
        bound = max(abs(delta).max(), 1e-6)
        im = axes[row, 4].scatter(xy[:, 0], xy[:, 1], c=delta, vmin=-bound, vmax=bound, s=12, cmap="RdBu_r")
        axes[row, 4].set_title("No-reconstruction minus shared\nPositive/red = shared helps")
        fig.colorbar(im, ax=axes[row, 4], shrink=.7)
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("All held-out genes, fixed seed; shape RMSE at each spot")
    return {"spatial_H_contribution": fig}


def plot_recovery_null(tables):
    import matplotlib.pyplot as plt
    data = tables["program_null"]
    data = data[(data.method == "shared") & (data.dictionary == "H")]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    scenarios = list(data.scenario.unique())
    for i, scenario in enumerate(scenarios):
        sub = data[data.scenario == scenario]
        ax.scatter(np.full(len(sub), i) - .1, sub.observed_mean_cosine, color="#3978B5", label="Learned H" if i == 0 else None)
        ax.scatter(np.full(len(sub), i) + .1, sub.permutation_q95, marker="x",
                   color="#D78036",
                   label="95th percentile of rematched gene-permutation null" if i == 0 else None)
    ax.set_xticks(range(len(scenarios)), scenarios, rotation=15)
    ax.set_ylabel("Mean matched cosine to generator H")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Does H recover more gene structure than a rematched null?")
    return {"H_recovery_vs_null": fig}
