"""Complete synthetic integration benchmark; truth is used only after prediction."""
from pathlib import Path
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch
if __package__:
    from .h_ablation import validate_optimizer_api, match_programs
    from .mapping_optimizer import NMFModel
    from .integration_analysis import split_genes, evaluate_transfer
else:
    from h_ablation import validate_optimizer_api, match_programs
    from mapping_optimizer import NMFModel
    from integration_analysis import split_genes, evaluate_transfer

INPUTS = ("S", "U", "total", "concat")


def feature_matrices(data, indices, input_mode):
    """Raw S, U, S+U, or [S | U]; no evaluation genes or truth used."""
    def build(prefix):
        s, u = (data[f"{prefix}_{c}"][:, indices].astype(np.float32)
                for c in ("spliced", "unspliced"))
        return {"S": s, "U": u, "total": s + u,
                "concat": np.concatenate([s, u], axis=1)}[input_mode]
    return build("reference"), build("observed")


def neighbor_weights(ref, query, k=15, weighted=False):
    """Same cosine/log normalization as the existing baseline; explicit zero fallback.

    Zero reference rows are excluded. Zero query rows receive a uniform distribution
    over nonzero references, rather than arbitrary first-k tie selection.
    """
    keep = ref.sum(axis=1) > 0
    if not keep.any():
        raise ValueError("No nonzero reference cells for this input")
    k = min(k, int(keep.sum()))
    if k < 1:
        raise ValueError("k must be positive")
    def normalize(a):
        x = np.log1p(1e4 * a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12))
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    dist = np.maximum(0, 1 - normalize(query) @ normalize(ref[keep]).T)
    ids = np.argsort(dist, axis=1, kind="stable")[:, :k]
    d = np.take_along_axis(dist, ids, axis=1)
    w = 1 / np.maximum(d, 1e-12) if weighted else np.ones_like(d)
    if weighted:
        exact = d <= 1e-12
        rows = exact.any(axis=1)
        w[rows] = exact[rows]
    w /= w.sum(axis=1, keepdims=True)
    q = np.zeros((len(query), len(ref)), dtype=np.float64)
    q[np.arange(len(query))[:, None], np.flatnonzero(keep)[ids]] = w
    zero = query.sum(axis=1) == 0
    q[zero] = keep / keep.sum()
    return q


def load_tangram(path, cells, spots, training_genes, input_mode):
    """Import a real cell-mode h5ad with honest training metadata, never relabel it."""
    import anndata as ad
    from scipy import sparse
    a = ad.read_h5ad(path)
    if (set(a.uns.get("training_genes", [])) != set(training_genes)
        or a.uns.get("input_mode") != input_mode
        or a.uns.get("density_prior") != "uniform"):
        raise ValueError("Tangram requires matching training_genes, input_mode and density_prior=uniform metadata")
    if (not a.obs_names.is_unique or not a.var_names.is_unique
        or set(a.obs_names) != set(cells) or set(a.var_names) != set(spots)):
        raise ValueError("Tangram cell/spot IDs differ")
    x = a[cells, spots].X
    m = x.toarray() if sparse.issparse(x) else np.asarray(x)
    if not np.isfinite(m).all() or (m < 0).any() or not np.allclose(m.sum(1), 1, atol=1e-4):
        raise ValueError("Tangram must contain finite nonnegative cell-by-spot row probabilities")
    return m


def run_comparison(data_dir, out, *, scenarios, seeds=(0, 1), epochs=200,
                   learning_rate=.1, split_seed=42, test_fraction=.2, n_neighbors=15,
                   device="cpu", inputs=INPUTS, tangram_paths=None, rerun=False,
                   selection_path=None):
    validate_optimizer_api()
    data_dir, out = Path(data_dir), Path(out)
    if epochs < 1 or not seeds or not scenarios or not inputs or any(i not in INPUTS for i in inputs):
        raise ValueError("Provide positive epochs, seeds, scenarios and valid inputs")
    tangram_paths = tangram_paths or {}  # keys: (scenario, input_mode, seed)
    selection = json.loads(Path(selection_path).read_text()) if selection_path else None
    if selection is not None:
        if selection.get("parameterization") != "learned_type_program" or selection.get("density_prior") != "uniform":
            raise ValueError("Incompatible K selection parameterization or density prior")
        if not set(scenarios) <= set(selection["cases"]):
            raise ValueError("K selection is missing requested scenarios")
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    config = dict(scenarios=list(scenarios), seeds=list(seeds), epochs=epochs,
                  learning_rate=learning_rate, split_seed=split_seed, test_fraction=test_fraction,
                  n_neighbors=n_neighbors, device=str(device), inputs=list(inputs),
                  density_prior="uniform", schema_version=2,
                  program_selection=selection,
                  data_sha256={s: digest(data_dir / s / "simulation_arrays.npz") for s in scenarios},
                  tangram={str(k): digest(v) for k, v in tangram_paths.items()},
                  code_sha256={name: digest(Path(__file__).with_name(name)) for name in
                               ("simulation_comparison.py", "mapping_optimizer.py", "integration_analysis.py", "h_ablation.py")})
    keys = ["summary", "gene_metrics", "composition_metrics", "program_recovery", "training_history", "input_qc"]
    if (out / "run.json").exists() and not rerun:
        old = json.loads((out / "run.json").read_text())
        if old != config:
            raise ValueError("Cached configuration/data/code differ. Choose a new OUT or set RERUN=True.")
        tables = {}
        for key in keys:
            try:
                tables[key] = pd.read_csv(out / f"{key}.csv")
            except pd.errors.EmptyDataError:
                tables[key] = pd.DataFrame()
        return tables
    out.mkdir(parents=True, exist_ok=True)
    # Invalidate an old completion marker before overwriting any output.
    if (out / "run.json").exists():
        (out / "run.json").unlink()
    datasets = {}
    eligible = None
    for scenario in scenarios:
        with np.load(data_dir / scenario / "simulation_arrays.npz", allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        datasets[scenario] = d
        detected = ((d["reference_spliced"] + d["reference_unspliced"]).sum(0) > 0) & ((d["observed_spliced"] + d["observed_unspliced"]).sum(0) > 0)
        names = set(d["genes"][detected])
        eligible = names if eligible is None else eligible & names
    ordered = datasets[scenarios[0]]["genes"].tolist()
    train, test = split_genes([g for g in ordered if g in eligible], test_fraction, split_seed)
    (out / "gene_split.json").write_text(json.dumps(dict(training_genes=train, test_genes=test), indent=2))
    records = {k: [] for k in keys}
    for scenario, d in datasets.items():
        genes = d["genes"].tolist()
        k_selected = None
        prior_weight = .5
        if selection is not None:
            if __package__:
                from .program_selection import input_fingerprint, observed_inputs
            else:
                from program_selection import input_fingerprint, observed_inputs
            chosen = selection["cases"][scenario]
            if input_fingerprint(observed_inputs(d)) != chosen["input_sha256"]:
                raise ValueError("K selection inputs do not match this dataset")
            if (set(train) != set(chosen["split"]["outer_training_genes"])
                or set(test) != set(chosen["split"]["test_genes"])):
                raise ValueError("K selection and integration must use the same outer gene split")
            k_selected = chosen["selected_K"]
            prior_weight = selection["type_prior_weight"]
        tr, te = [genes.index(g) for g in train], [genes.index(g) for g in test]
        cells = [f"Reference_{i+1:05d}" for i in range(len(d["reference_types"]))]
        spots = [f"Spot_{i+1:05d}" for i in range(len(d["n_cells"]))]
        t = np.eye(len(d["types"]), dtype=np.float32)[d["reference_types"]]
        truth = {c: pd.DataFrame(d[f"true_{c}"][:, te], index=spots, columns=test) for c in ("spliced", "unspliced")}
        true_comp = d["composition_counts"] / d["n_cells"][:, None]
        density = np.full(len(spots), 1 / len(spots), dtype=np.float32)
        for inp in inputs:
            ref, query = feature_matrices(d, tr, inp)
            records["input_qc"].append(dict(scenario=scenario, input=inp,
                zero_reference_rows=int((ref.sum(1) == 0).sum()), zero_spot_rows=int((query.sum(1) == 0).sum()),
                n_features=ref.shape[1], n_reference=len(ref), n_spots=len(query)))
            jobs = [(seed, mode) for seed in seeds for mode in
                    (("shared", "separate", "none", "fixed_random") if inp == "total" else ("shared",))]
            jobs += [(-1, "kNN"), (-1, "Weighted-kNN")]
            jobs += [(seed, "Tangram") for seed in seeds if (scenario, inp, seed) in tangram_paths]
            for seed, method in jobs:
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(torch.device(device))
                start = time.perf_counter()
                factors, losses, m = {}, None, None
                if method in ("kNN", "Weighted-kNN"):
                    q = neighbor_weights(ref, query, n_neighbors, method == "Weighted-kNN")
                    operator = q
                elif method == "Tangram":
                    m = load_tangram(tangram_paths[(scenario, inp, seed)], cells, spots, train, inp)
                    operator = m.T
                else:
                    torch.manual_seed(seed)
                    model = NMFModel(ref, query, t, density, torch.device(device), reconstruction_mode=method,
                                     n_programs=k_selected, type_prior_weight=prior_weight)
                    m, losses = model.train_model(epochs - 1, learning_rate=learning_rate, print_each=None)
                    factors = model.export_factors()
                    operator = m.T
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize(torch.device(device))
                runtime = time.perf_counter() - start
                pred = {c: pd.DataFrame(operator @ d[f"reference_{c}"][:, te], index=spots, columns=test) for c in truth}
                comp = operator @ t
                comp /= np.maximum(comp.sum(1, keepdims=True), 1e-12)
                tag = dict(scenario=scenario, input=inp, seed=seed, method=method)
                score = evaluate_transfer(truth, {method: pred}, training_genes=train, test_genes=test)
                for k, v in tag.items(): score[k] = v
                # Stratify by observed sparsity, not the unthinned truth's sparsity.
                for c in truth:
                    mask = score.channel == c
                    lookup = dict(zip(test, (d[f"observed_{c}"][:, te] == 0).mean(0)))
                    score.loc[mask, "input_zero_fraction"] = score.loc[mask, "gene"].map(lookup)
                records["gene_metrics"].append(score)
                row = dict(**tag, runtime_seconds=runtime, composition_rmse=float(np.sqrt(np.mean((comp-true_comp)**2))),
                           dominant_type_accuracy=float(np.mean(comp.argmax(1) == true_comp.argmax(1))))
                row["n_programs"] = k_selected if k_selected is not None else t.shape[1]
                row["program_parameterization"] = "learned_type_program" if selection is not None else "legacy_type_locked"
                for c in truth:
                    sub = score[score.channel == c]
                    for metric in ("pcc", "spearman", "pattern_rmse"):
                        row[f"{c}_{metric}"] = sub[metric].median()
                    row[f"{c}_valid_genes"] = int(sub.pcc.notna().sum())
                joint = score[score.channel == "joint"]
                row["fraction_mae"], row["fraction_coverage"] = joint.fraction_mae.median(), joint.coverage.median()
                records["summary"].append(row)
                for i, label in enumerate(d["types"]):
                    records["composition_metrics"].append(dict(**tag, cell_type=label,
                        rmse=float(np.sqrt(np.mean((comp[:, i]-true_comp[:, i])**2))), true_mean=float(true_comp[:, i].mean())))
                folder = out / scenario / inp / f"seed_{seed}" / method
                folder.mkdir(parents=True, exist_ok=True)
                saved = dict(composition=comp, predicted_spliced=pred["spliced"].to_numpy(),
                             predicted_unspliced=pred["unspliced"].to_numpy(), training_genes=np.array(train), testing_genes=np.array(test), **factors)
                saved["M" if m is not None else "Q"] = m if m is not None else operator
                np.savez_compressed(folder / "fit.npz", **saved)
                if losses is not None:
                    records["training_history"].append(pd.DataFrame(losses).assign(epoch=np.arange(epochs), **tag))
                # True generator H describes total expression, not U fraction or doubled features.
                if inp == "total":
                    for key in ("H", "H_sc", "H_sp"):
                        if key in factors:
                            # Rectangular matching is required when fitted and planted K differ.
                            match, null = match_programs(factors[key], d["program_H"][:, tr], n_permutations=50, seed=split_seed)
                            records["program_recovery"].append(match.assign(**tag, dictionary=key, null_q95=float(np.quantile(null, .95))))
                print(f"{scenario} | {inp} | {method} | seed={seed}: S={row['spliced_pcc']:.3f}, U={row['unspliced_pcc']:.3f}", flush=True)
    tables = {}
    for key, rows in records.items():
        tables[key] = pd.concat(rows, ignore_index=True) if rows and isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)
        tables[key].to_csv(out / f"{key}.csv", index=False)
    (out / "run.json").write_text(json.dumps(config, indent=2))
    return tables


def spatial_panels(data_dir, out, scenario, *, input_mode="total", seed=0, gene=None):
    """Prespecified held-out gene maps; same per-gene mean normalization for all methods."""
    import matplotlib.pyplot as plt
    with np.load(Path(data_dir) / scenario / "simulation_arrays.npz") as z:
        d = {k: z[k] for k in z.files}
    split = json.loads((Path(out) / "gene_split.json").read_text())
    gene = gene or split["test_genes"][0]
    if gene not in split["test_genes"]: raise ValueError("Choose a held-out gene")
    j = d["genes"].tolist().index(gene)
    datasets = {"Truth": {c: d[f"true_{c}"][:, j] for c in ("spliced", "unspliced")}}
    datasets["Observed"] = {c: d[f"observed_{c}"][:, j] for c in ("spliced", "unspliced")}
    for method in ("shared", "none", "kNN", "Weighted-kNN", "Tangram"):
        path = Path(out) / scenario / input_mode / f"seed_{-1 if 'kNN' in method else seed}" / method / "fit.npz"
        if path.exists():
            with np.load(path) as z:
                idx = z["testing_genes"].tolist().index(gene)
                datasets[method] = {c: z[f"predicted_{c}"][:, idx] for c in ("spliced", "unspliced")}
    fig, axes = plt.subplots(2, len(datasets), figsize=(3*len(datasets), 6), squeeze=False, layout="constrained")
    xy = d["coordinates"]
    for r, c in enumerate(("spliced", "unspliced")):
        values = [x[c]/max(float(x[c].mean()), 1e-12) for x in datasets.values()]
        vmax = max(float(v.max()) for v in values)
        for ax, name, val in zip(axes[r], datasets, values):
            im=ax.scatter(*xy.T, c=val, s=9, vmin=0, vmax=vmax, cmap="viridis")
            ax.set(title=f"{name} | {c}", aspect="equal"); ax.axis("off")
        fig.colorbar(im, ax=axes[r].tolist(), label="Expression / own spatial mean", shrink=.7)
    fig.suptitle(f"{scenario} | {input_mode} | held-out {gene}")
    return fig
