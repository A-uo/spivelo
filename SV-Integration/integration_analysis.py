"""Held-out count-transfer evaluation and shared H-program inspection.

Core calculations require only numpy/pandas. Plotting imports matplotlib lazily.
All count frames use spots/cells as rows and gene names as columns.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def _frame(value, name):
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a labelled pandas DataFrame")
    if not value.index.is_unique or not value.columns.is_unique or value.empty:
        raise ValueError(f"{name} must be nonempty with unique row/column names")
    a = value.to_numpy(dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    return value.astype(float)


def layer_frame(adata, layer, genes=None):
    """Extract a layer without modifying AnnData; densifies only selected genes."""
    view = adata if genes is None else adata[:, list(genes)]
    x = view.layers[layer]
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return _frame(pd.DataFrame(x, index=view.obs_names, columns=view.var_names), layer)


def split_genes(genes, test_fraction=0.2, random_state=0):
    """Deterministic train/test split; gene selection must precede evaluation."""
    genes = list(genes)
    if len(genes) < 2 or len(set(genes)) != len(genes):
        raise ValueError("At least two unique genes are required")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie between zero and one")
    n = min(len(genes) - 1, max(1, int(round(len(genes) * test_fraction))))
    chosen = set(np.random.default_rng(random_state).choice(len(genes), n, replace=False))
    return ([g for i, g in enumerate(genes) if i not in chosen],
            [g for i, g in enumerate(genes) if i in chosen])


def project_counts(mapping, layers):
    """Apply a cells x spots row-stochastic mapping to BOTH raw count layers.

    Works with SV-Integration or Tangram cell-mode mappings. Rows are explicitly
    aligned by cell ID. Output is weighted aggregate abundance, not integer counts.
    """
    mapping = _frame(mapping, "mapping")
    if not np.allclose(mapping.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Mapping must have cells as rows, each summing to one")
    result = {}
    for channel in ("spliced", "unspliced"):
        counts = _frame(layers[channel], channel)
        if set(mapping.index) != set(counts.index):
            raise ValueError("Mapping and counts must contain exactly the same cell IDs")
        result[channel] = pd.DataFrame(
            mapping.to_numpy().T @ counts.loc[mapping.index].to_numpy(),
            index=mapping.columns, columns=counts.columns)
    if not result["spliced"].columns.equals(result["unspliced"].columns):
        raise ValueError("Both count layers must use the same ordered genes")
    return result


def knn_transfer(sc_expression, spatial_expression, layers, training_genes,
                 n_neighbors=15, weighted=False, batch_size=256):
    """Spot-to-cell cosine kNN using log1p library-normalized TRAINING genes.

    Each spot receives a weighted mean of reference-cell raw counts. This differs
    in units from M.T @ counts, so evaluation reports scale-free spatial patterns
    and normalized errors, never compares uncalibrated raw-count RMSE.
    The identical neighbor weights transfer both layers. No spatial coordinates
    or evaluation genes are used for neighbor selection or library normalization.
    """
    sc_expression = _frame(sc_expression, "sc_expression")
    spatial_expression = _frame(spatial_expression, "spatial_expression")
    genes = list(training_genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Provide unique training genes")
    if not isinstance(n_neighbors, int) or not 1 <= n_neighbors <= len(sc_expression):
        raise ValueError("n_neighbors must be an integer between 1 and n_cells")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    def features(frame):
        a = frame.loc[:, genes].to_numpy()
        total = a.sum(axis=1, keepdims=True)
        if (total <= 0).any():
            raise ValueError("Zero training-gene library: filter observations before benchmarking")
        a = np.log1p(1e4 * a / total)
        return a / np.linalg.norm(a, axis=1, keepdims=True)

    ref, query = features(sc_expression), features(spatial_expression)
    source = {}
    for channel in ("spliced", "unspliced"):
        source[channel] = _frame(layers[channel], channel)
        if set(source[channel].index) != set(sc_expression.index):
            raise ValueError("Expression and layers must contain the same cell IDs")
        source[channel] = source[channel].loc[sc_expression.index]
    if not source["spliced"].columns.equals(source["unspliced"].columns):
        raise ValueError("Both layers must have identical ordered genes")
    out = {key: [] for key in source}
    for start in range(0, len(query), batch_size):
        distance = np.maximum(0, 1 - query[start:start + batch_size] @ ref.T)
        # Stable ordering makes ties reproducible for a fixed reference order.
        ids = np.argsort(distance, axis=1, kind="stable")[:, :n_neighbors]
        dist = np.take_along_axis(distance, ids, axis=1)
        weights = np.ones_like(dist)
        if weighted:
            weights = 1 / np.maximum(dist, 1e-12)
            exact = dist <= 1e-12
            has_exact = exact.any(axis=1)
            weights[has_exact] = exact[has_exact]
        weights /= weights.sum(axis=1, keepdims=True)
        for key, frame in source.items():
            out[key].append(np.einsum("bk,bkg->bg", weights, frame.to_numpy()[ids]))
    return {key: pd.DataFrame(np.concatenate(parts), index=spatial_expression.index,
                             columns=source[key].columns) for key, parts in out.items()}


def _corr(x, y, rank=False):
    if rank:
        x = pd.Series(x).rank().to_numpy()
        y = pd.Series(y).rank().to_numpy()
    x, y = x - np.mean(x), y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 0 and len(x) >= 3 else np.nan


def evaluate_transfer(truth, predictions, *, training_genes, test_genes,
                      min_total=5.0):
    """Return gene-level metrics for a method -> {spliced, unspliced} mapping.

    Correlations use original abundance across spots. pattern_rmse divides each
    gene by its own spatial mean (shape error; NOT count-calibration accuracy).
    Joint fraction MAE compares u/(u+s) only where measured u+s >= min_total;
    zero predicted totals are excluded and coverage is explicitly reported.
    Test-gene correlations that are undefined remain NaN, never silently zero.
    """
    genes = list(test_genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Provide nonempty unique test_genes")
    if set(genes) & set(training_genes):
        raise ValueError("Test genes overlap training genes (evaluation leakage)")
    if not np.isfinite(min_total) or min_total <= 0:
        raise ValueError("min_total must be finite and positive")
    if not predictions:
        raise ValueError("At least one prediction method is required")
    actual = {c: _frame(truth[c], c).loc[:, genes] for c in ("spliced", "unspliced")}
    spots = actual["spliced"].index
    if set(spots) != set(actual["unspliced"].index):
        raise ValueError("Truth channels must contain the same spot IDs")
    actual["unspliced"] = actual["unspliced"].loc[spots]
    rows = []
    for method, layers in predictions.items():
        predicted = {}
        for channel in actual:
            frame = _frame(layers[channel], f"{method}/{channel}")
            if set(frame.index) != set(spots):
                raise ValueError(f"{method}: prediction/truth spot IDs differ")
            predicted[channel] = frame.loc[spots, genes]
            for gene in genes:
                y, p = actual[channel][gene].to_numpy(), predicted[channel][gene].to_numpy()
                rmse = (float(np.sqrt(np.mean((y / y.mean() - p / p.mean()) ** 2)))
                        if y.mean() > 0 and p.mean() > 0 else np.nan)
                rows.append(dict(method=method, channel=channel, gene=gene,
                                 pcc=_corr(y, p), spearman=_corr(y, p, True),
                                 pattern_rmse=rmse, observed_mean=y.mean(),
                                 observed_zero_fraction=np.mean(y == 0), n_spots=len(y)))
        for gene in genes:
            s, u = (actual[c][gene].to_numpy() for c in ("spliced", "unspliced"))
            ps, pu = (predicted[c][gene].to_numpy() for c in ("spliced", "unspliced"))
            eligible = s + u >= min_total
            valid = eligible & (ps + pu > 0)
            error = (float(np.mean(np.abs(pu[valid] / (ps + pu)[valid]
                                         - u[valid] / (s + u)[valid]))) if valid.any() else np.nan)
            rows.append(dict(method=method, channel="joint", gene=gene,
                             fraction_mae=error, n_eligible=int(eligible.sum()),
                             n_valid=int(valid.sum()),
                             coverage=float(valid.sum() / eligible.sum()) if eligible.any() else np.nan))
    return pd.DataFrame(rows)


def summarize_transfer(scores):
    """Descriptive gene medians and valid-gene counts; not replicate-level CIs."""
    metrics = [c for c in ("pcc", "spearman", "pattern_rmse", "fraction_mae", "coverage")
               if c in scores]
    return scores.groupby(["method", "channel"], sort=False)[metrics].agg(["median", "count"])


def top_program_genes(H, n_top=15):
    """Export raw and row-L1-normalized H loadings; no functional claims inferred."""
    H = _frame(H, "H")
    if not isinstance(n_top, int) or n_top < 1:
        raise ValueError("n_top must be a positive integer")
    rows = []
    for program, values in H.iterrows():
        total = values.sum()
        for rank, (gene, value) in enumerate(values.sort_values(ascending=False, kind="stable").head(n_top).items(), 1):
            rows.append(dict(program=program, gene=gene, rank=rank, loading=value,
                             normalized_loading=value / total if total > 0 else 0))
    return pd.DataFrame(rows)


def plot_programs(H, W, V, coordinates, *, cell_types=None, n_top=5, programs=None):
    """H top-gene heatmap, mean W by cell type, and V spatial maps.

    Row-L1-normalize H for display and rescale W/V inversely to preserve WH/VH.
    V panels have individual color scales; compare within a program, not colors
    across programs. Coordinates must be a DataFrame indexed by spot IDs.
    """
    import matplotlib.pyplot as plt
    H, W, V = (_frame(a, n) for a, n in ((H, "H"), (W, "W"), (V, "V")))
    programs = list(H.index if programs is None else programs)
    if not programs or len(set(programs)) != len(programs):
        raise ValueError("Select at least one unique program")
    if not isinstance(coordinates, pd.DataFrame) or not coordinates.index.is_unique:
        raise ValueError("coordinates must have unique spot IDs")
    coords = coordinates.loc[V.index].iloc[:, :2].to_numpy(dtype=float)
    if coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Provide two finite spatial coordinate columns")
    scale = H.sum(axis=1).replace(0, 1)
    h = H.div(scale, axis=0)
    v, w = V.loc[:, programs].mul(scale.loc[programs], axis=1), W.loc[:, programs].mul(scale.loc[programs], axis=1)
    genes = list(dict.fromkeys(top_program_genes(H.loc[programs], n_top)["gene"]))
    fig_h, ax = plt.subplots(figsize=(max(7, len(genes) * .3), max(3, len(programs) * .45)))
    im = ax.imshow(h.loc[programs, genes], aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(genes)), genes, rotation=90)
    ax.set_yticks(range(len(programs)), programs)
    ax.set_title("Shared H programs (row-L1 normalized)")
    fig_h.colorbar(im, ax=ax, label="Gene loading")
    fig_h.tight_layout()
    cols = min(3, len(programs))
    fig_v, axes = plt.subplots(int(np.ceil(len(programs) / cols)), cols,
                              figsize=(4 * cols, 3.6 * int(np.ceil(len(programs) / cols))), squeeze=False)
    for ax, program in zip(axes.flat, programs):
        im = ax.scatter(coords[:, 0], coords[:, 1], c=v[program], s=12, cmap="viridis")
        ax.set_title(program)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        fig_v.colorbar(im, ax=ax, label="Rescaled V activity")
    for ax in list(axes.flat)[len(programs):]:
        ax.set_visible(False)
    fig_v.tight_layout()
    figures = {"H_loadings": fig_h, "V_spatial": fig_v}
    if cell_types is not None:
        labels = cell_types.reindex(w.index)
        if labels.isna().any():
            raise ValueError("cell_types must cover all W cell IDs")
        means = w.groupby(labels, observed=True).mean()
        fig, ax = plt.subplots(figsize=(max(5, len(programs) * .5), max(3, len(means) * .4)))
        im = ax.imshow(means, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(programs)), programs, rotation=90)
        ax.set_yticks(range(len(means)), means.index)
        ax.set_title("Mean rescaled W by cell type (type prior used in training)")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        figures["W_cell_types"] = fig
    return figures


def plot_transfer(scores, metric="pcc"):
    """Method x channel boxplots and paired per-gene S/U scatterplots."""
    import matplotlib.pyplot as plt
    if metric not in ("pcc", "spearman", "pattern_rmse"):
        raise ValueError("Choose pcc, spearman or pattern_rmse")
    methods = list(scores["method"].unique())
    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 2), 4))
    for offset, channel, color in ((-.18, "spliced", "#3978B5"), (.18, "unspliced", "#D78036")):
        arrays = [scores.loc[(scores.method == m) & (scores.channel == channel), metric].dropna().to_numpy() for m in methods]
        boxes = ax.boxplot([a if len(a) else [np.nan] for a in arrays], positions=np.arange(len(methods)) + offset,
                          widths=.3, patch_artist=True, manage_ticks=False)
        for patch in boxes["boxes"]:
            patch.set_facecolor(color)
        ax.plot([], [], color=color, linewidth=8, label=channel)
    ax.set_xticks(range(len(methods)), methods, rotation=15)
    ax.set_ylabel(metric)
    ax.set_title("Held-out genes: spatial transfer accuracy")
    ax.legend()
    fig.tight_layout()
    paired, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        table = scores[(scores.method == method) & scores.channel.isin(["spliced", "unspliced"])].pivot(index="gene", columns="channel", values=metric).dropna()
        ax.scatter(table.spliced, table.unspliced, s=15, alpha=.6)
        low, high = (-1, 1) if metric != "pattern_rmse" else (0, max(1., table.to_numpy().max() if len(table) else 1.))
        ax.plot([low, high], [low, high], "--", color="gray")
        ax.set(xlim=(low, high), ylim=(low, high), xlabel=f"Spliced {metric}", ylabel=f"Unspliced {metric}", title=f"{method} (n={len(table)})")
    paired.tight_layout()
    return {f"transfer_{metric}": fig, f"paired_{metric}": paired}


def save_figures(figures, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")


def plot_gene_transfer(truth, predictions, coordinates, gene):
    """Two-row spatial comparison and phase portraits with common channel scales."""
    import matplotlib.pyplot as plt
    panels = {"Measured": truth, **predictions}
    spots = truth["spliced"].index
    coords = coordinates.loc[spots].iloc[:, :2].to_numpy(dtype=float)
    if coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Provide two finite spatial coordinate columns")
    fig, axes = plt.subplots(2, len(panels), figsize=(3.5 * len(panels), 7), squeeze=False)
    for row, channel in enumerate(("spliced", "unspliced")):
        # Display spatial patterns rather than incomparable aggregate/mean units.
        values = {}
        for method, layers in panels.items():
            x = _frame(layers[channel], channel).loc[spots, gene].to_numpy()
            values[method] = x / x.mean() if x.mean() else x
        vmax = max(max(x) for x in values.values()) or 1
        for col, (method, x) in enumerate(values.items()):
            ax = axes[row, col]
            im = ax.scatter(coords[:, 0], coords[:, 1], c=x, vmin=0, vmax=vmax, s=12, cmap="viridis")
            ax.set_title(f"{method}\n{gene}: {channel}")
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")
            fig.colorbar(im, ax=ax, label="Abundance / spatial mean")
    fig.tight_layout()
    phase, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.5), squeeze=False)
    for ax, (method, layers) in zip(axes.flat, panels.items()):
        s, u = (layers[c].loc[spots, gene].to_numpy() for c in ("spliced", "unspliced"))
        # A single common scalar preserves U/S within each method.
        scale = np.mean(s + u) or 1
        ax.scatter(s / scale, u / scale, s=12, alpha=.5)
        ax.set(title=method, xlabel="Spliced / mean(S+U)", ylabel="Unspliced / mean(S+U)")
    xmax = max(ax.get_xlim()[1] for ax in axes.flat)
    ymax = max(ax.get_ylim()[1] for ax in axes.flat)
    for ax in axes.flat:
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
    phase.suptitle(gene)
    phase.tight_layout()
    return {"gene_spatial": fig, "gene_phase": phase}
