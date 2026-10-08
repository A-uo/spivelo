"""Reproducible, irregular spatial S/U benchmarks with independent reference cells.

This is a synthetic paired-count generator, NOT an RNA-velocity kinetic model.
It never uses a fitted mapping, reconstruction objective, or benchmark scores.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import softmax, expit


SCENARIOS = {
    "01_curved_layers": dict(geometry="crescent", biology_seed=101, observation_seed=201,
                             capture_s=.28, capture_u=.18,
                             purpose="Irregular curved laminae and continuous state gradients"),
    "02_branching_mixture": dict(geometry="branch", biology_seed=102, observation_seed=202,
                                 capture_s=.28, capture_u=.18,
                                 purpose="Branching domains, transition boundaries and mixed spots"),
    "03_rare_islands": dict(geometry="islands", biology_seed=103, observation_seed=203,
                            capture_s=.24, capture_u=.14,
                            purpose="Disconnected tissue islands, holes and localized rare type"),
    "04_branching_sparse": dict(geometry="branch", biology_seed=102, observation_seed=204,
                                capture_s=.065, capture_u=.018,
                                purpose="Same biological truth/reference as 02, reduced capture, especially U"),
}


def _coordinates(rng, geometry, n):
    """Rejection sample deformed nonconvex supports (no regular rectangular grid)."""
    accepted = []
    count = 0
    while count < n:
        xy = rng.uniform(-1.25, 1.25, size=(max(1000, n * 3), 2))
        x, y = xy.T
        theta = np.arctan2(y, x)
        radius = np.sqrt((x / 1.07) ** 2 + (y / .94) ** 2)
        edge = 1 + .12 * np.sin(3 * theta + .5) + .07 * np.cos(7 * theta)
        if geometry == "crescent":
            inside = (radius < edge) & (((x - .36) / .88) ** 2 + ((y - .10) / .64) ** 2 > 1)
            inside &= ~(((x + .55) / .12) ** 2 + ((y - .35) / .16) ** 2 < 1)
        elif geometry == "branch":
            trunk = (np.abs(x + .08 * np.sin(8 * y)) < .20) & (y < .20) & (y > -1.05)
            left = (np.abs(x + .72 * (y + .08) + .08 * np.sin(7 * y)) < .21) & (y > -.10)
            right = (np.abs(x - .74 * (y + .08) - .07 * np.cos(6 * y)) < .22) & (y > -.10)
            bridge = (np.abs(y - .43 - .07 * np.sin(7 * x)) < .10) & (np.abs(x) < .62)
            inside = (trunk | left | right | bridge) & (radius < edge)
        elif geometry == "islands":
            r1 = ((x + .52) / .61) ** 2 + ((y + .12) / .78) ** 2
            r2 = ((x - .65) / .42) ** 2 + ((y - .42) / .43) ** 2
            r3 = ((x - .48) / .40) ** 2 + ((y + .66) / .25) ** 2
            inside = (r1 < 1 + .18 * np.sin(6 * theta)) | (r2 < 1 + .12 * np.cos(9 * theta)) | (r3 < 1)
            hole = ((x + .52) / .19) ** 2 + ((y + .12) / .25) ** 2 < 1
            inside &= ~hole
        else:
            raise ValueError(f"Unknown geometry {geometry}")
        part = xy[inside]
        accepted.append(part)
        count += len(part)
    return np.concatenate(accepted)[:n]


def _spatial_fields(xy, geometry):
    x, y = xy.T
    if geometry == "crescent":
        state = np.clip((np.arctan2(y, x) + np.pi) / (2 * np.pi), 0, 1)
        radial = np.sqrt((x / 1.07) ** 2 + (y / .94) ** 2)
        depth = radial + .10 * np.sin(5 * np.arctan2(y, x))
        logits = -((depth[:, None] - np.linspace(.32, 1.10, 6)) / .15) ** 2
        logits += .4 * np.sin(6 * state[:, None] + np.arange(6))
    elif geometry == "branch":
        state = np.clip((y + 1.05) / 2.15, 0, 1)
        centers = np.array([[0, -.8], [0, -.25], [-.33, .38], [-.66, .8], [.32, .35], [.65, .8]])
        logits = -np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2) / .20
        logits += .45 * np.sin(8 * x[:, None] + np.arange(6))
    else:
        state = expit(1.8 * y + .9 * np.sin(5 * x))
        centers = np.array([[-.8, -.5], [-.7, .5], [-.2, -.2], [.65, .5], [.5, -.65], [-.25, .45]])
        logits = -np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2) / .19
        # Type 6 is confined to a small patch; reference still includes this type.
        logits[:, 5] = 2.5 - np.sum((xy - centers[5]) ** 2, axis=1) / .012
    proportions = .025 / 6 + .975 * softmax(logits, axis=1)
    return state, proportions


def generate_dataset(scenario, *, n_spots=700, n_reference=1200, n_genes=300,
                     seed_offset=0, n_programs=6, type_program_mode="legacy", program_overlap=0.0):
    """Return arrays and names, with no AnnData/PyTorch dependency.

    Spatial cells and reference cells are independently sampled from the same
    type/state family. Target counts are sums over hidden cells at each spot.
    Paired dense/sparse scenarios share geometry, cells and molecular truth.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Choose from {list(SCENARIOS)}")
    if n_spots < 30 or n_reference < 60 or n_genes < 60:
        raise ValueError("Use at least 30 spots, 60 reference cells, and 60 genes")
    config = dict(SCENARIOS[scenario])
    rng = np.random.default_rng(config["biology_seed"] + seed_offset)
    xy = _coordinates(rng, config["geometry"], n_spots)
    state, mixture = _spatial_fields(xy, config["geometry"])
    n_types = 6
    if not isinstance(n_programs, (int, np.integer)) or not 1 <= n_programs <= n_genes:
        raise ValueError("n_programs must be an integer in [1, n_genes]")
    if type_program_mode not in ("legacy", "mixed") or not 0 <= program_overlap <= 1:
        raise ValueError("Choose legacy/mixed and program_overlap in [0, 1]")
    if type_program_mode == "legacy" and n_programs != n_types:
        raise ValueError("Use type_program_mode='mixed' to decouple programs from types")
    # Program loadings plus dense background; factors are saved as generator truth.
    gene_program = np.arange(n_genes) % n_programs
    H = rng.gamma(.5, .12, (n_programs, n_genes))
    H[gene_program, np.arange(n_genes)] += rng.gamma(2.5, 1.0, n_genes)
    if program_overlap:
        for j in range(1, n_programs, 2):
            H[j] = (1-program_overlap)*H[j] + program_overlap*H[j-1]
    if type_program_mode == "legacy":
        type_program = rng.gamma(1.2, .20, (n_types, n_programs)) + np.eye(n_types) * 1.8
    else:
        # Every type can express several programs; multiple types share each one.
        type_program = rng.gamma(1.5, .6, (n_types, n_programs)) + .2
    peaks = np.linspace(.05, .95, n_programs)
    gene_scale = rng.lognormal(-.2, .65, n_genes)
    ratio_offset = rng.normal(-1.3, .6, n_genes)
    state_sensitivity = rng.normal(0, .7, n_genes)
    nonlinear = np.arange(n_genes) % 5 == 0

    def cell_means(types, times, library):
        gate = .35 + 1.2 * np.exp(-((times[:, None] - peaks) / .25) ** 2)
        activities = type_program[types] * gate
        total = (.25 + activities @ H) * gene_scale
        # 20% of genes have a state/type interaction outside exact shared NMF.
        total[:, nonlinear] *= np.exp(.40 * np.sin(7 * times[:, None]
                                                  + types[:, None] + gene_program[nonlinear]))
        fraction = expit(ratio_offset + 1.0 * (times[:, None] - .5) * state_sensitivity
                        + .45 * np.sin(2 * np.pi * times[:, None] + gene_program))
        total *= library[:, None]
        return total * (1 - fraction), total * fraction, activities * library[:, None]

    reference_types = rng.permutation(np.arange(n_reference) % n_types)
    reference_times = rng.uniform(0, 1, n_reference)
    ref_s_mean, ref_u_mean, ref_activity = cell_means(
        reference_types, reference_times, rng.lognormal(0, .25, n_reference))
    # Independent library noise, no recycling target cells into the reference.
    ref_noise = rng.gamma(14, 1 / 14, ref_s_mean.shape)
    ref_s = rng.poisson(ref_s_mean * ref_noise).astype(np.int32)
    ref_u = rng.poisson(ref_u_mean * ref_noise).astype(np.int32)

    n_cells = rng.integers(3, 10, size=n_spots)
    owner = np.repeat(np.arange(n_spots), n_cells)
    cumulative = np.cumsum(mixture[owner], axis=1)
    target_types = np.minimum((rng.random(len(owner))[:, None] > cumulative).sum(axis=1), n_types - 1)
    target_times = np.clip(state[owner] + rng.normal(0, .09, len(owner)), 0, 1)
    mean_s, mean_u, activities = cell_means(target_types, target_times, rng.lognormal(0, .25, len(owner)))
    expected_s = np.zeros((n_spots, n_genes))
    expected_u = np.zeros_like(expected_s)
    np.add.at(expected_s, owner, mean_s)
    np.add.at(expected_u, owner, mean_u)
    true_composition = np.zeros((n_spots, n_types), dtype=np.int32)
    np.add.at(true_composition, (owner, target_types), 1)
    program_activity = np.zeros((n_spots, n_programs))
    np.add.at(program_activity, owner, activities)
    noise = rng.gamma(14, 1 / 14, mean_s.shape)
    molecular_s = np.zeros((n_spots, n_genes), dtype=np.int32)
    molecular_u = np.zeros_like(molecular_s)
    np.add.at(molecular_s, owner, rng.poisson(mean_s * noise).astype(np.int32))
    np.add.at(molecular_u, owner, rng.poisson(mean_u * noise).astype(np.int32))
    # Spatial capture varies smoothly across tissue, shared profile in paired sets.
    relative_capture = np.clip(.75 + .20 * np.sin(3 * xy[:, 0]) * np.cos(4 * xy[:, 1]), .4, 1)
    capture_s = relative_capture * config["capture_s"]
    capture_u = relative_capture * config["capture_u"]
    obs_rng = np.random.default_rng(config["observation_seed"] + seed_offset)
    observed_s = obs_rng.binomial(molecular_s, capture_s[:, None]).astype(np.int32)
    observed_u = obs_rng.binomial(molecular_u, capture_u[:, None]).astype(np.int32)
    return dict(scenario=scenario, config=config | dict(seed_offset=seed_offset, n_spots=n_spots,
                n_reference=n_reference, n_genes=n_genes, generator_version="1.2",
                n_programs=n_programs, n_types=n_types, type_program_mode=type_program_mode,
                program_overlap=program_overlap),
                coordinates=xy, state=state, mixture_probabilities=mixture,
                composition_counts=true_composition, n_cells=n_cells,
                program_H=H * gene_scale, type_program=type_program, program_activity=program_activity,
                reference_activity=ref_activity, reference_types=reference_types,
                reference_times=reference_times, reference_spliced=ref_s, reference_unspliced=ref_u,
                observed_spliced=observed_s, observed_unspliced=observed_u,
                true_spliced=molecular_s, true_unspliced=molecular_u,
                expected_spliced=expected_s, expected_unspliced=expected_u,
                capture_s=capture_s, capture_u=capture_u, nonlinear_gene=nonlinear,
                genes=np.array([f"Gene_{i + 1:04d}" for i in range(n_genes)]),
                types=np.array([f"Type_{i + 1}" for i in range(n_types)]))


def write_dataset(data, directory):
    """Write ready-to-use h5ad inputs, isolated truth, metadata and a NPZ archive."""
    import anndata as ad
    from scipy.sparse import csr_matrix

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    genes = data["genes"]
    gene_frame = pd.DataFrame(index=genes)
    ref_obs = pd.DataFrame({"cell_type": pd.Categorical(data["types"][data["reference_types"]])},
                           index=[f"Reference_{i + 1:05d}" for i in range(len(data["reference_types"]))])
    spot_ids = [f"Spot_{i + 1:05d}" for i in range(len(data["coordinates"]))]
    # Inputs deliberately omit hidden type proportions, state, H and capture rates.
    ref = ad.AnnData(X=csr_matrix(data["reference_spliced"] + data["reference_unspliced"]),
                     obs=ref_obs, var=gene_frame.copy())
    spatial = ad.AnnData(X=csr_matrix(data["observed_spliced"] + data["observed_unspliced"]),
                         obs=pd.DataFrame(index=spot_ids), var=gene_frame.copy())
    truth = ad.AnnData(X=csr_matrix(data["true_spliced"] + data["true_unspliced"]),
                       obs=pd.DataFrame({"latent_state": data["state"], "n_cells": data["n_cells"],
                                         "capture_s": data["capture_s"], "capture_u": data["capture_u"]}, index=spot_ids),
                       var=pd.DataFrame({"nonlinear_gene": data["nonlinear_gene"]}, index=genes))
    for channel in ("spliced", "unspliced"):
        ref.layers[channel] = csr_matrix(data[f"reference_{channel}"])
        spatial.layers[channel] = csr_matrix(data[f"observed_{channel}"])
        truth.layers[channel] = csr_matrix(data[f"true_{channel}"])
        truth.layers[f"expected_{channel}"] = data[f"expected_{channel}"].astype(np.float32)
        capture = data["capture_s" if channel == "spliced" else "capture_u"]
        truth.layers[f"expected_observed_{channel}"] = (
            data[f"expected_{channel}"] * capture[:, None]).astype(np.float32)
    for a in (spatial, truth):
        a.obsm["spatial"] = data["coordinates"] * 1000
    truth.obsm["cell_type_proportions"] = pd.DataFrame(
        data["composition_counts"] / data["n_cells"][:, None], index=spot_ids, columns=data["types"])
    truth.obsm["program_activity"] = data["program_activity"].astype(np.float32)
    truth.uns["generator_H"] = pd.DataFrame(data["program_H"], index=[f"True_program_{i+1}" for i in range(data["program_H"].shape[0])], columns=genes)
    truth.uns["simulation_config"] = data["config"]
    for name, a in (("reference", ref), ("spatial_observed", spatial), ("spatial_truth", truth)):
        a.uns["synthetic"] = True
        a.uns["scenario"] = data["scenario"]
        a.write_h5ad(out / f"{name}.h5ad", compression="gzip")
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(out / "simulation_arrays.npz", **arrays)
    (out / "simulation_config.json").write_text(json.dumps(data["config"], indent=2), encoding="utf-8")
    truth.obsm["cell_type_proportions"].to_csv(out / "true_composition.csv")
    truth.uns["generator_H"].to_csv(out / "true_program_H.csv")
    return out


def plot_dataset(data):
    """Truth overview, with irregular supports and per-channel detection clearly shown."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    palette = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]
    xy = data["coordinates"]
    composition = data["composition_counts"] / data["n_cells"][:, None]
    entropy = -np.sum(composition * np.log(composition + 1e-12), axis=1) / np.log(6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    panels = [
        (composition.argmax(axis=1), "True dominant cell type", ListedColormap(palette)),
        (data["state"], "True continuous state", "viridis"),
        (entropy, "True composition entropy", "magma"),
        (np.log1p(data["observed_spliced"].sum(axis=1)), "Observed S library (log1p)", "viridis"),
        (np.log1p(data["observed_unspliced"].sum(axis=1)), "Observed U library (log1p)", "viridis"),
        ((data["observed_unspliced"] == 0).mean(axis=1), "Observed U zero fraction", "magma"),
    ]
    for i, (ax, (values, title, cmap)) in enumerate(zip(axes.flat, panels)):
        kwargs = {"norm": BoundaryNorm(np.arange(-.5, 6.5), 6)} if i == 0 else {}
        if i in (1, 2, 5):
            kwargs.update(vmin=0, vmax=1)
        im = ax.scatter(xy[:, 0], xy[:, 1], c=values, cmap=cmap, s=12, linewidths=0, **kwargs)
        ax.set(title=title, aspect="equal")
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, shrink=.75)
        if i == 0:
            cb.set_ticks(range(6), labels=data["types"])
    fig.suptitle(f"SYNTHETIC | {data['scenario']}", fontsize=15)
    return fig


def generate_suite(directory, *, n_spots=700, n_reference=1200, n_genes=300, seed_offset=0):
    """Generate four datasets and overview figures; never run or favor a mapping method."""
    import matplotlib.pyplot as plt
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        data = generate_dataset(scenario, n_spots=n_spots, n_reference=n_reference,
                                n_genes=n_genes, seed_offset=seed_offset)
        folder = write_dataset(data, out / scenario)
        fig = plot_dataset(data)
        fig.savefig(folder / "overview.png", dpi=180)
        fig.savefig(folder / "overview.svg")
        plt.close(fig)
        rows.append(dict(scenario=scenario, n_spots=n_spots, n_reference=n_reference,
                         n_genes=n_genes, s_zero_fraction=float((data["observed_spliced"] == 0).mean()),
                         u_zero_fraction=float((data["observed_unspliced"] == 0).mean()),
                         median_s_library=float(np.median(data["observed_spliced"].sum(axis=1))),
                         median_u_library=float(np.median(data["observed_unspliced"].sum(axis=1))),
                         mean_type6_fraction=float((data["composition_counts"][:, 5] / data["n_cells"]).mean())))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "dataset_summary.csv", index=False)
    return summary
