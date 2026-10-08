"""Utility helpers for the cell-type-aware VELOVI workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Union
from urllib.request import urlretrieve
import warnings
import random

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler


def get_permutation_scores(save_path: Union[str, Path] = Path("data/")) -> pd.DataFrame:
    """Download and return the reference permutation score table."""
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    output_path = save_path / "permutation_scores.csv"
    if not output_path.is_file():
        urlretrieve(
            url="https://figshare.com/ndownloader/files/36658185",
            filename=output_path,
        )

    return pd.read_csv(output_path)


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
    cell_type_key: Optional[str] = None,
    cells_per_cluster: int = 10**5,
    cluster_column: str = "clusters",
    remove_clusters: Optional[Sequence[str]] = None,
    min_max_scale: bool = True,
    filter_on_r2: bool = True,
) -> AnnData:
    """Preprocess spliced/unspliced matrices for VELOVI.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.
    cell_type_key
        Optional categorical ``adata.obs`` key used later by
        ``VELOVI.setup_anndata(..., cell_type_key=...)``. This function does not
        transform the annotation, but it validates that the field exists and is categorical.
    cells_per_cluster
        Maximum number of cells to keep per cluster (cell2fate-style option).
        If a cluster has more cells, it is randomly downsampled.
    cluster_column
        Cluster column used for optional filtering/downsampling. If ``cell_type_key`` is not
        provided, this column is also used as the cell type key.
    remove_clusters
        Optional iterable of cluster labels to remove before model setup.
    min_max_scale
        Min-max scale spliced and unspliced layers independently.
    filter_on_r2
        Filter genes using deterministic scVelo fit diagnostics.

    Returns
    -------
    Preprocessed ``AnnData``.
    """
    if spliced_layer not in adata.layers:
        raise KeyError(f"{spliced_layer!r} not found in adata.layers")
    if unspliced_layer not in adata.layers:
        raise KeyError(f"{unspliced_layer!r} not found in adata.layers")

    if cell_type_key is None:
        cell_type_key = cluster_column

    if cell_type_key is not None:
        if cell_type_key not in adata.obs:
            raise KeyError(f"{cell_type_key!r} not found in adata.obs")
        if not pd.api.types.is_categorical_dtype(adata.obs[cell_type_key]):
            adata.obs[cell_type_key] = adata.obs[cell_type_key].astype("category")

    if cluster_column is not None and cluster_column in adata.obs:
        if not pd.api.types.is_categorical_dtype(adata.obs[cluster_column]):
            adata.obs[cluster_column] = adata.obs[cluster_column].astype("category")

        if remove_clusters is not None and len(remove_clusters) > 0:
            remove_set = set(map(str, remove_clusters))
            keep_mask = ~adata.obs[cluster_column].astype(str).isin(remove_set).values
            adata = adata[keep_mask].copy()

        if cells_per_cluster is not None and cells_per_cluster > 0:
            rng = np.random.default_rng(0)
            keep_indices = []
            cluster_values = adata.obs[cluster_column].astype(str).values
            for cl in np.unique(cluster_values):
                idx = np.where(cluster_values == cl)[0]
                if len(idx) > cells_per_cluster:
                    idx = rng.choice(idx, size=cells_per_cluster, replace=False)
                keep_indices.append(idx)
            keep_indices = np.concatenate(keep_indices, axis=0)
            keep_indices.sort()
            adata = adata[keep_indices].copy()

    if min_max_scale:
        adata.layers[spliced_layer] = MinMaxScaler().fit_transform(adata.layers[spliced_layer])
        adata.layers[unspliced_layer] = MinMaxScaler().fit_transform(
            adata.layers[unspliced_layer]
        )

    if filter_on_r2:
        scv.tl.velocity(adata, mode="deterministic")
        has_r2 = "velocity_r2" in adata.var.columns
        has_gamma = "velocity_gamma" in adata.var.columns
        has_velocity_genes = "velocity_genes" in adata.var.columns

        if has_r2 and has_gamma:
            gene_mask = np.logical_and(adata.var["velocity_r2"] > 0, adata.var["velocity_gamma"] > 0)
            adata = adata[:, gene_mask].copy()
        else:
            warnings.warn(
                "Skipping velocity_r2/velocity_gamma filtering because one or both columns are missing."
            )

        if has_velocity_genes:
            adata = adata[:, adata.var["velocity_genes"].astype(bool).values].copy()
        else:
            warnings.warn("Skipping velocity_genes filtering because adata.var['velocity_genes'] is missing.")

    return adata


def get_training_data_cell2fate_style(
    adata: AnnData,
    remove_clusters: Optional[Sequence[str]] = None,
    cells_per_cluster: int = 100,
    cluster_column: str = "clusters",
    min_shared_counts: int = 10,
    n_var_genes: int = 2000,
    random_seed: int = 1,
) -> AnnData:
    """Cell2fate-compatible training-data reduction pipeline.

    This mirrors the `cell2fate.utils.get_training_data` style:
    1. Optional cluster removal.
    2. Cap cells per cluster via random subsampling.
    3. Gene filtering with shared counts and dispersion.
    4. Keep layers `spliced`/`unspliced` as dense float32 arrays.
    """
    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        raise KeyError(
            "Expected raw count layers `adata.layers['spliced']` and "
            "`adata.layers['unspliced']` for cell2fate-style preprocessing."
        )
    if cluster_column not in adata.obs.columns:
        raise KeyError(f"{cluster_column!r} not found in adata.obs")

    if not pd.api.types.is_categorical_dtype(adata.obs[cluster_column]):
        adata.obs[cluster_column] = adata.obs[cluster_column].astype("category")

    adata = adata.copy()
    random.seed(a=random_seed)

    if remove_clusters is not None and len(remove_clusters) > 0:
        remove_set = set(map(str, remove_clusters))
        keep_mask = ~adata.obs[cluster_column].astype(str).isin(remove_set).values
        adata = adata[keep_mask, :].copy()

    unique_celltypes = np.unique(adata.obs[cluster_column].astype(str).values)
    keep_idx = []
    for ct in unique_celltypes:
        subset = np.where(adata.obs[cluster_column].astype(str).values == ct)[0]
        if len(subset) > cells_per_cluster:
            subset = np.array(random.sample(list(subset), cells_per_cluster), dtype=np.int64)
        keep_idx.extend(subset.tolist())
    keep_idx = np.asarray(sorted(keep_idx), dtype=np.int64)
    adata = adata[keep_idx, :].copy()

    scv.pp.filter_genes(adata, min_shared_counts=min_shared_counts)
    sc.pp.normalize_total(adata, target_sum=1e4)
    scv.pp.filter_genes_dispersion(adata, n_top_genes=n_var_genes)

    if scipy.sparse.issparse(adata.layers["spliced"]):
        adata.layers["spliced"] = np.asarray(adata.layers["spliced"].toarray(), dtype=np.float32)
    else:
        adata.layers["spliced"] = np.asarray(adata.layers["spliced"], dtype=np.float32)

    if scipy.sparse.issparse(adata.layers["unspliced"]):
        adata.layers["unspliced"] = np.asarray(adata.layers["unspliced"].toarray(), dtype=np.float32)
    else:
        adata.layers["unspliced"] = np.asarray(adata.layers["unspliced"], dtype=np.float32)

    return adata


def get_max_modules_cell2fate_style(adata: AnnData, leiden_resolution: float = 0.75) -> int:
    """Estimate max module number following cell2fate-style heuristic.

    Runs Leiden on cells using `spliced + unspliced` and returns:
    `round(1.15 * n_leiden_clusters)`.
    """
    if "spliced" not in adata.layers or "unspliced" not in adata.layers:
        raise KeyError(
            "Expected raw count layers `adata.layers['spliced']` and "
            "`adata.layers['unspliced']` for module heuristic."
        )

    adata_copy = adata.copy()
    adata_copy.X = adata_copy.layers["unspliced"] + adata_copy.layers["spliced"]
    sc.pp.normalize_total(adata_copy, target_sum=1e4)
    sc.pp.log1p(adata_copy)
    sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3.0, min_disp=0.5)
    adata_copy = adata_copy[:, adata_copy.var.highly_variable].copy()
    sc.pp.scale(adata_copy, max_value=10)
    sc.pp.neighbors(adata_copy)
    adata_copy = sc.tl.leiden(adata_copy, resolution=leiden_resolution, copy=True)
    n_leiden = len(np.unique(adata_copy.obs["leiden"]))
    return int(np.round(1.15 * n_leiden))


def get_training_data(
    adata: AnnData,
    remove_clusters: Optional[Sequence[str]] = None,
    cells_per_cluster: int = 100,
    cluster_column: str = "clusters",
    min_shared_counts: int = 10,
    n_var_genes: int = 2000,
    random_seed: int = 1,
) -> AnnData:
    """Backward-compatible alias for cell2fate-style training preprocessing."""
    return get_training_data_cell2fate_style(
        adata=adata,
        remove_clusters=remove_clusters,
        cells_per_cluster=cells_per_cluster,
        cluster_column=cluster_column,
        min_shared_counts=min_shared_counts,
        n_var_genes=n_var_genes,
        random_seed=random_seed,
    )


def get_max_modules(adata: AnnData, leiden_resolution: float = 0.75) -> int:
    """Backward-compatible alias for cell2fate-style module-count heuristic."""
    return get_max_modules_cell2fate_style(adata=adata, leiden_resolution=leiden_resolution)


def _resolve_cell_mask(
    adata: AnnData,
    cell_subset: Optional[
        Union[str, Sequence[int], Sequence[bool], np.ndarray, Callable[[pd.DataFrame], np.ndarray]]
    ] = None,
    groupby: Optional[str] = None,
) -> np.ndarray:
    """Resolve cell selection into a boolean mask of length `n_obs`."""
    if cell_subset is None or (isinstance(cell_subset, str) and cell_subset.lower() == "all"):
        return np.ones(adata.n_obs, dtype=bool)

    if callable(cell_subset):
        mask = np.asarray(cell_subset(adata.obs))
        if mask.dtype != bool or mask.shape[0] != adata.n_obs:
            raise ValueError("Callable `cell_subset` must return boolean mask with length n_obs.")
        return mask

    arr = np.asarray(cell_subset)
    if arr.dtype == bool:
        if arr.shape[0] != adata.n_obs:
            raise ValueError(f"Boolean `cell_subset` length must be n_obs={adata.n_obs}, got {arr.shape[0]}.")
        return arr

    if np.issubdtype(arr.dtype, np.integer):
        idx = arr.astype(int)
        if idx.ndim != 1:
            raise ValueError("Integer `cell_subset` must be 1D indices.")
        mask = np.zeros(adata.n_obs, dtype=bool)
        mask[idx] = True
        return mask

    labels = arr.astype(str)
    if labels.ndim != 1:
        raise ValueError("Label-based `cell_subset` must be 1D.")
    label_set = set(labels.tolist())

    if groupby is not None:
        if groupby not in adata.obs.columns:
            raise KeyError(f"`groupby='{groupby}'` not found in adata.obs.")
        return adata.obs[groupby].astype(str).isin(label_set).values

    obs_name_mask = np.asarray(adata.obs_names.astype(str).isin(label_set), dtype=bool)
    if np.any(obs_name_mask):
        return obs_name_mask

    raise ValueError(
        "Label-based `cell_subset` needs `groupby`, unless labels are valid `adata.obs_names`."
    )


def _resolve_cell_subset(
    adata: AnnData,
    cell_subset: Optional[Union[str, Sequence[int], Sequence[bool], np.ndarray, Callable[[pd.DataFrame], np.ndarray]]] = None,
    groupby: Optional[str] = None,
) -> AnnData:
    """Return a subset view/copy used for plotting.

    Supported `cell_subset`:
    - `None` or `"all"`: keep all cells.
    - sequence/array of bool with length `n_obs`.
    - sequence/array of int indices.
    - sequence of group labels (requires `groupby`).
    - callable: `fn(obs_df) -> bool mask`.
    """
    mask = _resolve_cell_mask(adata, cell_subset=cell_subset, groupby=groupby)
    if np.all(mask):
        return adata
    return adata[mask].copy()


def _ensure_scvelo_layers(
    adata: AnnData,
    spliced_layer: str = "spliced",
    unspliced_layer: str = "unspliced",
) -> None:
    """Ensure scvelo-conventional layer aliases (`Ms`, `Mu`) exist."""
    if "Ms" not in adata.layers and spliced_layer in adata.layers:
        adata.layers["Ms"] = adata.layers[spliced_layer]
    if "Mu" not in adata.layers and unspliced_layer in adata.layers:
        adata.layers["Mu"] = adata.layers[unspliced_layer]


def plot_rna_velocity_flow(
    adata: AnnData,
    cell_type_key: Optional[str] = None,
    cell_subset: Optional[Union[str, Sequence[int], Sequence[bool], np.ndarray, Callable[[pd.DataFrame], np.ndarray]]] = "all",
    groupby: Optional[str] = None,
    vkey: str = "velocity",
    basis: str = "umap",
    spatial_basis: str = "spatial",
    plot_stream: bool = True,
    plot_grid: bool = True,
    plot_spatial_stream: bool = True,
    compute_embedding_if_missing: bool = True,
    n_neighbors: int = 15,
    n_jobs: int = 8,
    color_map: str = "inferno",
    legend_loc: str = "right margin",
    stream_title: Optional[str] = None,
    grid_title: Optional[str] = None,
    spatial_stream_title: Optional[str] = None,
    spatial_prob_keys: Optional[Sequence[str]] = None,
    spatial_velocity_mode: str = "stream",
    spatial_velocity_grid_n: int = 20,
    spatial_velocity_grid_min_count: int = 3,
    spatial_velocity_key: Optional[str] = None,
    spatial_palette: Optional[dict] = None,
    spatial_spot_size: float = 5.0,
    spatial_show_img: bool = True,
    spatial_img_key: str = "hires",
    spatial_img_alpha: float = 1.0,
    spatial_image_cmap: str = "Greys_r",
    spatial_arrow_color: str = "k",
    spatial_show_colorbar: bool = True,
    spatial_colorbar_position: Optional[str] = "right",
    spatial_colorbar_tick_size: float = 8,
    spatial_legend_title_fontsize: float = 9,
    spatial_colorbar_shape: Optional[dict] = None,
    spatial_colorbar_label_kw: Optional[dict] = None,
    spatial_figsize: tuple[float, float] = (11, 8),
    spatial_stream_density: float = 2.0,
    spatial_stream_linewidth: float = 1.0,
    spatial_stream_arrowsize: float = 1.0,
    spatial_stream_arrowstyle: str = "-|>",
    spatial_stream_maxlength: float = 4.0,
    spatial_stream_integration_direction: str = "both",
    show: bool = True,
):
    """Plot RNA velocity flow on manifold and optional spatial basis.

    This function wraps the common sequence:
    1) ensure `Ms/Mu`
    2) ensure embedding (e.g. UMAP)
    3) compute `velocity_graph`
    4) plot stream/grid

    Parameters
    ----------
    cell_subset
        External control for plotting subset:
        - `"all"`/`None`: all cells
        - bool mask / int indices
        - list of labels with `groupby` (e.g. cell types)
    groupby
        Obs column used when `cell_subset` is label-based.
    plot_spatial_stream
        If `True` and `adata.obsm[f"X_{spatial_basis}"]` exists, also draw spatial stream.
    """
    ad = _resolve_cell_subset(adata, cell_subset=cell_subset, groupby=groupby)
    _ensure_scvelo_layers(ad)

    color = cell_type_key if (cell_type_key is not None and cell_type_key in ad.obs.columns) else None

    if compute_embedding_if_missing and f"X_{basis}" not in ad.obsm:
        sc.pp.pca(ad)
        sc.pp.neighbors(ad, n_neighbors=n_neighbors)
        sc.tl.umap(ad)

    # Compute velocity graph once; basis-specific projections are handled by plotting calls.
    if "hvgk" in ad.uns and "spliced_corrected" not in ad.layers:
        raise ValueError("Re-export the revised HVGK posterior to obtain spliced_corrected before plotting.")
    expression_key = "spliced_corrected" if "spliced_corrected" in ad.layers else "Ms"
    scv.tl.velocity_graph(ad, vkey=vkey, xkey=expression_key, n_jobs=n_jobs)

    axes = {}
    if basis is not None and f"X_{basis}" in ad.obsm:
        if plot_stream:
            axes["stream"] = scv.pl.velocity_embedding_stream(
                ad,
                basis=basis,
                vkey=vkey,
                color=color,
                legend_loc=legend_loc,
                title=stream_title or f"RNA velocity stream ({basis})",
                show=show,
            )
        if plot_grid:
            axes["grid"] = scv.pl.velocity_embedding_grid(
                ad,
                basis=basis,
                vkey=vkey,
                color=color,
                title=grid_title or f"RNA velocity grid ({basis})",
                show=show,
            )

    if plot_spatial_stream and f"X_{spatial_basis}" in ad.obsm:
        if spatial_prob_keys is not None and len(spatial_prob_keys) > 0:
            # Spatial arrows require 2D velocity vectors in obsm.
            # Prefer user-provided key, then `<vkey>_<spatial_basis>`, then `velocity_spatial`.
            resolved_spatial_velocity_key = spatial_velocity_key
            if resolved_spatial_velocity_key is None:
                candidate_keys = [f"{vkey}_{spatial_basis}"]
                if spatial_basis == "spatial":
                    candidate_keys.append("velocity_spatial")
                for ck in candidate_keys:
                    if ck in ad.obsm:
                        resolved_spatial_velocity_key = ck
                        break

            if resolved_spatial_velocity_key is None:
                scv.tl.velocity_embedding(ad, basis=spatial_basis, vkey=vkey)
                gen_key = f"{vkey}_{spatial_basis}"
                if gen_key in ad.obsm:
                    resolved_spatial_velocity_key = gen_key
                elif spatial_basis == "spatial" and "velocity_spatial" in ad.obsm:
                    resolved_spatial_velocity_key = "velocity_spatial"
                else:
                    raise KeyError(
                        "Unable to find or compute 2D spatial velocity in `adata.obsm`. "
                        f"Tried `{gen_key}` and `velocity_spatial`."
                    )

            fig, ax = plot_spatial_probability_with_velocity(
                adata=ad,
                prob_keys=list(spatial_prob_keys),
                velocity_key=resolved_spatial_velocity_key,
                spatial_basis=spatial_basis,
                mode=spatial_velocity_mode,
                n_grid=spatial_velocity_grid_n,
                min_count=spatial_velocity_grid_min_count,
                palette=spatial_palette,
                circle_diameter=spatial_spot_size,
                show_img=spatial_show_img,
                img_key=spatial_img_key,
                img_alpha=spatial_img_alpha,
                image_cmap=spatial_image_cmap,
                arrow_color=spatial_arrow_color,
                show_colorbar=spatial_show_colorbar,
                colorbar_position=spatial_colorbar_position,
                colorbar_tick_size=spatial_colorbar_tick_size,
                legend_title_fontsize=spatial_legend_title_fontsize,
                colorbar_shape=spatial_colorbar_shape,
                colorbar_label_kw=spatial_colorbar_label_kw,
                figsize=spatial_figsize,
                stream_density=spatial_stream_density,
                stream_linewidth=spatial_stream_linewidth,
                stream_arrowsize=spatial_stream_arrowsize,
                stream_arrowstyle=spatial_stream_arrowstyle,
                stream_maxlength=spatial_stream_maxlength,
                stream_integration_direction=spatial_stream_integration_direction,
                title=spatial_stream_title or f"RNA-state projection ({spatial_basis}; not migration)",
            )
            axes["spatial_probability_velocity"] = (fig, ax)
            if show:
                plt.show()
        else:
            axes["spatial_stream"] = scv.pl.velocity_embedding_stream(
                ad,
                basis=spatial_basis,
                vkey=vkey,
                color=color,
                legend_loc=legend_loc,
                title=spatial_stream_title or f"RNA velocity stream ({spatial_basis})",
                show=show,
            )
    elif plot_spatial_stream:
        warnings.warn(
            f"Skip spatial stream: `adata.obsm['X_{spatial_basis}']` not found.",
            UserWarning,
        )

    return axes, ad


def plot_selected_cell_velocity(
    adata: AnnData,
    cell_subset: Union[str, Sequence[int], Sequence[bool], np.ndarray, Callable[[pd.DataFrame], np.ndarray]],
    basis: str = "spatial",
    groupby: Optional[str] = None,
    vkey: str = "velocity",
    velocity_key: Optional[str] = None,
    compute_velocity_embedding_if_missing: bool = True,
    show_background: bool = True,
    background_color: str = "#D0D0D0",
    background_size: float = 12.0,
    selected_color: str = "#E64B35",
    selected_size: float = 24.0,
    arrow_color: str = "#111111",
    arrow_width: float = 0.003,
    arrow_alpha: float = 0.95,
    arrow_scale: Optional[float] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
):
    """Plot velocity arrows for a selected subset of cells on a chosen basis.

    Parameters
    ----------
    cell_subset
        Target cells. Supports bool mask, integer indices, obs_names, labels with
        `groupby`, or callable `fn(adata.obs) -> bool mask`.
    basis
        Embedding basis in `adata.obsm`. Tries `X_{basis}` first, then `{basis}`.
    velocity_key
        Optional 2D velocity key in `adata.obsm` or `adata.layers`.
        If omitted, resolves as `f"{vkey}_{basis}"` (and `velocity_spatial` for
        `basis="spatial"`), computing via `scv.tl.velocity_embedding` when enabled.
    """
    obsm_basis_key = f"X_{basis}"
    if obsm_basis_key in adata.obsm:
        coords = np.asarray(adata.obsm[obsm_basis_key], dtype=float)
    elif basis in adata.obsm:
        coords = np.asarray(adata.obsm[basis], dtype=float)
    else:
        raise KeyError(f"Embedding `{obsm_basis_key}`/`{basis}` not found in `adata.obsm`.")

    resolved_velocity_key = velocity_key
    if resolved_velocity_key is None:
        candidates = [f"{vkey}_{basis}"]
        if basis == "spatial":
            candidates.append("velocity_spatial")
        for ck in candidates:
            if ck in adata.obsm or ck in adata.layers:
                resolved_velocity_key = ck
                break

    if resolved_velocity_key is None and compute_velocity_embedding_if_missing:
        scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey)
        generated_key = f"{vkey}_{basis}"
        if generated_key in adata.obsm:
            resolved_velocity_key = generated_key
        elif basis == "spatial" and "velocity_spatial" in adata.obsm:
            resolved_velocity_key = "velocity_spatial"

    if resolved_velocity_key is None:
        raise KeyError(
            "Unable to resolve 2D velocity vectors. Provide `velocity_key` or ensure "
            f"`{vkey}_{basis}` exists in `adata.obsm`."
        )

    if resolved_velocity_key in adata.obsm:
        velocity = np.asarray(adata.obsm[resolved_velocity_key], dtype=float)
    elif resolved_velocity_key in adata.layers:
        velocity = np.asarray(adata.layers[resolved_velocity_key], dtype=float)
    else:
        raise KeyError(
            f"`{resolved_velocity_key}` not found in `adata.obsm` or `adata.layers`."
        )

    if velocity.ndim != 2 or velocity.shape[1] != 2:
        raise ValueError(
            f"`{resolved_velocity_key}` must be [n_obs, 2], got shape {tuple(velocity.shape)}."
        )

    mask = _resolve_cell_mask(adata, cell_subset=cell_subset, groupby=groupby)
    if int(np.sum(mask)) == 0:
        raise ValueError("`cell_subset` selected 0 cells.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    if show_background:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=background_size,
            c=background_color,
            alpha=0.45,
            edgecolors="none",
        )

    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=selected_size,
        c=selected_color,
        alpha=0.95,
        edgecolors="none",
    )
    ax.quiver(
        coords[mask, 0],
        coords[mask, 1],
        velocity[mask, 0],
        velocity[mask, 1],
        color=arrow_color,
        width=arrow_width,
        alpha=arrow_alpha,
        angles="xy",
        scale_units="xy",
        scale=arrow_scale,
    )
    ax.set_xlabel(f"{basis}_1")
    ax.set_ylabel(f"{basis}_2")
    ax.set_title(title or f"Selected-cell velocity ({basis}, n={int(np.sum(mask))})")
    ax.set_aspect("equal")
    if show:
        plt.show()

    return fig, ax


def plot_selected_spatial_probability_velocity_flow(
    adata: AnnData,
    prob_keys: Sequence[str],
    cell_subset: Union[str, Sequence[int], Sequence[bool], np.ndarray, Callable[[pd.DataFrame], np.ndarray]],
    groupby: Optional[str] = None,
    spatial_basis: str = "spatial",
    vkey: str = "velocity",
    spatial_velocity_key: Optional[str] = None,
    spatial_velocity_mode: str = "stream",
    n_grid: int = 20,
    min_count: int = 3,
    palette: Optional[dict] = None,
    circle_diameter: float = 5.0,
    show_img: bool = True,
    img_key: str = "hires",
    img_alpha: float = 1.0,
    image_cmap: str = "Greys_r",
    arrow_color: str = "k",
    show_colorbar: bool = True,
    colorbar_position: Optional[str] = "right",
    colorbar_tick_size: float = 8,
    legend_title_fontsize: float = 9,
    colorbar_shape: Optional[dict] = None,
    colorbar_label_kw: Optional[dict] = None,
    figsize: tuple[float, float] = (11, 8),
    stream_density: float = 2.0,
    stream_linewidth: float = 1.0,
    stream_arrowsize: float = 1.0,
    stream_arrowstyle: str = "-|>",
    stream_maxlength: float = 4.0,
    stream_integration_direction: str = "both",
    compute_velocity_embedding_if_missing: bool = True,
    title: Optional[str] = None,
):
    """Plot full probability background with flow field from selected cells only.

    Background colors are built from `prob_keys` over all cells, while velocity
    arrows/stream are computed only on `cell_subset`.
    """
    mask = _resolve_cell_mask(adata, cell_subset=cell_subset, groupby=groupby)
    if int(np.sum(mask)) == 0:
        raise ValueError("`cell_subset` selected 0 cells.")

    resolved_spatial_velocity_key = spatial_velocity_key
    if resolved_spatial_velocity_key is None:
        candidate_keys = [f"{vkey}_{spatial_basis}"]
        if spatial_basis == "spatial":
            candidate_keys.append("velocity_spatial")
        for ck in candidate_keys:
            if ck in adata.obsm:
                resolved_spatial_velocity_key = ck
                break

    if resolved_spatial_velocity_key is None and compute_velocity_embedding_if_missing:
        scv.tl.velocity_embedding(adata, basis=spatial_basis, vkey=vkey)
        gen_key = f"{vkey}_{spatial_basis}"
        if gen_key in adata.obsm:
            resolved_spatial_velocity_key = gen_key
        elif spatial_basis == "spatial" and "velocity_spatial" in adata.obsm:
            resolved_spatial_velocity_key = "velocity_spatial"
        else:
            raise KeyError(
                "Unable to find or compute 2D spatial velocity in `adata.obsm`. "
                f"Tried `{gen_key}` and `velocity_spatial`."
            )

    fig, ax = plot_spatial_probability_with_velocity(
        adata=adata,
        prob_keys=list(prob_keys),
        velocity_key=resolved_spatial_velocity_key,
        spatial_basis=spatial_basis,
        mode=spatial_velocity_mode,
        n_grid=n_grid,
        min_count=min_count,
        circle_diameter=circle_diameter,
        palette=palette,
        show_img=show_img,
        img_key=img_key,
        img_alpha=img_alpha,
        image_cmap=image_cmap,
        arrow_color=arrow_color,
        show_colorbar=show_colorbar,
        colorbar_position=colorbar_position,
        colorbar_tick_size=colorbar_tick_size,
        legend_title_fontsize=legend_title_fontsize,
        colorbar_shape=colorbar_shape,
        colorbar_label_kw=colorbar_label_kw,
        figsize=figsize,
        stream_density=stream_density,
        stream_linewidth=stream_linewidth,
        stream_arrowsize=stream_arrowsize,
        stream_arrowstyle=stream_arrowstyle,
        stream_maxlength=stream_maxlength,
        stream_integration_direction=stream_integration_direction,
        title=title or f"RNA velocity ({spatial_basis}, selected-cell flow)",
        velocity_mask=mask,
    )
    return fig, ax, mask


def _mix_prob_colors(prob: np.ndarray, colors_hex: Sequence[str]) -> np.ndarray:
    """Mix cell-type probabilities into RGB colors."""
    base = np.array([plt.matplotlib.colors.to_rgb(c) for c in colors_hex], dtype=float)  # [K, 3]
    rgb = prob @ base  # [N, 3]
    return np.clip(rgb, 0.0, 1.0)


def _plot_grid_velocity(
    ax: plt.Axes,
    coords: np.ndarray,
    velocity: np.ndarray,
    n_grid: int = 20,
    min_count: int = 3,
    arrow_color: str = "k",
):
    x, y = coords[:, 0], coords[:, 1]
    u, v = velocity[:, 0], velocity[:, 1]
    xbins = np.linspace(x.min(), x.max(), n_grid + 1)
    ybins = np.linspace(y.min(), y.max(), n_grid + 1)
    Xc, Yc, Uc, Vc = [], [], [], []
    for i in range(n_grid):
        for j in range(n_grid):
            m = (x >= xbins[i]) & (x < xbins[i + 1]) & (y >= ybins[j]) & (y < ybins[j + 1])
            if int(np.sum(m)) >= int(min_count):
                Xc.append((xbins[i] + xbins[i + 1]) / 2.0)
                Yc.append((ybins[j] + ybins[j + 1]) / 2.0)
                Uc.append(float(np.mean(u[m])))
                Vc.append(float(np.mean(v[m])))
    if len(Xc) == 0:
        return
    Xc, Yc, Uc, Vc = map(np.asarray, (Xc, Yc, Uc, Vc))
    nrm = np.sqrt(Uc**2 + Vc**2)
    vmax = np.percentile(nrm, 98) if nrm.size > 0 else 1.0
    scale = 1.0 / max(vmax, 1e-8)
    ax.quiver(Xc, Yc, Uc * scale, Vc * scale, color=arrow_color, width=0.003, alpha=0.9)


def _compute_grid_velocity_field(
    coords: np.ndarray,
    velocity: np.ndarray,
    n_grid: int = 20,
    min_count: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-cell velocity to a regular grid field for streamplot."""
    x, y = coords[:, 0], coords[:, 1]
    u, v = velocity[:, 0], velocity[:, 1]

    xedges = np.linspace(float(np.min(x)), float(np.max(x)), int(n_grid) + 1)
    yedges = np.linspace(float(np.min(y)), float(np.max(y)), int(n_grid) + 1)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    U = np.full((n_grid, n_grid), np.nan, dtype=float)
    V = np.full((n_grid, n_grid), np.nan, dtype=float)

    for i in range(n_grid):
        for j in range(n_grid):
            mask = (x >= xedges[i]) & (x < xedges[i + 1]) & (y >= yedges[j]) & (y < yedges[j + 1])
            if int(np.sum(mask)) >= int(min_count):
                U[j, i] = float(np.mean(u[mask]))
                V[j, i] = float(np.mean(v[mask]))

    valid = np.isfinite(U) & np.isfinite(V)
    if np.any(valid):
        speed = np.sqrt(U[valid] ** 2 + V[valid] ** 2)
        vmax = np.percentile(speed, 98) if speed.size > 0 else 1.0
        scale = 1.0 / max(vmax, 1e-8)
        U[valid] *= scale
        V[valid] *= scale

    Xg, Yg = np.meshgrid(xcenters, ycenters)
    return Xg, Yg, U, V


def plot_spatial_probability_with_velocity(
    adata: AnnData,
    prob_keys: Sequence[str],
    velocity_key: str = "velocity",
    spatial_basis: str = "spatial",
    mode: str = "grid",
    n_grid: int = 20,
    min_count: int = 3,
    circle_diameter: float = 5.0,
    palette: Optional[dict] = None,
    show_img: bool = True,
    img_key: str = "hires",
    img_alpha: float = 1.0,
    image_cmap: str = "Greys_r",
    arrow_color: str = "k",
    show_colorbar: bool = True,
    colorbar_position: Optional[str] = "right",
    colorbar_tick_size: float = 8,
    legend_title_fontsize: float = 9,
    colorbar_shape: Optional[dict] = None,
    colorbar_label_kw: Optional[dict] = None,
    figsize: tuple[float, float] = (11, 8),
    stream_density: float = 2.0,
    stream_linewidth: float = 1.0,
    stream_arrowsize: float = 1.0,
    stream_arrowstyle: str = "-|>",
    stream_maxlength: float = 4.0,
    stream_integration_direction: str = "both",
    title: Optional[str] = None,
    velocity_mask: Optional[np.ndarray] = None,
):
    """Plot spatial probabilities (soft composition) with velocity arrows.

    - `prob_keys`: columns in `adata.obs` representing probabilities per cell type.
    - `mode`: `"quiver"` for per-cell arrows or `"grid"` for aggregated grid arrows.
    - `velocity_mask`: optional boolean mask (`n_obs`) selecting cells that
      contribute to velocity arrows/stream.
    """
    spatial_obsm_key = f"X_{spatial_basis}"
    if spatial_obsm_key in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_obsm_key], dtype=float)
    elif spatial_basis in adata.obsm:
        coords = np.asarray(adata.obsm[spatial_basis], dtype=float)
    else:
        raise KeyError(
            f"`adata.obsm['{spatial_obsm_key}']` and `adata.obsm['{spatial_basis}']` are both missing."
        )
    for k in prob_keys:
        if k not in adata.obs.columns:
            raise KeyError(f"`{k}` not found in adata.obs.")

    prob = np.asarray(adata.obs[list(prob_keys)].values, dtype=float)
    prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-8, None)

    if palette is None:
        default_colors = [
            "#F0E442",
            "#D55E00",
            "#56B4E9",
            "#009E73",
            "#5A14A5",
            "#C8C8C8",
            "#323232",
            "#1f77b4",
            "#ff7f0e",
        ]
        colors_hex = default_colors[: len(prob_keys)]
    else:
        fallback = ["#F0E442", "#D55E00", "#56B4E9", "#009E73", "#5A14A5", "#C8C8C8", "#323232"]
        colors_hex = [palette.get(k, fallback[i % len(fallback)]) for i, k in enumerate(prob_keys)]
    rgb = _mix_prob_colors(prob, colors_hex)
    palette_safe = {k: colors_hex[i] for i, k in enumerate(prob_keys)}

    if velocity_key in adata.obsm:
        vel = np.asarray(adata.obsm[velocity_key], dtype=float)
    elif velocity_key in adata.layers:
        vel = np.asarray(adata.layers[velocity_key], dtype=float)
        if vel.ndim != 2 or vel.shape[1] != 2:
            raise ValueError("If velocity is provided in layers, it must be [n_obs, 2] for spatial plotting.")
    else:
        raise KeyError(f"`{velocity_key}` not found in adata.obsm or adata.layers.")

    if velocity_mask is None:
        velocity_mask = np.ones(adata.n_obs, dtype=bool)
    else:
        velocity_mask = np.asarray(velocity_mask, dtype=bool)
        if velocity_mask.shape[0] != adata.n_obs:
            raise ValueError(
                f"`velocity_mask` length must be n_obs={adata.n_obs}, got {velocity_mask.shape[0]}."
            )
    if int(np.sum(velocity_mask)) == 0:
        raise ValueError("`velocity_mask` selected 0 cells.")

    img = None
    scaled_coords = coords
    if "spatial" in adata.uns and len(adata.uns["spatial"]) > 0:
        spatial_meta = list(adata.uns["spatial"].values())[0]
        if show_img:
            img = spatial_meta.get("images", {}).get(img_key, None)
        # Match Visium image coordinate space when raw spatial coordinates are provided.
        if spatial_basis == "spatial" and "spatial" in adata.obsm:
            sf_key = f"tissue_{img_key}_scalef"
            sf = spatial_meta.get("scalefactors", {}).get(sf_key, None)
            if sf is not None:
                scaled_coords = np.asarray(adata.obsm["spatial"], dtype=float) * float(sf)

    fig, ax = None, None
    if show_colorbar:
        try:
            from spnmf._space import plot_spatial as spnmf_plot_spatial

            with mpl.rc_context({"figure.figsize": figsize, "axes.grid": False}):
                fig = spnmf_plot_spatial(
                    adata=adata,
                    color=list(prob_keys),
                    labels=list(prob_keys),
                    show_img=show_img,
                    img_key=img_key,
                    style="fast",
                    max_color_quantile=0.992,
                    circle_diameter=circle_diameter,
                    colorbar_position=colorbar_position,
                    palette=palette_safe,
                    legend_title_fontsize=legend_title_fontsize,
                    colorbar_tick_size=colorbar_tick_size,
                    colorbar_shape=colorbar_shape or {},
                    colorbar_label_kw=colorbar_label_kw or {},
                )
            ax = fig.axes[0]
        except Exception as e:
            warnings.warn(f"Falling back to simple scatter (colorbar disabled): {e}")
            show_colorbar = False

    if not show_colorbar:
        fig, ax = plt.subplots(figsize=figsize)
        if img is not None and show_img:
            ax.imshow(img, aspect="equal", alpha=img_alpha, origin="lower", cmap=image_cmap)
        ax.scatter(
            scaled_coords[:, 0],
            scaled_coords[:, 1],
            c=rgb,
            s=circle_diameter**2,
            alpha=0.85,
            edgecolors="none",
        )
    if mode == "quiver":
        ax.quiver(
            scaled_coords[velocity_mask, 0],
            scaled_coords[velocity_mask, 1],
            vel[velocity_mask, 0],
            vel[velocity_mask, 1],
            color=arrow_color,
            width=0.002,
            alpha=0.7,
        )
    else:
        Xg, Yg, Ug, Vg = _compute_grid_velocity_field(
            coords=scaled_coords[velocity_mask],
            velocity=vel[velocity_mask],
            n_grid=n_grid,
            min_count=min_count,
        )
        valid = np.isfinite(Ug) & np.isfinite(Vg)
        if np.any(valid):
            speed = np.sqrt(Ug**2 + Vg**2)
            lw = stream_linewidth * (0.5 + 1.5 * np.nan_to_num(speed, nan=0.0))
            ax.streamplot(
                Xg,
                Yg,
                np.ma.masked_invalid(Ug),
                np.ma.masked_invalid(Vg),
                color=arrow_color,
                density=stream_density,
                linewidth=lw,
                arrowsize=stream_arrowsize,
                arrowstyle=stream_arrowstyle,
                maxlength=stream_maxlength,
                integration_direction=stream_integration_direction,
            )
    ax.set_xlim(scaled_coords[:, 0].min(), scaled_coords[:, 0].max())
    ax.set_ylim(scaled_coords[:, 1].max(), scaled_coords[:, 1].min())
    ax.set_aspect("equal")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    return fig, ax
