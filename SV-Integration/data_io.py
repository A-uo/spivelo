"""Read h5ad or standard 10x Visium Space Ranger output directories."""
from pathlib import Path

import numpy as np
import pandas as pd


def read_spatial_data(path, *, count_file="filtered_feature_bc_matrix.h5",
                      library_id="visium", load_images=True):
    """Accept h5ad, an outs directory, or a parent containing outs.

    Supports HDF5 or filtered_feature_bc_matrix/ Matrix Market counts and old/new
    tissue position CSV layouts. Gene names are not silently made unique.
    Standard 10x gene-expression counts do not supply spliced/unspliced layers.
    """
    import anndata as ad
    import scanpy as sc
    import json

    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".h5ad":
            raise ValueError("Spatial file input must be .h5ad; otherwise pass a Visium directory")
        return ad.read_h5ad(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    root = path / "outs" if (path / "outs").is_dir() else path
    matrix = root / count_file
    if matrix.is_file():
        a = sc.read_10x_h5(str(matrix))
    elif (root / "filtered_feature_bc_matrix").is_dir():
        a = sc.read_10x_mtx(str(root / "filtered_feature_bc_matrix"),
                            var_names="gene_symbols", make_unique=False)
    else:
        raise FileNotFoundError(f"No {count_file} or filtered_feature_bc_matrix directory in {root}")
    spatial = root / "spatial"
    positions = read_visium_positions(spatial)
    missing = a.obs_names.difference(positions.index)
    if len(missing):
        raise ValueError(f"{len(missing)} count barcodes have no spatial position; e.g. {list(missing[:3])}")
    positions = positions.loc[a.obs_names]
    for key in ("in_tissue", "array_row", "array_col"):
        a.obs[key] = positions[key].to_numpy()
    # AnnData uses x,y: pixel column first, pixel row second.
    a.obsm["spatial"] = positions[["pxl_col_in_fullres", "pxl_row_in_fullres"]].to_numpy(dtype=float)
    info = {"images": {}, "scalefactors": {}, "metadata": {"source": str(root.resolve())}}
    scale_file = spatial / "scalefactors_json.json"
    if scale_file.is_file():
        info["scalefactors"] = json.loads(scale_file.read_text(encoding="utf-8"))
    if load_images:
        from matplotlib.image import imread
        for resolution in ("hires", "lowres"):
            image = spatial / f"tissue_{resolution}_image.png"
            if image.is_file():
                info["images"][resolution] = imread(image)
    a.uns["spatial"] = {library_id: info}
    return a


def read_visium_positions(spatial_directory):
    """Parse both Space Ranger position CSV layouts with barcode validation."""
    directory = Path(spatial_directory)
    columns = ["barcode", "in_tissue", "array_row", "array_col",
               "pxl_row_in_fullres", "pxl_col_in_fullres"]
    modern, legacy = directory / "tissue_positions.csv", directory / "tissue_positions_list.csv"
    source = modern if modern.is_file() else legacy
    if not source.is_file():
        raise FileNotFoundError(f"No tissue_positions.csv or tissue_positions_list.csv in {directory}")
    # Some exports retain the legacy filename but include a header (and vice versa).
    # Inspect content, not the filename. Never coerce malformed data to NaN/drop it.
    raw = pd.read_csv(source, header=None, dtype=str, encoding="utf-8-sig")
    if raw.empty:
        raise ValueError("Spatial position CSV is empty")
    first = raw.iloc[0].str.strip().tolist()
    if set(columns) <= set(first):
        if len(set(first)) != len(first):
            raise ValueError("Spatial position CSV contains duplicate column names")
        positions = raw.iloc[1:].copy()
        positions.columns = first
    else:
        if raw.shape[1] != len(columns):
            raise ValueError("Headerless spatial position CSV must have exactly six columns")
        positions = raw.copy()
        positions.columns = columns
    if not set(columns) <= set(positions.columns):
        raise ValueError("Spatial position CSV lacks required columns")
    positions["barcode"] = positions.barcode.str.strip()
    if (positions.empty or positions.barcode.isna().any()
            or positions.barcode.eq("").any() or positions.barcode.duplicated().any()):
        raise ValueError("Spatial position barcodes must be present and unique")
    positions = positions.set_index("barcode")
    numeric = positions[columns[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Spatial positions must be finite")
    return numeric


def resolve_duplicate_genes(adata, policy="error"):
    """Return a copy without ambiguous symbols, or fail; never invent matching IDs.

    drop removes ALL occurrences of duplicated names, not an arbitrary first gene.
    Prefer shared stable gene IDs upstream if those duplicated symbols are needed.
    """
    if policy not in ("error", "drop"):
        raise ValueError("duplicate policy must be 'error' or 'drop'")
    duplicated = adata.var_names.duplicated(keep=False)
    if not duplicated.any():
        return adata
    names = adata.var_names[duplicated].unique().tolist()
    if policy == "error":
        raise ValueError(f"Ambiguous duplicate gene names: {names[:10]}; use stable gene IDs or policy='drop'")
    if duplicated.all():
        raise ValueError("All genes have ambiguous duplicate names")
    result = adata[:, ~duplicated].copy()
    result.uns["excluded_duplicate_gene_names"] = names
    print(f"Excluded {int(duplicated.sum())} columns with {len(names)} duplicate gene names: {names[:10]}")
    return result


def attach_splicing_layers(spatial, counts_path, *, duplicate_policy="error"):
    """Attach measured layers from h5ad/loom by exact spot and gene IDs.

    Extra source observations/genes are allowed; all target IDs must be present.
    Never strip barcode suffixes or substitute X for missing splicing counts.
    """
    import anndata as ad
    path = Path(counts_path)
    if path.suffix.lower() == ".h5ad":
        source = ad.read_h5ad(path)
    elif path.suffix.lower() == ".loom":
        source = ad.read_loom(path)
    else:
        raise ValueError("Splicing counts must be h5ad or loom")
    source = resolve_duplicate_genes(source, duplicate_policy)
    for a in (spatial, source):
        if not a.obs_names.is_unique or not a.var_names.is_unique:
            raise ValueError("Unique spot and gene IDs are required for layer alignment")
    missing_spots = spatial.obs_names.difference(source.obs_names)
    missing_genes = spatial.var_names.difference(source.var_names)
    if len(missing_spots) or len(missing_genes):
        raise ValueError(f"Splicing source missing {len(missing_spots)} spots and {len(missing_genes)} genes; "
                         "align barcode/gene naming before attaching counts")
    for layer in ("spliced", "unspliced"):
        if layer not in source.layers:
            raise ValueError(f"Splicing source lacks {layer}")
    source = source[spatial.obs_names, spatial.var_names]
    for layer in ("spliced", "unspliced"):
        spatial.layers[layer] = source.layers[layer].copy()
    return spatial
