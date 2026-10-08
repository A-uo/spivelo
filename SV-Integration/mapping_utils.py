from anndata import AnnData
import scanpy as sc
import logging
import numpy as np
from numpy import ndarray
from .utils import one_hot_encoding, annotate_gene_sparsity
import torch
from .mapping_optimizer import NMFModel
import pandas as pd
import uuid

def pp_adatas(
    adata_sc: AnnData,
    adata_sp: AnnData,
    genes: list[str] | None = None,
    gene_to_lowercase: bool = True,
) -> tuple[int, int]:
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if genes is None:
        genes = adata_sc.var_names

    if gene_to_lowercase:
        adata_sc.var_names = [g.lower() for g in adata_sc.var_names]
        adata_sp.var_names = [g.lower() for g in adata_sp.var_names]
        genes = [g.lower() for g in genes]

    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()

    genes = list(dict.fromkeys(g for g in genes if g in adata_sc.var_names and g in adata_sp.var_names))

    adata_sc.uns["training_genes"] = genes
    adata_sp.uns["training_genes"] = genes

    logging.info(
        f"{len(genes)} training genes are saved in `uns``training_genes` of both single cell and spatial Anndatas."
    )

    overlap_genes = [g for g in adata_sc.var_names if g in adata_sp.var_names]

    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes

    logging.info(
        f"{len(overlap_genes)} overlapped genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas."
    )

    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]

    logging.info(
        "uniform based density prior is calculated and saved in `obs``uniform_density` of the spatial Anndata."
    )

    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()

    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(
        rna_count_per_spot
    )

    logging.info(
        "rna count based density prior is calculated and saved in `obs``rna_count_based_density` of the spatial Anndata."
    )

    # 为两个 AnnData 生成相同的预处理ID
    preprocessing_id = str(uuid.uuid4())
    adata_sc.uns["preprocessing_id"] = preprocessing_id
    adata_sp.uns["preprocessing_id"] = preprocessing_id

    logging.info(f"Preprocessing ID {preprocessing_id} saved in both AnnData objects.")


    return len(genes), len(overlap_genes)


def map_cells_to_space(
    adata_sc: AnnData,
    adata_sp: AnnData,
    cluster_label: str,
    device: str = "cpu",
    learning_rate: float = 0.1,
    num_epochs: int = 1000,
    verbose: bool = True,
    density_prior: str | ndarray = "rna_count_based",
    call_back = None,
    random_state: int | None = None,
    n_programs: int | None = None,
    type_prior_weight: float = 0.5,
) -> AnnData:
    assert adata_sc.uns.get("preprocessing_id") == adata_sp.uns.get("preprocessing_id")

    training_genes = adata_sc.uns["training_genes"]

    S = adata_sc[:, training_genes].X
    G = adata_sp[:, training_genes].X
    S = S.toarray() if hasattr(S, "toarray") else np.asarray(S)
    G = G.toarray() if hasattr(G, "toarray") else np.asarray(G)

    T = one_hot_encoding(adata_sc.obs[cluster_label]).values

    if type(density_prior) is str:
        d_mode = density_prior
        d = np.array(adata_sp.obs[density_prior + "_density"])
    elif type(density_prior) is ndarray:
        d_mode = "customized_density"
        d = density_prior

    use_device = torch.device(device)

    if verbose:
        print_each = 100
    else:
        print_each = None

    logging.info(
        f"Begin training with {len(training_genes)} genes, {T.shape[1]} {cluster_label} and {d_mode}"
    )

    if random_state is not None:
        torch.manual_seed(random_state)
    model = NMFModel(S, G, T, d, use_device, n_programs=n_programs,
                     type_prior_weight=type_prior_weight)

    mapping_matrix, training_history = model.train_model(
        num_epochs, learning_rate, print_each, call_back
    )

    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    factors = model.export_factors()
    programs = [f"Program_{i + 1}" for i in range(factors["H"].shape[0])]
    adata_map.obsm["program_W"] = pd.DataFrame(
        factors["W"], index=adata_map.obs_names, columns=programs)
    adata_map.varm["program_V"] = pd.DataFrame(
        factors["V"], index=adata_map.var_names, columns=programs)
    adata_map.uns["program_H"] = pd.DataFrame(
        factors["H"], index=programs, columns=training_genes)
    adata_map.uns["training_genes"] = list(training_genes)
    adata_map.uns["program_type_prior"] = [str(x) for x in adata_sc.obs[cluster_label].unique()]
    adata_map.uns["n_programs"] = model.n_programs
    adata_map.uns["type_prior_weight"] = type_prior_weight
    adata_map.uns["program_parameterization"] = "legacy_type_locked" if n_programs is None else "learned_type_program"
    if "A" in factors:
        adata_map.uns["type_program_A"] = pd.DataFrame(
            factors["A"], index=adata_map.uns["program_type_prior"], columns=programs)

    G_predicted = adata_map.X.T @ S
    cos_sims = []
    for v1, v2 in zip(G.T, G_predicted.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_cs = pd.DataFrame(cos_sims, training_genes, columns=["train_score"])
    df_cs = df_cs.sort_values(by="train_score", ascending=False)
    adata_map.uns["train_genes_df"] = df_cs

    # Annotate sparsity of each training genes
    annotate_gene_sparsity(adata_sc)
    annotate_gene_sparsity(adata_sp)
    adata_map.uns["train_genes_df"]["sparsity_sc"] = adata_sc[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_sp"] = adata_sp[
        :, training_genes
    ].var.sparsity
    adata_map.uns["train_genes_df"]["sparsity_diff"] = (
        adata_sp[:, training_genes].var.sparsity
        - adata_sc[:, training_genes].var.sparsity
    )

    adata_map.uns["training_history"] = training_history
    adata_map.uns["preprocessing_id"] = adata_sc.uns["preprocessing_id"]

    return adata_map
