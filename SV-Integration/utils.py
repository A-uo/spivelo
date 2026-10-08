# # import pandas as pd
# # from pandas import Series, DataFrame
# # import numpy as np
# # from anndata import AnnData
# # import logging
# # import scanpy as sc
# # from numpy import ndarray
# # from sklearn.metrics import auc
# # from scipy.sparse import issparse
# # from collections import defaultdict


# # def one_hot_encoding(labels: Series, keep_aggregate: bool = False) -> DataFrame:
# #     df_enriched = pd.DataFrame({"cl": labels})
# #     for i in labels.unique():
# #         df_enriched[i] = list(map(int, df_enriched["cl"] == i))
# #     if not keep_aggregate:
# #         del df_enriched["cl"]
# #     return df_enriched


# # def annotate_gene_sparsity(adata: AnnData) -> None:
# #     mask = adata.X != 0
# #     gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
# #     gene_sparsity = np.asarray(gene_sparsity)
# #     gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
# #     adata.var["sparsity"] = gene_sparsity


# # def project_cell_annotations(
# #     adata_map: AnnData, adata_sp: AnnData, annotation: str = "cell_type"
# # ) -> None:
# #     df = one_hot_encoding(adata_map.obs[annotation])
# #     df_ct_prob = adata_map.X.T @ df
# #     df_ct_prob.index = adata_map.var.index

# #     adata_sp.obsm["spnmf_ct_pred"] = df_ct_prob
# #     logging.info(
# #         "spatial prediction dataframe is saved in `obsm` `spnmf_ct_pred` of the spatial AnnData."
# #     )


# # def project_genes(adata_map: AnnData, adata_sc: AnnData) -> AnnData:
# #     assert adata_sc.uns.get("preprocessing_id") == adata_map.uns.get("preprocessing_id")

# #     # put all var index to lower case to align
# #     adata_sc.var_names = [g.lower() for g in adata_sc.var_names]

# #     # make varnames unique for adata_sc
# #     adata_sc.var_names_make_unique()

# #     # remove all-zero-valued genes
# #     sc.pp.filter_genes(adata_sc, min_cells=1)

# #     if not adata_map.obs_names.equals(adata_sc.obs_names):
# #         raise ValueError("The two AnnDatas need to have same `obs` index.")
# #     if hasattr(adata_sc.X, "toarray"):
# #         adata_sc.X = adata_sc.X.toarray()
# #     X_space = adata_map.X.T @ adata_sc.X
# #     adata_ge = sc.AnnData(
# #         X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
# #     )
# #     training_genes = adata_map.uns["train_genes_df"].index.values
# #     adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
# #     adata_ge.uns["preprocessing_id"] = adata_sc.uns["preprocessing_id"]
# #     adata_ge.layers["spliced"] = adata_map.X.T @ adata_sc.layers["spliced"]
# #     adata_ge.layers["unspliced"] = adata_map.X.T @ adata_sc.layers["unspliced"]
# #     # adata_ge.layers["velocity"] = adata_map.X.T @ adata_sc.layers["velocity"]
# #     return adata_ge


# # def compare_spatial_geneexp(
# #     adata_ge: AnnData,
# #     adata_sp: AnnData,
# #     adata_sc: AnnData | None = None,
# #     genes: list[str] | None = None,
# # ) -> DataFrame:
# #     # Check if training_genes/overlap_genes key exist/is valid in adatas.uns
# #     if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
# #         raise ValueError("Missing spnmf parameters. Run `pp_adatas()`.")

# #     if not set(["training_genes", "overlap_genes"]).issubset(set(adata_ge.uns.keys())):
# #         raise ValueError(
# #             "Missing spnmf parameters. Use `project_genes()` to get adata_ge."
# #         )

# #     assert list(adata_sp.uns["overlap_genes"]) == list(adata_ge.uns["overlap_genes"])

# #     if genes is None:
# #         overlap_genes = adata_ge.uns["overlap_genes"]
# #     else:
# #         overlap_genes = genes

# #     annotate_gene_sparsity(adata_sp)

# #     # Annotate cosine similarity of each training gene
# #     cos_sims = []

# #     if hasattr(adata_ge.X, "toarray"):
# #         X_1 = adata_ge[:, overlap_genes].X.toarray()
# #     else:
# #         X_1 = adata_ge[:, overlap_genes].X
# #     if hasattr(adata_sp.X, "toarray"):
# #         X_2 = adata_sp[:, overlap_genes].X.toarray()
# #     else:
# #         X_2 = adata_sp[:, overlap_genes].X

# #     for v1, v2 in zip(X_1.T, X_2.T):
# #         norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
# #         cos_sims.append((v1 @ v2) / norm_sq)

# #     df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
# #     for adata in [adata_ge, adata_sp]:
# #         if "is_training" in adata.var.keys():
# #             df_g["is_training"] = adata.var.is_training

# #     df_g["sparsity_sp"] = adata_sp[:, overlap_genes].var.sparsity

# #     if adata_sc is not None:
# #         if not set(["training_genes", "overlap_genes"]).issubset(
# #             set(adata_sc.uns.keys())
# #         ):
# #             raise ValueError("Missing spnmf parameters. Run `pp_adatas()`.")

# #         assert list(adata_sc.uns["overlap_genes"]) == list(
# #             adata_sp.uns["overlap_genes"]
# #         )
# #         annotate_gene_sparsity(adata_sc)

# #         df_g = df_g.merge(
# #             pd.DataFrame(adata_sc[:, overlap_genes].var["sparsity"]),
# #             left_index=True,
# #             right_index=True,
# #         )
# #         df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
# #         df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

# #     else:
# #         logging.info(
# #             "To create dataframe with column 'sparsity_sc' or 'aprsity_diff', please also pass adata_sc to the function."
# #         )

# #     if genes is not None:
# #         df_g = df_g.loc[genes]

# #     df_g = df_g.sort_values(by="score", ascending=False)
# #     return df_g


# # def eval_metric(
# #     df_all_genes: DataFrame, test_genes: ndarray | list | None = None
# # ) -> tuple[dict, tuple]:
# #     # validate test_genes:
# #     if test_genes is not None:
# #         if not set(test_genes).issubset(set(df_all_genes.index.values)):
# #             raise ValueError(
# #                 "the input of test_genes should be subset of genes of input dataframe"
# #             )
# #         test_genes = np.unique(test_genes)

# #     else:
# #         test_genes = list(set(df_all_genes[~df_all_genes["is_training"]].index.values))

# #     # calculate:
# #     test_gene_scores = df_all_genes.loc[test_genes]["score"]
# #     test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
# #     test_score_avg = test_gene_scores.mean()
# #     train_score_avg = df_all_genes[df_all_genes["is_training"]]["score"].mean()

# #     # sp sparsity weighted score
# #     test_score_sps_sp_g2 = np.sum(
# #         (test_gene_scores * (1 - test_gene_sparsity_sp))
# #         / (1 - test_gene_sparsity_sp).sum()
# #     )

# #     # tm metric
# #     # Fit polynomial'
# #     xs = list(test_gene_scores)
# #     ys = list(test_gene_sparsity_sp)
# #     pol_deg = 2
# #     pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
# #     pol_xs: ndarray | list = np.linspace(0, 1, 10)  # x linearly spaced
# #     pol = np.poly1d(pol_cs)  # build polynomial as function
# #     pol_ys: ndarray | list = [pol(x) for x in pol_xs]  # compute polys

# #     if pol_ys[0] > 1:
# #         pol_ys[0] = 1

# #     # if real root when y = 0, add point (x, 0):
# #     roots = pol.r
# #     root = None
# #     for i in range(len(roots)):
# #         if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
# #             root = roots[i]
# #             break

# #     if root is not None:
# #         pol_xs = np.append(pol_xs, root)
# #         pol_ys = np.append(pol_ys, 0)

# #     np.append(pol_xs, 1)
# #     np.append(pol_ys, pol(1))

# #     # remove point that are out of [0,1]
# #     del_idx = []
# #     for i in range(len(pol_xs)):
# #         if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
# #             del_idx.append(i)

# #     pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
# #     pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

# #     # Compute are under the curve of polynomial
# #     auc_test_score = np.real(auc(pol_xs, pol_ys))

# #     metric_dict = {
# #         "avg_test_score": test_score_avg,
# #         "avg_train_score": train_score_avg,
# #         "sp_sparsity_score": test_score_sps_sp_g2,
# #         "auc_score": auc_test_score,
# #     }

# #     auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

# #     return metric_dict, auc_coordinates


# # def create_segment_cell_df(adata_sp: AnnData) -> None:
# #     if "image_features" not in adata_sp.obsm.keys():
# #         raise ValueError(
# #             "Missing parameter for spnmf deconvolution. Run `sqidpy.im.calculate_image_features`."
# #         )

# #     centroids = adata_sp.obsm["image_features"][["segmentation_centroid"]].copy()
# #     centroids["centroids_idx"] = [
# #         np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
# #         for k, i in zip(
# #             adata_sp.obs.index.values,
# #             adata_sp.obsm["image_features"]["segmentation_label"],
# #         )
# #     ]
# #     centroids_idx = centroids.explode("centroids_idx")
# #     centroids_coords = centroids.explode("segmentation_centroid")
# #     segmentation_df = pd.DataFrame(
# #         centroids_coords["segmentation_centroid"].to_list(),
# #         columns=["y", "x"],
# #         index=centroids_coords.index,
# #     )
# #     segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
# #     segmentation_df.index.set_names("spot_idx", inplace=True)
# #     segmentation_df.reset_index(
# #         drop=False,
# #         inplace=True,
# #     )

# #     adata_sp.uns["spnmf_cell_segmentation"] = segmentation_df
# #     adata_sp.obsm["spnmf_spot_centroids"] = centroids["centroids_idx"].values
# #     logging.info(
# #         "cell segmentation dataframe is saved in `uns` `spnmf_cell_segmentation` of the spatial AnnData."
# #     )
# #     logging.info(
# #         "spot centroids is saved in `obsm` `spnmf_spot_centroids` of the spatial AnnData."
# #     )


# # def allocate_integer_counts(C_float: ndarray, c: ndarray) -> ndarray:
# #     C_int = np.floor(C_float).astype(int)  # 取整数部分
# #     remainders = C_float - C_int  # 小数部分

# #     for i in range(len(c)):
# #         target = c[i]
# #         current_sum = C_int[i].sum()
# #         remaining = target - current_sum

# #         if remaining > 0:
# #             # 获取当前点位的小数部分，按降序排列的索引
# #             decimals = remainders[i]
# #             sorted_indices = np.argsort(-decimals)  # 降序排列

# #             # 选择前remaining个索引（小数部分最大的）
# #             top_indices = sorted_indices[:remaining]
# #             C_int[i, top_indices] += 1

# #     # 验证行和正确性
# #     assert np.allclose(C_int.sum(axis=1), c), "行和与目标值不符！"
# #     return C_int


# # def count_cell_annotations2(
# #     adata_map: AnnData,
# #     adata_sc: AnnData,
# #     adata_sp: AnnData,
# #     annotation: str = "cell_type",
# # ) -> None:
# #     if "spatial" not in adata_sp.obsm.keys():
# #         raise ValueError(
# #             "Missing spatial information in AnnDatas. Please make sure coordinates are saved with AnnData.obsm['spatial']"
# #         )

# #     if "image_features" not in adata_sp.obsm.keys():
# #         raise ValueError(
# #             "Missing parameter for spnmf deconvolution. Run `sqidpy.im.calculate_image_features`."
# #         )

# #     if (
# #         "spnmf_cell_segmentation" not in adata_sp.uns.keys()
# #         or "spnmf_spot_centroids" not in adata_sp.obsm.keys()
# #     ):
# #         raise ValueError(
# #             "Missing parameter for spnmf deconvolution. Run `create_segment_cell_df`."
# #         )

# #     xs = adata_sp.obsm["spatial"][:, 1]
# #     ys = adata_sp.obsm["spatial"][:, 0]
# #     cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

# #     df_segmentation = adata_sp.uns["spnmf_cell_segmentation"]
# #     centroids = adata_sp.obsm["spnmf_spot_centroids"]

# #     # create a dataframe
# #     df_vox_cells = df_vox_cells = pd.DataFrame(
# #         data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
# #         index=list(adata_sp.obs.index),
# #     )

# #     # get the most probable voxel for each cell
# #     resulting_voxels = np.argmax(adata_map.X, axis=1)

# #     # create a list with filtered cells and the voxels where they have been placed with the
# #     # highest probability a cell i is filtered if F_i > threshold'

# #     vox_ct = list(zip(resulting_voxels, adata_sc.obs[annotation]))

# #     df_classes = one_hot_encoding(adata_sc.obs[annotation])
# #     for index, i in enumerate(df_classes.columns):
# #         df_vox_cells[i] = 0

# #     for k, v in vox_ct:
# #         df_vox_cells.iloc[k, df_vox_cells.columns.get_loc(v)] += 1

# #     adata_sp.obsm["spnmf_ct_count"] = df_vox_cells
# #     logging.info(
# #         f"spatial cell count dataframe is saved in `obsm` `spnmf_ct_count` of the spatial AnnData."
# #     )


# # def count_cell_annotations(
# #     adata_map: AnnData,
# #     adata_sc: AnnData,
# #     adata_sp: AnnData,
# #     annotation: str = "cell_type",
# # ) -> None:
# #     # 校验输入完整性
# #     if "spatial" not in adata_sp.obsm:
# #         raise ValueError("Missing spatial coordinates in adata_sp.obsm['spatial']")
# #     if "image_features" not in adata_sp.obsm:
# #         raise ValueError("Run squidpy.im.calculate_image_features first")
# #     if "spnmf_cell_segmentation" not in adata_sp.uns:
# #         raise ValueError("Run create_segment_cell_df first")

# #     xs = adata_sp.obsm["spatial"][:, 1]
# #     ys = adata_sp.obsm["spatial"][:, 0]
# #     cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

# #     # df_segmentation = adata_sp.uns["spnmf_cell_segmentation"]
# #     centroids = adata_sp.obsm["spnmf_spot_centroids"]

# #     # 提取关键数据
# #     M = adata_map.X  # 概率矩阵 (cells × spots)
# #     c = adata_sp.obsm["image_features"]["segmentation_label"].astype(
# #         float
# #     )  # 点位细胞数向量
# #     type = one_hot_encoding(adata_sc.obs[annotation])
# #     T = type.values  # 细胞类型独热编码矩阵

# #     # 概率矩阵缩放
# #     S = np.array(M.sum(axis=0)).flatten()  # 计算各点位总概率
# #     valid_spots = (S > 1e-8) & (c > 0)  # 处理除零情况
# #     scale_factors = np.zeros_like(S)
# #     scale_factors[valid_spots] = c[valid_spots] / S[valid_spots]

# #     # 核心矩阵运算: C = diag(c/S) * M.T * T
# #     M_scaled = M.multiply(scale_factors).T if issparse(M) else (M * scale_factors).T
# #     C = M_scaled @ T

# #     # 转换为整数矩阵
# #     c_values = adata_sp.obsm["image_features"]["segmentation_label"].values.astype(int)
# #     C_int = allocate_integer_counts(C.astype(float), c_values)  # 确保输入为浮点

# #     # 构建输出数据框
# #     df_ct = pd.DataFrame(
# #         C_int,
# #         index=list(adata_sp.obs.index),
# #         columns=type.columns,
# #     )

# #     df_vox_cells = pd.DataFrame(
# #         data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
# #         index=list(adata_sp.obs.index),
# #     )

# #     df_vox_cells = df_vox_cells.join(df_ct)

# #     # 验证行和等于c (处理浮点误差)
# #     # np.testing.assert_allclose(df_ct.sum(axis=1), c, rtol=1e-3)
# #     adata_sp.obsm["spnmf_ct_count"] = df_vox_cells
# #     adata_sp.uns["spnmf_cell_types"] = type.columns


# # def df_to_cell_types(df: DataFrame, cell_types) -> defaultdict:
# #     df_cum_sums = df[cell_types].cumsum(axis=1)

# #     df_c = df.copy()

# #     for i in df_cum_sums.columns:
# #         df_c[i] = df_cum_sums[i]

# #     cell_types_mapped = defaultdict(list)
# #     for i_index, i in enumerate(cell_types):
# #         for j_index, j in df_c.iterrows():
# #             start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
# #             end_ind = j[i]
# #             cell_types_mapped[i].extend(j["centroids"][start_ind:end_ind].tolist())
# #     return cell_types_mapped


# # def deconvolve_cell_annotations2(
# #     adata_sp: AnnData, filter_cell_annotation=None, cluster_label="cluster"
# # ) -> AnnData:
# #     # 1. 检查必要的 spnmf 计算结果是否存在
# #     if (
# #         "spnmf_ct_count" not in adata_sp.obsm
# #         or "spnmf_cell_segmentation" not in adata_sp.uns
# #     ):
# #         raise ValueError(
# #             "Missing spnmf parameters. Run `count_cell_annotations` first."
# #         )

# #     # 2. 取出 segmentation DataFrame（每行对应一个分割出的 voxel）
# #     segmentation_df = adata_sp.uns["spnmf_cell_segmentation"].copy()
# #     #   这里假设 segmentation_df 至少包含列 ['spot_idx', 'centroids', 'x', 'y', ...]
# #     #   其中 'spot_idx' 对应的是 adata_sp.obs.index 中的 spot ID（通常是字符串）。

# #     # 3. 确定要过滤的细胞类型列名
# #     if filter_cell_annotation is None:
# #         filter_cell_annotation = adata_sp.uns["spnmf_cell_types"]
# #     else:
# #         filter_cell_annotation = pd.unique(filter_cell_annotation)

# #     # 4. 构建 “每个 voxel 属于哪个细胞类型” 的映射
# #     #    adata_sp.obsm["spnmf_ct_count"] 是一个 DataFrame：
# #     #    index 顺序与 segmentation_df 一一对应，列为所有细胞类型。
# #     df_vox_cells = adata_sp.obsm["spnmf_ct_count"].copy()

# #     #    假定 df_to_cell_types(df_vox_cells, filter_cell_annotation) 返回一个 dict：
# #     #      { cell_type_name: [centroid1, centroid2, ...], ... }
# #     #    即每个细胞类型对应一个包含该类型所有 voxel centroid 的列表。
# #     cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)

# #     # 5. 将 “centroid ↔ 细胞类型” 映成一个 DataFrame
# #     df_list = []
# #     for cell_type, centroid_list in cell_types_mapped.items():
# #         tmp = pd.DataFrame(
# #             {
# #                 "centroids": np.array(centroid_list, dtype="object"),
# #                 cluster_label: cell_type,
# #             }
# #         )
# #         df_list.append(tmp)
# #     cluster_df = pd.concat(df_list, axis=0).reset_index(drop=True)
# #     # cluster_df 列为 ['centroids', 'cluster']，每行是一个 voxel centroid 对应的细胞类型

# #     # 6. 将 segmentation_df 与 cluster_df 在 'centroids' 列上合并，
# #     #    以获得 voxel 对应的 spot_idx（spot ID）、(x, y) 坐标，以及 cluster（cell type）
# #     merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
# #     #    merged_df 至少含 ['spot_idx', 'centroids', 'x', 'y', ..., 'cluster']
# #     #    去重以防万一
# #     merged_df = merged_df.drop_duplicates(
# #         subset=["centroids", cluster_label, "spot_idx"]
# #     ).copy()

# #     # 7. 对每个 spot_idx（spot ID）分组，取出现次数最多的 'cluster' 作为该 spot 的标注
# #     def mode_per_group(subdf):
# #         # 对该 spot 分组，返回出现频率最高的 cluster
# #         return subdf[cluster_label].value_counts().idxmax()

# #     # spot_to_cluster: Index = spot_idx (spot ID, 与 adata_sp.obs.index 对应)，
# #     # value = 该 spot 的优势细胞类型
# #     spot_to_cluster = (
# #         merged_df.groupby("spot_idx")
# #         .apply(mode_per_group)
# #         .rename(cluster_label)
# #         .astype("category")
# #     )
# #     # spot_to_cluster 的 index dtype 与 adata_sp.obs.index dtype 一致（通常是字符串）

# #     # 8. 构造新的 obs：在原始 adata_sp.obs 的基础上，添加一列 'cluster'
# #     new_obs = adata_sp.obs.copy(deep=True)
# #     # 初始化一列全为 pd.NA 的 category，category 的 categories 取自 spot_to_cluster 中的类别
# #     new_obs[cluster_label] = pd.Categorical(
# #         [pd.NA] * adata_sp.n_obs, categories=spot_to_cluster.cat.categories
# #     )

# #     # 9. 将 spot_to_cluster 中每个 spot_id 对应的 cluster 填到 new_obs
# #     for spot_id, ct in spot_to_cluster.items():
# #         if spot_id in new_obs.index:
# #             new_obs.at[spot_id, cluster_label] = ct

# #     # 10. 生成一个新的 AnnData，保持 X、var、obsm["spatial"]、uns 不变，仅替换 obs
# #     adata_segment = sc.AnnData(
# #         X=adata_sp.X.copy(),  # n_obs × n_vars
# #         obs=new_obs,  # 新的 obs（带 'cluster' 列）
# #         var=adata_sp.var.copy(),  # 保留 var
# #         obsm={"spatial": adata_sp.obsm["spatial"].copy()},
# #         uns=adata_sp.uns.copy(),
# #     )

# #     return adata_segment


# # def deconvolve_cell_annotations(
# #     adata_sp: AnnData, filter_cell_annotation=None, cluster_label="cluster"
# # ) -> AnnData:
# #     if (
# #         "spnmf_ct_count" not in adata_sp.obsm.keys()
# #         or "spnmf_cell_segmentation" not in adata_sp.uns.keys()
# #     ):
# #         raise ValueError("Missing spnmf parameters. Run `count_cell_annotations`.")

# #     segmentation_df = adata_sp.uns["spnmf_cell_segmentation"]

# #     if filter_cell_annotation is None:
# #         filter_cell_annotation = adata_sp.uns["spnmf_cell_types"]
# #     else:
# #         filter_cell_annotation = pd.unique(filter_cell_annotation)

# #     df_vox_cells = adata_sp.obsm["spnmf_ct_count"]
# #     cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)
# #     df_list = []
# #     for k in cell_types_mapped.keys():
# #         df = pd.DataFrame({"centroids": np.array(cell_types_mapped[k], dtype="object")})
# #         df[cluster_label] = k
# #         df_list.append(df)
# #     cluster_df = pd.concat(df_list, axis=0)
# #     cluster_df.reset_index(inplace=True, drop=True)

# #     merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
# #     merged_df.drop(columns="spot_idx", inplace=True)
# #     merged_df.drop_duplicates(inplace=True)
# #     merged_df.dropna(inplace=True)
# #     merged_df.reset_index(inplace=True, drop=True)

# #     adata_segment = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
# #     adata_segment.obsm["spatial"] = merged_df[["y", "x"]].to_numpy()
# #     adata_segment.uns = adata_sp.uns

# #     return adata_segment


# import pandas as pd
# from pandas import Series, DataFrame
# import numpy as np
# from anndata import AnnData
# import logging
# import scanpy as sc
# from numpy import ndarray
# from sklearn.metrics import auc
# from scipy.sparse import issparse
# from collections import defaultdict


# def one_hot_encoding(labels: Series, keep_aggregate: bool = False) -> DataFrame:
#     df_enriched = pd.DataFrame({"cl": labels})
#     for i in labels.unique():
#         df_enriched[i] = list(map(int, df_enriched["cl"] == i))
#     if not keep_aggregate:
#         del df_enriched["cl"]
#     return df_enriched


# def annotate_gene_sparsity(adata: AnnData) -> None:
#     mask = adata.X != 0
#     gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
#     gene_sparsity = np.asarray(gene_sparsity)
#     gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
#     adata.var["sparsity"] = gene_sparsity


# def project_cell_annotations(
#     adata_map: AnnData, adata_sp: AnnData, annotation: str = "cell_type"
# ) -> None:
#     df = one_hot_encoding(adata_map.obs[annotation])
#     df_ct_prob = adata_map.X.T @ df
#     df_ct_prob.index = adata_map.var.index

#     adata_sp.obsm["spnmf_ct_pred"] = df_ct_prob
#     logging.info(
#         "spatial prediction dataframe is saved in `obsm` `spnmf_ct_pred` of the spatial AnnData."
#     )


# def project_genes(adata_map: AnnData, adata_sc: AnnData) -> AnnData:
#     assert adata_sc.uns.get("preprocessing_id") == adata_map.uns.get("preprocessing_id")

#     # put all var index to lower case to align
#     adata_sc.var_names = [g.lower() for g in adata_sc.var_names]

#     # make varnames unique for adata_sc
#     adata_sc.var_names_make_unique()

#     # remove all-zero-valued genes
#     sc.pp.filter_genes(adata_sc, min_cells=1)

#     if not adata_map.obs_names.equals(adata_sc.obs_names):
#         raise ValueError("The two AnnDatas need to have same `obs` index.")
#     if hasattr(adata_sc.X, "toarray"):
#         adata_sc.X = adata_sc.X.toarray()
#     X_space = adata_map.X.T @ adata_sc.X
#     adata_ge = sc.AnnData(
#         X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
#     )
#     training_genes = adata_map.uns["train_genes_df"].index.values
#     adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
#     adata_ge.uns["preprocessing_id"] = adata_sc.uns["preprocessing_id"]
#     adata_ge.layers["spliced"] = adata_map.X.T @ adata_sc.layers["spliced"]
#     adata_ge.layers["unspliced"] = adata_map.X.T @ adata_sc.layers["unspliced"]
#     # adata_ge.layers["velocity"] = adata_map.X.T @ adata_sc.layers["velocity"]
#     return adata_ge


# def compare_spatial_geneexp(
#     adata_ge: AnnData,
#     adata_sp: AnnData,
#     adata_sc: AnnData | None = None,
#     genes: list[str] | None = None,
# ) -> DataFrame:
#     # Check if training_genes/overlap_genes key exist/is valid in adatas.uns
#     if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
#         raise ValueError("Missing spnmf parameters. Run `pp_adatas()`.")

#     if not set(["training_genes", "overlap_genes"]).issubset(set(adata_ge.uns.keys())):
#         raise ValueError(
#             "Missing spnmf parameters. Use `project_genes()` to get adata_ge."
#         )

#     assert list(adata_sp.uns["overlap_genes"]) == list(adata_ge.uns["overlap_genes"])

#     if genes is None:
#         overlap_genes = adata_ge.uns["overlap_genes"]
#     else:
#         overlap_genes = genes

#     annotate_gene_sparsity(adata_sp)

#     # Annotate cosine similarity of each training gene
#     cos_sims = []

#     if hasattr(adata_ge.X, "toarray"):
#         X_1 = adata_ge[:, overlap_genes].X.toarray()
#     else:
#         X_1 = adata_ge[:, overlap_genes].X
#     if hasattr(adata_sp.X, "toarray"):
#         X_2 = adata_sp[:, overlap_genes].X.toarray()
#     else:
#         X_2 = adata_sp[:, overlap_genes].X

#     for v1, v2 in zip(X_1.T, X_2.T):
#         norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
#         cos_sims.append((v1 @ v2) / norm_sq)

#     df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
#     for adata in [adata_ge, adata_sp]:
#         if "is_training" in adata.var.keys():
#             df_g["is_training"] = adata.var.is_training

#     df_g["sparsity_sp"] = adata_sp[:, overlap_genes].var.sparsity

#     if adata_sc is not None:
#         if not set(["training_genes", "overlap_genes"]).issubset(
#             set(adata_sc.uns.keys())
#         ):
#             raise ValueError("Missing spnmf parameters. Run `pp_adatas()`.")

#         assert list(adata_sc.uns["overlap_genes"]) == list(
#             adata_sp.uns["overlap_genes"]
#         )
#         annotate_gene_sparsity(adata_sc)

#         df_g = df_g.merge(
#             pd.DataFrame(adata_sc[:, overlap_genes].var["sparsity"]),
#             left_index=True,
#             right_index=True,
#         )
#         df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
#         df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

#     else:
#         logging.info(
#             "To create dataframe with column 'sparsity_sc' or 'aprsity_diff', please also pass adata_sc to the function."
#         )

#     if genes is not None:
#         df_g = df_g.loc[genes]

#     df_g = df_g.sort_values(by="score", ascending=False)
#     return df_g


# def eval_metric(
#     df_all_genes: DataFrame, test_genes: ndarray | list | None = None
# ) -> tuple[dict, tuple]:
#     # validate test_genes:
#     if test_genes is not None:
#         if not set(test_genes).issubset(set(df_all_genes.index.values)):
#             raise ValueError(
#                 "the input of test_genes should be subset of genes of input dataframe"
#             )
#         test_genes = np.unique(test_genes)

#     else:
#         test_genes = list(set(df_all_genes[~df_all_genes["is_training"]].index.values))

#     # calculate:
#     test_gene_scores = df_all_genes.loc[test_genes]["score"]
#     test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
#     test_score_avg = test_gene_scores.mean()
#     train_score_avg = df_all_genes[df_all_genes["is_training"]]["score"].mean()

#     # sp sparsity weighted score
#     test_score_sps_sp_g2 = np.sum(
#         (test_gene_scores * (1 - test_gene_sparsity_sp))
#         / (1 - test_gene_sparsity_sp).sum()
#     )

#     # tm metric
#     # Fit polynomial'
#     xs = list(test_gene_scores)
#     ys = list(test_gene_sparsity_sp)
#     pol_deg = 2
#     pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
#     pol_xs: ndarray | list = np.linspace(0, 1, 10)  # x linearly spaced
#     pol = np.poly1d(pol_cs)  # build polynomial as function
#     pol_ys: ndarray | list = [pol(x) for x in pol_xs]  # compute polys

#     if pol_ys[0] > 1:
#         pol_ys[0] = 1

#     # if real root when y = 0, add point (x, 0):
#     roots = pol.r
#     root = None
#     for i in range(len(roots)):
#         if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
#             root = roots[i]
#             break

#     if root is not None:
#         pol_xs = np.append(pol_xs, root)
#         pol_ys = np.append(pol_ys, 0)

#     np.append(pol_xs, 1)
#     np.append(pol_ys, pol(1))

#     # remove point that are out of [0,1]
#     del_idx = []
#     for i in range(len(pol_xs)):
#         if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
#             del_idx.append(i)

#     pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
#     pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

#     # Compute are under the curve of polynomial
#     auc_test_score = np.real(auc(pol_xs, pol_ys))

#     metric_dict = {
#         "avg_test_score": test_score_avg,
#         "avg_train_score": train_score_avg,
#         "sp_sparsity_score": test_score_sps_sp_g2,
#         "auc_score": auc_test_score,
#     }

#     auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

#     return metric_dict, auc_coordinates


# def create_segment_cell_df(adata_sp: AnnData) -> None:
#     if "image_features" not in adata_sp.obsm.keys():
#         raise ValueError(
#             "Missing parameter for spnmf deconvolution. Run `sqidpy.im.calculate_image_features`."
#         )

#     centroids = adata_sp.obsm["image_features"][["segmentation_centroid"]].copy()
#     centroids["centroids_idx"] = [
#         np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
#         for k, i in zip(
#             adata_sp.obs.index.values,
#             adata_sp.obsm["image_features"]["segmentation_label"],
#         )
#     ]
#     centroids_idx = centroids.explode("centroids_idx")
#     centroids_coords = centroids.explode("segmentation_centroid")
#     segmentation_df = pd.DataFrame(
#         centroids_coords["segmentation_centroid"].to_list(),
#         columns=["y", "x"],
#         index=centroids_coords.index,
#     )
#     segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
#     segmentation_df.index.set_names("spot_idx", inplace=True)
#     segmentation_df.reset_index(
#         drop=False,
#         inplace=True,
#     )

#     adata_sp.uns["spnmf_cell_segmentation"] = segmentation_df
#     adata_sp.obsm["spnmf_spot_centroids"] = centroids["centroids_idx"].values
#     logging.info(
#         "cell segmentation dataframe is saved in `uns` `spnmf_cell_segmentation` of the spatial AnnData."
#     )
#     logging.info(
#         "spot centroids is saved in `obsm` `spnmf_spot_centroids` of the spatial AnnData."
#     )


# def infer_spot_cell_types_no_image(
#     adata_sp: AnnData,
#     ct_pred_key: str = "spnmf_ct_pred",
#     output_key: str = "ct_pred",
#     filtered_key: str = "ct_pred_filtered",
#     smooth_key: str = "ct_pred_smooth",
#     confidence_key: str = "ct_confidence",
#     margin_key: str = "ct_margin",
#     entropy_key: str = "ct_entropy",
#     unknown_label: str = "Unknown",
#     confidence_min: float = 0.45,
#     margin_min: float = 0.15,
#     entropy_max: float = 0.70,
#     smooth_alpha: float = 0.60,
#     spatial_adj_key: str = "spatial_connectivities",
#     run_smoothing: bool = True,
# ) -> dict:
#     """Infer spot-level cell types from composition matrix without image.

#     This is intended for spot-level spatial transcriptomics when image segmentation
#     is unavailable. It provides:
#     1) hard label via argmax
#     2) confidence/margin/entropy diagnostics
#     3) optional Unknown filtering
#     4) optional adjacency smoothing

#     Required input:
#     - `adata_sp.obsm[ct_pred_key]`: shape [n_spots, n_cell_types]
#       (e.g. produced by `project_cell_annotations`).
#     """
#     if ct_pred_key not in adata_sp.obsm:
#         raise KeyError(f"`adata_sp.obsm['{ct_pred_key}']` not found.")

#     ct_pred = adata_sp.obsm[ct_pred_key]
#     if isinstance(ct_pred, pd.DataFrame):
#         ct_df = ct_pred.copy()
#     else:
#         ct_pred = np.asarray(ct_pred)
#         if ct_pred.ndim != 2:
#             raise ValueError(f"`adata_sp.obsm['{ct_pred_key}']` must be rank-2, got shape={ct_pred.shape}.")
#         colnames = [f"ct_{i}" for i in range(ct_pred.shape[1])]
#         ct_df = pd.DataFrame(ct_pred, index=adata_sp.obs_names, columns=colnames)

#     P = ct_df.values.astype(float)
#     P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-8, None)
#     celltypes = np.asarray(ct_df.columns)

#     top1_idx = P.argmax(axis=1)
#     top1_prob = P[np.arange(P.shape[0]), top1_idx]
#     sorted_probs = np.sort(P, axis=1)
#     top2_prob = sorted_probs[:, -2] if P.shape[1] > 1 else np.zeros(P.shape[0], dtype=P.dtype)
#     margin = top1_prob - top2_prob
#     entropy = -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(axis=1) / np.log(max(P.shape[1], 2))

#     adata_sp.obs[output_key] = pd.Categorical(celltypes[top1_idx], categories=list(celltypes))
#     adata_sp.obs[confidence_key] = top1_prob
#     adata_sp.obs[margin_key] = margin
#     adata_sp.obs[entropy_key] = entropy

#     unknown_mask = (
#         (adata_sp.obs[confidence_key].values < confidence_min)
#         | (adata_sp.obs[margin_key].values < margin_min)
#         | (adata_sp.obs[entropy_key].values > entropy_max)
#     )
#     filtered = adata_sp.obs[output_key].astype(str).values
#     filtered[unknown_mask] = unknown_label
#     adata_sp.obs[filtered_key] = pd.Categorical(filtered)

#     result = {
#         "proportions": P,
#         "celltypes": celltypes,
#         "unknown_fraction": float(np.mean(unknown_mask)),
#     }

#     if run_smoothing:
#         if spatial_adj_key not in adata_sp.obsp:
#             logging.warning(
#                 f"Skip smoothing: `adata_sp.obsp['{spatial_adj_key}']` not found."
#             )
#         else:
#             A = adata_sp.obsp[spatial_adj_key]
#             A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
#             A = np.asarray(A, dtype=float)
#             if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != adata_sp.n_obs:
#                 raise ValueError(
#                     f"`adata_sp.obsp['{spatial_adj_key}']` must be square [n_obs, n_obs], got {A.shape}."
#                 )
#             A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-8, None)
#             alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
#             P_smooth = alpha * (A @ P) + (1.0 - alpha) * P
#             P_smooth = P_smooth / np.clip(P_smooth.sum(axis=1, keepdims=True), 1e-8, None)
#             idx = P_smooth.argmax(axis=1)
#             adata_sp.obs[smooth_key] = pd.Categorical(celltypes[idx], categories=list(celltypes))
#             adata_sp.obsm[f"{ct_pred_key}_smooth"] = pd.DataFrame(
#                 P_smooth, index=adata_sp.obs_names, columns=ct_df.columns
#             )
#             result["proportions_smooth"] = P_smooth

#     return result


# def allocate_integer_counts(C_float: ndarray, c: ndarray) -> ndarray:
#     C_int = np.floor(C_float).astype(int)  # 取整数部分
#     remainders = C_float - C_int  # 小数部分

#     for i in range(len(c)):
#         target = c[i]
#         current_sum = C_int[i].sum()
#         remaining = target - current_sum

#         if remaining > 0:
#             # 获取当前点位的小数部分，按降序排列的索引
#             decimals = remainders[i]
#             sorted_indices = np.argsort(-decimals)  # 降序排列

#             # 选择前remaining个索引（小数部分最大的）
#             top_indices = sorted_indices[:remaining]
#             C_int[i, top_indices] += 1

#     # 验证行和正确性
#     assert np.allclose(C_int.sum(axis=1), c), "行和与目标值不符！"
#     return C_int


# def count_cell_annotations2(
#     adata_map: AnnData,
#     adata_sc: AnnData,
#     adata_sp: AnnData,
#     annotation: str = "cell_type",
# ) -> None:
#     if "spatial" not in adata_sp.obsm.keys():
#         raise ValueError(
#             "Missing spatial information in AnnDatas. Please make sure coordinates are saved with AnnData.obsm['spatial']"
#         )

#     if "image_features" not in adata_sp.obsm.keys():
#         raise ValueError(
#             "Missing parameter for spnmf deconvolution. Run `sqidpy.im.calculate_image_features`."
#         )

#     if (
#         "spnmf_cell_segmentation" not in adata_sp.uns.keys()
#         or "spnmf_spot_centroids" not in adata_sp.obsm.keys()
#     ):
#         raise ValueError(
#             "Missing parameter for spnmf deconvolution. Run `create_segment_cell_df`."
#         )

#     xs = adata_sp.obsm["spatial"][:, 1]
#     ys = adata_sp.obsm["spatial"][:, 0]
#     cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

#     df_segmentation = adata_sp.uns["spnmf_cell_segmentation"]
#     centroids = adata_sp.obsm["spnmf_spot_centroids"]

#     # create a dataframe
#     df_vox_cells = df_vox_cells = pd.DataFrame(
#         data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
#         index=list(adata_sp.obs.index),
#     )

#     # get the most probable voxel for each cell
#     resulting_voxels = np.argmax(adata_map.X, axis=1)

#     # create a list with filtered cells and the voxels where they have been placed with the
#     # highest probability a cell i is filtered if F_i > threshold'

#     vox_ct = list(zip(resulting_voxels, adata_sc.obs[annotation]))

#     df_classes = one_hot_encoding(adata_sc.obs[annotation])
#     for index, i in enumerate(df_classes.columns):
#         df_vox_cells[i] = 0

#     for k, v in vox_ct:
#         df_vox_cells.iloc[k, df_vox_cells.columns.get_loc(v)] += 1

#     adata_sp.obsm["spnmf_ct_count"] = df_vox_cells
#     logging.info(
#         f"spatial cell count dataframe is saved in `obsm` `spnmf_ct_count` of the spatial AnnData."
#     )


# def count_cell_annotations(
#     adata_map: AnnData,
#     adata_sc: AnnData,
#     adata_sp: AnnData,
#     annotation: str = "cell_type",
# ) -> None:
#     # 校验输入完整性
#     if "spatial" not in adata_sp.obsm:
#         raise ValueError("Missing spatial coordinates in adata_sp.obsm['spatial']")
#     if "image_features" not in adata_sp.obsm:
#         raise ValueError("Run squidpy.im.calculate_image_features first")
#     if "spnmf_cell_segmentation" not in adata_sp.uns:
#         raise ValueError("Run create_segment_cell_df first")

#     xs = adata_sp.obsm["spatial"][:, 1]
#     ys = adata_sp.obsm["spatial"][:, 0]
#     cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

#     # df_segmentation = adata_sp.uns["spnmf_cell_segmentation"]
#     centroids = adata_sp.obsm["spnmf_spot_centroids"]

#     # 提取关键数据
#     M = adata_map.X  # 概率矩阵 (cells × spots)
#     c = adata_sp.obsm["image_features"]["segmentation_label"].astype(
#         float
#     )  # 点位细胞数向量
#     type = one_hot_encoding(adata_sc.obs[annotation])
#     T = type.values  # 细胞类型独热编码矩阵

#     # 概率矩阵缩放
#     S = np.array(M.sum(axis=0)).flatten()  # 计算各点位总概率
#     valid_spots = (S > 1e-8) & (c > 0)  # 处理除零情况
#     scale_factors = np.zeros_like(S)
#     scale_factors[valid_spots] = c[valid_spots] / S[valid_spots]

#     # 核心矩阵运算: C = diag(c/S) * M.T * T
#     M_scaled = M.multiply(scale_factors).T if issparse(M) else (M * scale_factors).T
#     C = M_scaled @ T

#     # 转换为整数矩阵
#     c_values = adata_sp.obsm["image_features"]["segmentation_label"].values.astype(int)
#     C_int = allocate_integer_counts(C.astype(float), c_values)  # 确保输入为浮点

#     # 构建输出数据框
#     df_ct = pd.DataFrame(
#         C_int,
#         index=list(adata_sp.obs.index),
#         columns=type.columns,
#     )

#     df_vox_cells = pd.DataFrame(
#         data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
#         index=list(adata_sp.obs.index),
#     )

#     df_vox_cells = df_vox_cells.join(df_ct)

#     # 验证行和等于c (处理浮点误差)
#     # np.testing.assert_allclose(df_ct.sum(axis=1), c, rtol=1e-3)
#     adata_sp.obsm["spnmf_ct_count"] = df_vox_cells
#     adata_sp.uns["spnmf_cell_types"] = type.columns


# def df_to_cell_types(df: DataFrame, cell_types) -> defaultdict:
#     df_cum_sums = df[cell_types].cumsum(axis=1)

#     df_c = df.copy()

#     for i in df_cum_sums.columns:
#         df_c[i] = df_cum_sums[i]

#     cell_types_mapped = defaultdict(list)
#     for i_index, i in enumerate(cell_types):
#         for j_index, j in df_c.iterrows():
#             start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
#             end_ind = j[i]
#             cell_types_mapped[i].extend(j["centroids"][start_ind:end_ind].tolist())
#     return cell_types_mapped


# def deconvolve_cell_annotations2(
#     adata_sp: AnnData, filter_cell_annotation=None, cluster_label="cluster"
# ) -> AnnData:
#     # 1. 检查必要的 spnmf 计算结果是否存在
#     if (
#         "spnmf_ct_count" not in adata_sp.obsm
#         or "spnmf_cell_segmentation" not in adata_sp.uns
#     ):
#         raise ValueError(
#             "Missing spnmf parameters. Run `count_cell_annotations` first."
#         )

#     # 2. 取出 segmentation DataFrame（每行对应一个分割出的 voxel）
#     segmentation_df = adata_sp.uns["spnmf_cell_segmentation"].copy()
#     #   这里假设 segmentation_df 至少包含列 ['spot_idx', 'centroids', 'x', 'y', ...]
#     #   其中 'spot_idx' 对应的是 adata_sp.obs.index 中的 spot ID（通常是字符串）。

#     # 3. 确定要过滤的细胞类型列名
#     if filter_cell_annotation is None:
#         filter_cell_annotation = adata_sp.uns["spnmf_cell_types"]
#     else:
#         filter_cell_annotation = pd.unique(filter_cell_annotation)

#     # 4. 构建 “每个 voxel 属于哪个细胞类型” 的映射
#     #    adata_sp.obsm["spnmf_ct_count"] 是一个 DataFrame：
#     #    index 顺序与 segmentation_df 一一对应，列为所有细胞类型。
#     df_vox_cells = adata_sp.obsm["spnmf_ct_count"].copy()

#     #    假定 df_to_cell_types(df_vox_cells, filter_cell_annotation) 返回一个 dict：
#     #      { cell_type_name: [centroid1, centroid2, ...], ... }
#     #    即每个细胞类型对应一个包含该类型所有 voxel centroid 的列表。
#     cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)

#     # 5. 将 “centroid ↔ 细胞类型” 映成一个 DataFrame
#     df_list = []
#     for cell_type, centroid_list in cell_types_mapped.items():
#         tmp = pd.DataFrame(
#             {
#                 "centroids": np.array(centroid_list, dtype="object"),
#                 cluster_label: cell_type,
#             }
#         )
#         df_list.append(tmp)
#     cluster_df = pd.concat(df_list, axis=0).reset_index(drop=True)
#     # cluster_df 列为 ['centroids', 'cluster']，每行是一个 voxel centroid 对应的细胞类型

#     # 6. 将 segmentation_df 与 cluster_df 在 'centroids' 列上合并，
#     #    以获得 voxel 对应的 spot_idx（spot ID）、(x, y) 坐标，以及 cluster（cell type）
#     merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
#     #    merged_df 至少含 ['spot_idx', 'centroids', 'x', 'y', ..., 'cluster']
#     #    去重以防万一
#     merged_df = merged_df.drop_duplicates(
#         subset=["centroids", cluster_label, "spot_idx"]
#     ).copy()

#     # 7. 对每个 spot_idx（spot ID）分组，取出现次数最多的 'cluster' 作为该 spot 的标注
#     def mode_per_group(subdf):
#         # 对该 spot 分组，返回出现频率最高的 cluster
#         return subdf[cluster_label].value_counts().idxmax()

#     # spot_to_cluster: Index = spot_idx (spot ID, 与 adata_sp.obs.index 对应)，
#     # value = 该 spot 的优势细胞类型
#     spot_to_cluster = (
#         merged_df.groupby("spot_idx")
#         .apply(mode_per_group)
#         .rename(cluster_label)
#         .astype("category")
#     )
#     # spot_to_cluster 的 index dtype 与 adata_sp.obs.index dtype 一致（通常是字符串）

#     # 8. 构造新的 obs：在原始 adata_sp.obs 的基础上，添加一列 'cluster'
#     new_obs = adata_sp.obs.copy(deep=True)
#     # 初始化一列全为 pd.NA 的 category，category 的 categories 取自 spot_to_cluster 中的类别
#     new_obs[cluster_label] = pd.Categorical(
#         [pd.NA] * adata_sp.n_obs, categories=spot_to_cluster.cat.categories
#     )

#     # 9. 将 spot_to_cluster 中每个 spot_id 对应的 cluster 填到 new_obs
#     for spot_id, ct in spot_to_cluster.items():
#         if spot_id in new_obs.index:
#             new_obs.at[spot_id, cluster_label] = ct

#     # 10. 生成一个新的 AnnData，保持 X、var、obsm["spatial"]、uns 不变，仅替换 obs
#     adata_segment = sc.AnnData(
#         X=adata_sp.X.copy(),  # n_obs × n_vars
#         obs=new_obs,  # 新的 obs（带 'cluster' 列）
#         var=adata_sp.var.copy(),  # 保留 var
#         obsm={"spatial": adata_sp.obsm["spatial"].copy()},
#         uns=adata_sp.uns.copy(),
#     )

#     return adata_segment


# def deconvolve_cell_annotations(
#     adata_sp: AnnData, filter_cell_annotation=None, cluster_label="cluster"
# ) -> AnnData:
#     if (
#         "spnmf_ct_count" not in adata_sp.obsm.keys()
#         or "spnmf_cell_segmentation" not in adata_sp.uns.keys()
#     ):
#         raise ValueError("Missing spnmf parameters. Run `count_cell_annotations`.")

#     segmentation_df = adata_sp.uns["spnmf_cell_segmentation"]

#     if filter_cell_annotation is None:
#         filter_cell_annotation = adata_sp.uns["spnmf_cell_types"]
#     else:
#         filter_cell_annotation = pd.unique(filter_cell_annotation)

#     df_vox_cells = adata_sp.obsm["spnmf_ct_count"]
#     cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)
#     df_list = []
#     for k in cell_types_mapped.keys():
#         df = pd.DataFrame({"centroids": np.array(cell_types_mapped[k], dtype="object")})
#         df[cluster_label] = k
#         df_list.append(df)
#     cluster_df = pd.concat(df_list, axis=0)
#     cluster_df.reset_index(inplace=True, drop=True)

#     merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
#     merged_df.drop(columns="spot_idx", inplace=True)
#     merged_df.drop_duplicates(inplace=True)
#     merged_df.dropna(inplace=True)
#     merged_df.reset_index(inplace=True, drop=True)

#     adata_segment = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
#     adata_segment.obsm["spatial"] = merged_df[["y", "x"]].to_numpy()
#     adata_segment.uns = adata_sp.uns

#     return adata_segment
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from anndata import AnnData
import logging
import scanpy as sc
from numpy import ndarray
from sklearn.metrics import auc
from scipy.sparse import issparse
from collections import defaultdict


def one_hot_encoding(labels: Series, keep_aggregate: bool = False) -> DataFrame:
    df_enriched = pd.DataFrame({"cl": labels})
    for i in labels.unique():
        df_enriched[i] = list(map(int, df_enriched["cl"] == i))
    if not keep_aggregate:
        del df_enriched["cl"]
    return df_enriched


def annotate_gene_sparsity(adata: AnnData) -> None:
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
    adata.var["sparsity"] = gene_sparsity


def project_cell_annotations(
    adata_map: AnnData, adata_sp: AnnData, annotation: str = "cell_type"
) -> None:
    df = one_hot_encoding(adata_map.obs[annotation])
    df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index

    adata_sp.obsm["spnmf_ct_pred"] = df_ct_prob
    logging.info(
        "spatial prediction dataframe is saved in `obsm` `spnmf_ct_pred` of the spatial AnnData."
    )


def project_genes(adata_map: AnnData, adata_sc: AnnData) -> AnnData:
    assert adata_sc.uns.get("preprocessing_id") == adata_map.uns.get("preprocessing_id")

    # put all var index to lower case to align
    adata_sc.var_names = [g.lower() for g in adata_sc.var_names]

    # make varnames unique for adata_sc
    adata_sc.var_names_make_unique()

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)

    if not adata_map.obs_names.equals(adata_sc.obs_names):
        raise ValueError("The two AnnDatas need to have same `obs` index.")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = adata_sc.X.toarray()
    X_space = adata_map.X.T @ adata_sc.X
    adata_ge = sc.AnnData(
        X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
    )
    training_genes = adata_map.uns["train_genes_df"].index.values
    adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    adata_ge.uns["preprocessing_id"] = adata_sc.uns["preprocessing_id"]
    adata_ge.layers["spliced"] = adata_map.X.T @ adata_sc.layers["spliced"]
    adata_ge.layers["unspliced"] = adata_map.X.T @ adata_sc.layers["unspliced"]
    return adata_ge


def compare_spatial_geneexp(
    adata_ge: AnnData,
    adata_sp: AnnData,
    adata_sc: AnnData | None = None,
    genes: list[str] | None = None,
) -> DataFrame:
    # Check if training_genes/overlap_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing spnmf parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_ge.uns.keys())):
        raise ValueError(
            "Missing spnmf parameters. Use `project_genes()` to get adata_ge."
        )

    assert list(adata_sp.uns["overlap_genes"]) == list(adata_ge.uns["overlap_genes"])

    if genes is None:
        overlap_genes = adata_ge.uns["overlap_genes"]
    else:
        overlap_genes = genes

    annotate_gene_sparsity(adata_sp)

    # Annotate cosine similarity of each training gene
    cos_sims = []

    if hasattr(adata_ge.X, "toarray"):
        X_1 = adata_ge[:, overlap_genes].X.toarray()
    else:
        X_1 = adata_ge[:, overlap_genes].X
    if hasattr(adata_sp.X, "toarray"):
        X_2 = adata_sp[:, overlap_genes].X.toarray()
    else:
        X_2 = adata_sp[:, overlap_genes].X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
    for adata in [adata_ge, adata_sp]:
        if "is_training" in adata.var.keys():
            df_g["is_training"] = adata.var.is_training

    df_g["sparsity_sp"] = adata_sp[:, overlap_genes].var.sparsity

    if adata_sc is not None:
        if not set(["training_genes", "overlap_genes"]).issubset(
            set(adata_sc.uns.keys())
        ):
            raise ValueError("Missing spnmf parameters. Run `pp_adatas()`.")

        assert list(adata_sc.uns["overlap_genes"]) == list(
            adata_sp.uns["overlap_genes"]
        )
        annotate_gene_sparsity(adata_sc)

        df_g = df_g.merge(
            pd.DataFrame(adata_sc[:, overlap_genes].var["sparsity"]),
            left_index=True,
            right_index=True,
        )
        df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
        df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

    else:
        logging.info(
            "To create dataframe with column 'sparsity_sc' or 'aprsity_diff', please also pass adata_sc to the function."
        )

    if genes is not None:
        df_g = df_g.loc[genes]

    df_g = df_g.sort_values(by="score", ascending=False)
    return df_g


def eval_metric(
    df_all_genes: DataFrame, test_genes: ndarray | list | None = None
) -> tuple[dict, tuple]:
    # validate test_genes:
    if test_genes is not None:
        if not set(test_genes).issubset(set(df_all_genes.index.values)):
            raise ValueError(
                "the input of test_genes should be subset of genes of input dataframe"
            )
        test_genes = np.unique(test_genes)

    else:
        test_genes = list(set(df_all_genes[~df_all_genes["is_training"]].index.values))

    # calculate:
    test_gene_scores = df_all_genes.loc[test_genes]["score"]
    test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
    test_score_avg = test_gene_scores.mean()
    train_score_avg = df_all_genes[df_all_genes["is_training"]]["score"].mean()

    # sp sparsity weighted score
    test_score_sps_sp_g2 = np.sum(
        (test_gene_scores * (1 - test_gene_sparsity_sp))
        / (1 - test_gene_sparsity_sp).sum()
    )

    # tm metric
    # Fit polynomial'
    xs = list(test_gene_scores)
    ys = list(test_gene_sparsity_sp)
    pol_deg = 2
    pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
    pol_xs: ndarray | list = np.linspace(0, 1, 10)  # x linearly spaced
    pol = np.poly1d(pol_cs)  # build polynomial as function
    pol_ys: ndarray | list = [pol(x) for x in pol_xs]  # compute polys

    if pol_ys[0] > 1:
        pol_ys[0] = 1

    # if real root when y = 0, add point (x, 0):
    roots = pol.r
    root = None
    for i in range(len(roots)):
        if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
            root = roots[i]
            break

    if root is not None:
        pol_xs = np.append(pol_xs, root)
        pol_ys = np.append(pol_ys, 0)

    np.append(pol_xs, 1)
    np.append(pol_ys, pol(1))

    # remove point that are out of [0,1]
    del_idx = []
    for i in range(len(pol_xs)):
        if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
            del_idx.append(i)

    pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
    pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

    # Compute are under the curve of polynomial
    auc_test_score = np.real(auc(pol_xs, pol_ys))

    metric_dict = {
        "avg_test_score": test_score_avg,
        "avg_train_score": train_score_avg,
        "sp_sparsity_score": test_score_sps_sp_g2,
        "auc_score": auc_test_score,
    }

    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

    return metric_dict, auc_coordinates


def create_segment_cell_df(adata_sp: AnnData) -> None:
    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for spnmf deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    centroids = adata_sp.obsm["image_features"][["segmentation_centroid"]].copy()
    centroids["centroids_idx"] = [
        np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
        for k, i in zip(
            adata_sp.obs.index.values,
            adata_sp.obsm["image_features"]["segmentation_label"],
        )
    ]
    centroids_idx = centroids.explode("centroids_idx")
    centroids_coords = centroids.explode("segmentation_centroid")
    segmentation_df = pd.DataFrame(
        centroids_coords["segmentation_centroid"].to_list(),
        columns=["y", "x"],
        index=centroids_coords.index,
    )
    segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
    segmentation_df.index.set_names("spot_idx", inplace=True)
    segmentation_df.reset_index(
        drop=False,
        inplace=True,
    )

    adata_sp.uns["spnmf_cell_segmentation"] = segmentation_df
    adata_sp.obsm["spnmf_spot_centroids"] = centroids["centroids_idx"].values
    logging.info(
        "cell segmentation dataframe is saved in `uns` `spnmf_cell_segmentation` of the spatial AnnData."
    )
    logging.info(
        "spot centroids is saved in `obsm` `spnmf_spot_centroids` of the spatial AnnData."
    )


def infer_spot_cell_types_no_image(
    adata_sp: AnnData,
    ct_pred_key: str = "spnmf_ct_pred",
    output_prob_key: str = "cell_type_composition",
    output_key: str = "ct_pred",
    filtered_key: str = "ct_pred_filtered",
    smooth_key: str = "ct_pred_smooth",
    confidence_key: str = "ct_confidence",
    margin_key: str = "ct_margin",
    entropy_key: str = "ct_entropy",
    unknown_label: str = "Unknown",
    confidence_min: float = 0.45,
    margin_min: float = 0.15,
    entropy_max: float = 0.70,
    smooth_alpha: float = 0.60,
    spatial_adj_key: str = "spatial_connectivities",
    run_smoothing: bool = True,
    assign_hard_labels: bool = True,
) -> dict:
    """Infer spot-level cell types from composition matrix without image.

    This is intended for spot-level spatial transcriptomics when image segmentation
    is unavailable. It provides:
    1) hard label via argmax
    2) confidence/margin/entropy diagnostics
    3) optional Unknown filtering
    4) optional adjacency smoothing

    Required input:
    - `adata_sp.obsm[ct_pred_key]`: shape [n_spots, n_cell_types]
      (e.g. produced by `project_cell_annotations`).
    """
    if ct_pred_key not in adata_sp.obsm:
        raise KeyError(f"`adata_sp.obsm['{ct_pred_key}']` not found.")

    ct_pred = adata_sp.obsm[ct_pred_key]
    if isinstance(ct_pred, pd.DataFrame):
        ct_df = ct_pred.copy()
    else:
        ct_pred = np.asarray(ct_pred)
        if ct_pred.ndim != 2:
            raise ValueError(f"`adata_sp.obsm['{ct_pred_key}']` must be rank-2, got shape={ct_pred.shape}.")
        colnames = [f"ct_{i}" for i in range(ct_pred.shape[1])]
        ct_df = pd.DataFrame(ct_pred, index=adata_sp.obs_names, columns=colnames)

    P = ct_df.values.astype(float)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-8, None)
    celltypes = np.asarray(ct_df.columns)

    # Always keep full composition probabilities (soft annotation).
    prob_df = pd.DataFrame(P, index=adata_sp.obs_names, columns=ct_df.columns)
    adata_sp.obsm[output_prob_key] = prob_df

    top1_idx = P.argmax(axis=1)
    top1_prob = P[np.arange(P.shape[0]), top1_idx]
    sorted_probs = np.sort(P, axis=1)
    top2_prob = sorted_probs[:, -2] if P.shape[1] > 1 else np.zeros(P.shape[0], dtype=P.dtype)
    margin = top1_prob - top2_prob
    entropy = -(P * np.log(np.clip(P, 1e-12, 1.0))).sum(axis=1) / np.log(max(P.shape[1], 2))

    adata_sp.obs[confidence_key] = top1_prob
    adata_sp.obs[margin_key] = margin
    adata_sp.obs[entropy_key] = entropy

    unknown_mask = (
        (adata_sp.obs[confidence_key].values < confidence_min)
        | (adata_sp.obs[margin_key].values < margin_min)
        | (adata_sp.obs[entropy_key].values > entropy_max)
    )
    if assign_hard_labels:
        adata_sp.obs[output_key] = pd.Categorical(celltypes[top1_idx], categories=list(celltypes))
        filtered = adata_sp.obs[output_key].astype(str).values
        filtered[unknown_mask] = unknown_label
        adata_sp.obs[filtered_key] = pd.Categorical(filtered)

    result = {
        "proportions": P,
        "celltypes": celltypes,
        "unknown_fraction": float(np.mean(unknown_mask)),
    }

    if run_smoothing:
        if spatial_adj_key not in adata_sp.obsp:
            logging.warning(
                f"Skip smoothing: `adata_sp.obsp['{spatial_adj_key}']` not found."
            )
        else:
            A = adata_sp.obsp[spatial_adj_key]
            A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
            A = np.asarray(A, dtype=float)
            if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != adata_sp.n_obs:
                raise ValueError(
                    f"`adata_sp.obsp['{spatial_adj_key}']` must be square [n_obs, n_obs], got {A.shape}."
                )
            A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-8, None)
            alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
            P_smooth = alpha * (A @ P) + (1.0 - alpha) * P
            P_smooth = P_smooth / np.clip(P_smooth.sum(axis=1, keepdims=True), 1e-8, None)
            idx = P_smooth.argmax(axis=1)
            if assign_hard_labels:
                adata_sp.obs[smooth_key] = pd.Categorical(celltypes[idx], categories=list(celltypes))
            adata_sp.obsm[f"{ct_pred_key}_smooth"] = pd.DataFrame(
                P_smooth, index=adata_sp.obs_names, columns=ct_df.columns
            )
            result["proportions_smooth"] = P_smooth

    return result


def allocate_integer_counts(C_float: ndarray, c: ndarray) -> ndarray:
    C_int = np.floor(C_float).astype(int)  # 取整数部分
    remainders = C_float - C_int  # 小数部分

    for i in range(len(c)):
        target = c[i]
        current_sum = C_int[i].sum()
        remaining = target - current_sum

        if remaining > 0:
            # 获取当前点位的小数部分，按降序排列的索引
            decimals = remainders[i]
            sorted_indices = np.argsort(-decimals)  # 降序排列

            # 选择前remaining个索引（小数部分最大的）
            top_indices = sorted_indices[:remaining]
            C_int[i, top_indices] += 1

    # 验证行和正确性
    assert np.allclose(C_int.sum(axis=1), c), "行和与目标值不符！"
    return C_int


def count_cell_annotations2(
    adata_map: AnnData,
    adata_sc: AnnData,
    adata_sp: AnnData,
    annotation: str = "cell_type",
) -> None:
    if "spatial" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing spatial information in AnnDatas. Please make sure coordinates are saved with AnnData.obsm['spatial']"
        )

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for spnmf deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    if (
        "spnmf_cell_segmentation" not in adata_sp.uns.keys()
        or "spnmf_spot_centroids" not in adata_sp.obsm.keys()
    ):
        raise ValueError(
            "Missing parameter for spnmf deconvolution. Run `create_segment_cell_df`."
        )

    xs = adata_sp.obsm["spatial"][:, 1]
    ys = adata_sp.obsm["spatial"][:, 0]
    cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

    df_segmentation = adata_sp.uns["spnmf_cell_segmentation"]
    centroids = adata_sp.obsm["spnmf_spot_centroids"]

    # create a dataframe
    df_vox_cells = df_vox_cells = pd.DataFrame(
        data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
        index=list(adata_sp.obs.index),
    )

    # get the most probable voxel for each cell
    resulting_voxels = np.argmax(adata_map.X, axis=1)

    # create a list with filtered cells and the voxels where they have been placed with the
    # highest probability a cell i is filtered if F_i > threshold'

    vox_ct = list(zip(resulting_voxels, adata_sc.obs[annotation]))

    df_classes = one_hot_encoding(adata_sc.obs[annotation])
    for index, i in enumerate(df_classes.columns):
        df_vox_cells[i] = 0

    for k, v in vox_ct:
        df_vox_cells.iloc[k, df_vox_cells.columns.get_loc(v)] += 1

    adata_sp.obsm["spnmf_ct_count"] = df_vox_cells
    logging.info(
        f"spatial cell count dataframe is saved in `obsm` `spnmf_ct_count` of the spatial AnnData."
    )


def count_cell_annotations(
    adata_map: AnnData,
    adata_sc: AnnData,
    adata_sp: AnnData,
    annotation: str = "cell_type",
) -> None:
    # 校验输入完整性
    if "spatial" not in adata_sp.obsm:
        raise ValueError("Missing spatial coordinates in adata_sp.obsm['spatial']")
    if "image_features" not in adata_sp.obsm:
        raise ValueError("Run squidpy.im.calculate_image_features first")
    if "spnmf_cell_segmentation" not in adata_sp.uns:
        raise ValueError("Run create_segment_cell_df first")

    xs = adata_sp.obsm["spatial"][:, 1]
    ys = adata_sp.obsm["spatial"][:, 0]
    cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

    # df_segmentation = adata_sp.uns["spnmf_cell_segmentation"]
    centroids = adata_sp.obsm["spnmf_spot_centroids"]

    # 提取关键数据
    M = adata_map.X  # 概率矩阵 (cells × spots)
    c = adata_sp.obsm["image_features"]["segmentation_label"].astype(
        float
    )  # 点位细胞数向量
    type = one_hot_encoding(adata_sc.obs[annotation])
    T = type.values  # 细胞类型独热编码矩阵

    # 概率矩阵缩放
    S = np.array(M.sum(axis=0)).flatten()  # 计算各点位总概率
    valid_spots = (S > 1e-8) & (c > 0)  # 处理除零情况
    scale_factors = np.zeros_like(S)
    scale_factors[valid_spots] = c[valid_spots] / S[valid_spots]

    # 核心矩阵运算: C = diag(c/S) * M.T * T
    M_scaled = M.multiply(scale_factors).T if issparse(M) else (M * scale_factors).T
    C = M_scaled @ T

    # 转换为整数矩阵
    c_values = adata_sp.obsm["image_features"]["segmentation_label"].values.astype(int)
    C_int = allocate_integer_counts(C.astype(float), c_values)  # 确保输入为浮点

    # 构建输出数据框
    df_ct = pd.DataFrame(
        C_int,
        index=list(adata_sp.obs.index),
        columns=type.columns,
    )

    df_vox_cells = pd.DataFrame(
        data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
        index=list(adata_sp.obs.index),
    )

    df_vox_cells = df_vox_cells.join(df_ct)

    # 验证行和等于c (处理浮点误差)
    # np.testing.assert_allclose(df_ct.sum(axis=1), c, rtol=1e-3)
    adata_sp.obsm["spnmf_ct_count"] = df_vox_cells
    adata_sp.uns["spnmf_cell_types"] = type.columns


def df_to_cell_types(df: DataFrame, cell_types) -> defaultdict:
    df_cum_sums = df[cell_types].cumsum(axis=1)

    df_c = df.copy()

    for i in df_cum_sums.columns:
        df_c[i] = df_cum_sums[i]

    cell_types_mapped = defaultdict(list)
    for i_index, i in enumerate(cell_types):
        for j_index, j in df_c.iterrows():
            start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
            end_ind = j[i]
            cell_types_mapped[i].extend(j["centroids"][start_ind:end_ind].tolist())
    return cell_types_mapped


def deconvolve_cell_annotations2(
    adata_sp: AnnData, filter_cell_annotation=None, cluster_label="cluster"
) -> AnnData:
    # 1. 检查必要的 spnmf 计算结果是否存在
    if (
        "spnmf_ct_count" not in adata_sp.obsm
        or "spnmf_cell_segmentation" not in adata_sp.uns
    ):
        raise ValueError(
            "Missing spnmf parameters. Run `count_cell_annotations` first."
        )

    # 2. 取出 segmentation DataFrame（每行对应一个分割出的 voxel）
    segmentation_df = adata_sp.uns["spnmf_cell_segmentation"].copy()
    #   这里假设 segmentation_df 至少包含列 ['spot_idx', 'centroids', 'x', 'y', ...]
    #   其中 'spot_idx' 对应的是 adata_sp.obs.index 中的 spot ID（通常是字符串）。

    # 3. 确定要过滤的细胞类型列名
    if filter_cell_annotation is None:
        filter_cell_annotation = adata_sp.uns["spnmf_cell_types"]
    else:
        filter_cell_annotation = pd.unique(filter_cell_annotation)

    # 4. 构建 “每个 voxel 属于哪个细胞类型” 的映射
    #    adata_sp.obsm["spnmf_ct_count"] 是一个 DataFrame：
    #    index 顺序与 segmentation_df 一一对应，列为所有细胞类型。
    df_vox_cells = adata_sp.obsm["spnmf_ct_count"].copy()

    #    假定 df_to_cell_types(df_vox_cells, filter_cell_annotation) 返回一个 dict：
    #      { cell_type_name: [centroid1, centroid2, ...], ... }
    #    即每个细胞类型对应一个包含该类型所有 voxel centroid 的列表。
    cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)

    # 5. 将 “centroid ↔ 细胞类型” 映成一个 DataFrame
    df_list = []
    for cell_type, centroid_list in cell_types_mapped.items():
        tmp = pd.DataFrame(
            {
                "centroids": np.array(centroid_list, dtype="object"),
                cluster_label: cell_type,
            }
        )
        df_list.append(tmp)
    cluster_df = pd.concat(df_list, axis=0).reset_index(drop=True)
    # cluster_df 列为 ['centroids', 'cluster']，每行是一个 voxel centroid 对应的细胞类型

    # 6. 将 segmentation_df 与 cluster_df 在 'centroids' 列上合并，
    #    以获得 voxel 对应的 spot_idx（spot ID）、(x, y) 坐标，以及 cluster（cell type）
    merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
    #    merged_df 至少含 ['spot_idx', 'centroids', 'x', 'y', ..., 'cluster']
    #    去重以防万一
    merged_df = merged_df.drop_duplicates(
        subset=["centroids", cluster_label, "spot_idx"]
    ).copy()

    # 7. 对每个 spot_idx（spot ID）分组，取出现次数最多的 'cluster' 作为该 spot 的标注
    def mode_per_group(subdf):
        # 对该 spot 分组，返回出现频率最高的 cluster
        return subdf[cluster_label].value_counts().idxmax()

    # spot_to_cluster: Index = spot_idx (spot ID, 与 adata_sp.obs.index 对应)，
    # value = 该 spot 的优势细胞类型
    spot_to_cluster = (
        merged_df.groupby("spot_idx")
        .apply(mode_per_group)
        .rename(cluster_label)
        .astype("category")
    )
    # spot_to_cluster 的 index dtype 与 adata_sp.obs.index dtype 一致（通常是字符串）

    # 8. 构造新的 obs：在原始 adata_sp.obs 的基础上，添加一列 'cluster'
    new_obs = adata_sp.obs.copy(deep=True)
    # 初始化一列全为 pd.NA 的 category，category 的 categories 取自 spot_to_cluster 中的类别
    new_obs[cluster_label] = pd.Categorical(
        [pd.NA] * adata_sp.n_obs, categories=spot_to_cluster.cat.categories
    )

    # 9. 将 spot_to_cluster 中每个 spot_id 对应的 cluster 填到 new_obs
    for spot_id, ct in spot_to_cluster.items():
        if spot_id in new_obs.index:
            new_obs.at[spot_id, cluster_label] = ct

    # 10. 生成一个新的 AnnData，保持 X、var、obsm["spatial"]、uns 不变，仅替换 obs
    adata_segment = sc.AnnData(
        X=adata_sp.X.copy(),  # n_obs × n_vars
        obs=new_obs,  # 新的 obs（带 'cluster' 列）
        var=adata_sp.var.copy(),  # 保留 var
        obsm={"spatial": adata_sp.obsm["spatial"].copy()},
        uns=adata_sp.uns.copy(),
    )

    return adata_segment


def deconvolve_cell_annotations(
    adata_sp: AnnData, filter_cell_annotation=None, cluster_label="cluster"
) -> AnnData:
    if (
        "spnmf_ct_count" not in adata_sp.obsm.keys()
        or "spnmf_cell_segmentation" not in adata_sp.uns.keys()
    ):
        raise ValueError("Missing spnmf parameters. Run `count_cell_annotations`.")

    segmentation_df = adata_sp.uns["spnmf_cell_segmentation"]

    if filter_cell_annotation is None:
        filter_cell_annotation = adata_sp.uns["spnmf_cell_types"]
    else:
        filter_cell_annotation = pd.unique(filter_cell_annotation)

    df_vox_cells = adata_sp.obsm["spnmf_ct_count"]
    cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)
    df_list = []
    for k in cell_types_mapped.keys():
        df = pd.DataFrame({"centroids": np.array(cell_types_mapped[k], dtype="object")})
        df[cluster_label] = k
        df_list.append(df)
    cluster_df = pd.concat(df_list, axis=0)
    cluster_df.reset_index(inplace=True, drop=True)

    merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
    merged_df.drop(columns="spot_idx", inplace=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(inplace=True, drop=True)

    adata_segment = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    adata_segment.obsm["spatial"] = merged_df[["y", "x"]].to_numpy()
    adata_segment.uns = adata_sp.uns

    return adata_segment
