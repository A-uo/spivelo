import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from pandas import DataFrame
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .utils import eval_metric
from numpy import ndarray
import matplotlib as mpl

def construct_obs_plot(
    df_plot: DataFrame, adata: AnnData, perc: float = 0, suffix: str | None = None
) -> None:
    # clip
    df_plot = df_plot.clip(df_plot.quantile(perc), df_plot.quantile(1 - perc), axis=1)

    # normalize
    df_plot = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())

    if suffix:
        df_plot = df_plot.add_suffix(f" ({suffix})")
    adata.obs = pd.concat([adata.obs, df_plot], axis=1)


def plot_cell_annotation_sc(
    adata_sp: AnnData,
    annotation_list: list[str],
    x: str = "x",
    y: str = "y",
    spot_size: float | None = None,
    scale_factor: float | None = None,
    perc: float = 0,
    alpha_img: float = 1.0,
    bw: bool = False,
    ncols: int = 5,       # 每行 11 列
    font_size: int = 25,   # 字体变大
) -> None:
    # 清理之前的 obs 中同名列
    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)

    # 构建 df_plot
    df = adata_sp.obsm["spnmf_ct_pred"][annotation_list]
    construct_obs_plot(df, adata_sp, perc=perc)

    # 处理非 Visium 数据：补 spatial 坐标
    if "spatial" not in adata_sp.obsm.keys():
        coords = np.vstack([adata_sp.obs[x].values, adata_sp.obs[y].values]).T
        adata_sp.obsm["spatial"] = coords

    # 参数检查
    if (
        "spatial" not in adata_sp.uns.keys()
        and spot_size is None
        and scale_factor is None
    ):
        raise ValueError(
            "Spot Size and Scale Factor cannot be None when ad_sp.uns['spatial'] does not exist"
        )

    if (
        "spatial" in adata_sp.uns.keys()
        and spot_size is not None
        and scale_factor is not None
    ):
        raise ValueError(
            "Spot Size and Scale Factor should be None when ad_sp.uns['spatial'] exists"
        )

    # 在局部上下文中调大字体 + 指定每行 ncols=11
    with mpl.rc_context(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": font_size - 2,
        }
    ):
        sc.pl.spatial(
            adata_sp,
            color=annotation_list,
            cmap="viridis",
            show=False,       # 直接显示
            frameon=False,
            spot_size=spot_size,
            scale_factor=scale_factor,
            alpha_img=alpha_img,
            bw=bw,
            ncols=ncols,     # 关键：每行 11 个 panel
        )
    plt.savefig("figures/cell_maping_mouse_chicken.png",bbox_inches='tight',dpi=300)
    # 用完后把临时加到 obs 的列删掉
    adata_sp.obs.drop(annotation_list, inplace=True, errors="ignore", axis=1)


def plot_training_scores(
    adata_map: AnnData, bins: int = 10, alpha: float = 0.7
) -> None:
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    df = adata_map.uns["train_genes_df"]
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    #     axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df, y="train_score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_sc",
        ax=axs_f[1],
        alpha=alpha,
        color="coral",
    )

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_sp",
        ax=axs_f[2],
        alpha=alpha,
        color="coral",
    )

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df,
        y="train_score",
        x="sparsity_diff",
        ax=axs_f[3],
        alpha=alpha,
        color="coral",
    )

    plt.tight_layout()


# def plot_genes_sc(
#     genes: list[str],
#     adata_measured: AnnData,
#     adata_predicted: AnnData,
#     x: str = "x",
#     y: str = "y",
#     spot_size: float | None = None,
#     scale_factor: float | None = None,
#     cmap: str = "inferno",
#     perc: float = 0,
#     alpha_img: float = 1.0,
#     bw: bool = False,
#     return_figure: bool = False,
# ) -> Figure | None:
#     # remove df_plot in obs
#     adata_measured.obs.drop(
#         [f"{gene} (measured)" for gene in genes],
#         inplace=True,
#         errors="ignore",
#         axis=1,
#     )
#     adata_predicted.obs.drop(
#         [f"{gene} (predicted)" for gene in genes],
#         inplace=True,
#         errors="ignore",
#         axis=1,
#     )

#     adata_measured.var_names = [g.lower() for g in adata_measured.var_names]
#     adata_predicted.var_names = [g.lower() for g in adata_predicted.var_names]

#     adata_predicted.obsm = adata_measured.obsm
#     adata_predicted.uns = adata_measured.uns

#     # remove previous df_plot in obs
#     adata_measured.obs.drop(
#         [f"{gene} (measured)" for gene in genes],
#         inplace=True,
#         errors="ignore",
#         axis=1,
#     )
#     adata_predicted.obs.drop(
#         [f"{gene} (predicted)" for gene in genes],
#         inplace=True,
#         errors="ignore",
#         axis=1,
#     )

#     # construct df_plot
#     data = []
#     for ix, gene in enumerate(genes):
#         if gene not in adata_measured.var_names:
#             data.append(np.zeros_like(adata_measured[:, 0].X.toarray().flatten()))
#         else:
#             data.append(adata_measured[:, gene].X.toarray().flatten())

#     df = pd.DataFrame(
#         data=np.array(data).T,
#         columns=genes,
#         index=adata_measured.obs_names,
#     )
#     construct_obs_plot(df, adata_measured, suffix="measured")

#     df = pd.DataFrame(
#         data=adata_predicted[:, genes].X.toarray(),
#         columns=genes,
#         index=adata_predicted.obs_names,
#     )
#     construct_obs_plot(df, adata_predicted, perc=perc, suffix="predicted")

#     fig = plt.figure(figsize=(7, len(genes) * 3.5))
#     gs = GridSpec(len(genes), 2, figure=fig)

#     # non visium data
#     if "spatial" not in adata_measured.obsm.keys():
#         # add spatial coordinates to obsm of spatial data
#         coords = [
#             [x, y]
#             for x, y in zip(adata_measured.obs[x].values, adata_measured.obs[y].values)
#         ]
#         adata_measured.obsm["spatial"] = np.array(coords)
#         coords = [
#             [x, y]
#             for x, y in zip(
#                 adata_predicted.obs[x].values, adata_predicted.obs[y].values
#             )
#         ]
#         adata_predicted.obsm["spatial"] = np.array(coords)

#     if ("spatial" not in adata_measured.uns.keys()) and (
#         spot_size is None and scale_factor is None
#     ):
#         raise ValueError(
#             "Spot Size and Scale Factor cannot be None when ad_sp.uns['spatial'] does not exist"
#         )

#     for ix, gene in enumerate(genes):
#         ax_m = fig.add_subplot(gs[ix, 0])
#         sc.pl.spatial(
#             adata_measured,
#             spot_size=spot_size,
#             scale_factor=scale_factor,
#             color=[f"{gene} (measured)"],
#             frameon=False,
#             ax=ax_m,
#             show=False,
#             cmap=cmap,
#             alpha_img=alpha_img,
#             bw=bw,
#         )
#         ax_p = fig.add_subplot(gs[ix, 1])
#         sc.pl.spatial(
#             adata_predicted,
#             spot_size=spot_size,
#             scale_factor=scale_factor,
#             color=[f"{gene} (predicted)"],
#             frameon=False,
#             ax=ax_p,
#             show=False,
#             cmap=cmap,
#             alpha_img=alpha_img,
#             bw=bw,
#         )

#     #     sc.pl.spatial(adata_measured, color=['{} (measured)'.format(gene) for gene in genes], frameon=False)
#     #     sc.pl.spatial(adata_predicted, color=['{} (predicted)'.format(gene) for gene in genes], frameon=False)

#     # remove df_plot in obs
#     adata_measured.obs.drop(
#         [f"{gene} (measured)" for gene in genes],
#         inplace=True,
#         errors="ignore",
#         axis=1,
#     )
#     adata_predicted.obs.drop(
#         [f"{gene} (predicted)" for gene in genes],
#         inplace=True,
#         errors="ignore",
#         axis=1,
#     )

#     if return_figure:
#         return fig
#     else:
#         return None

def plot_genes_sc(
    genes, 
    adata_measured, 
    adata_predicted,
    x="x",
    y="y",
    spot_size=None, 
    scale_factor=None, 
    cmap="inferno", 
    perc=0,
    alpha_img=1.0,
    bw=False,
    return_figure=False,
    save_path=None,         # 新增：保存路径（如 "out/plots.png" 或 "plots.pdf"）
    dpi=300                 # 新增：保存分辨率
):
    """
    绘制 measured vs. predicted 的空间表达并可选保存图片。

    Parameters
    ----------
    genes : list[str]
        需要展示的基因名列表。
    adata_measured, adata_predicted : anndata.AnnData
        实测与预测的 AnnData。
    x, y : str
        非 Visium 数据时，obs 中表示坐标的列名。
    spot_size, scale_factor : float | None
        传给 sc.pl.spatial 的点大小/缩放系数。
    cmap : str
        颜色映射。
    perc : float
        传给 construct_obs_plot 的参数（保持原逻辑）。
    alpha_img : float
        背景图透明度。
    bw : bool
        是否显示灰度背景（sc.pl.spatial 选项）。
    return_figure : bool
        是否返回 matplotlib Figure。
    save_path : str | None
        若提供，则保存到此路径（后缀决定格式，如 .png/.pdf/.svg）。
    dpi : int
        保存分辨率（对位图格式生效）。
    """

    # remove df_plot in obs
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )

    # prepare adatas
    convert_adata_array(adata_measured)

    adata_measured.var.index = [g.lower() for g in adata_measured.var.index]
    adata_predicted.var.index = [g.lower() for g in adata_predicted.var.index]

    adata_predicted.obsm = adata_measured.obsm
    adata_predicted.uns = adata_measured.uns

    # remove previous df_plot in obs
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )

    # construct df_plot
    data = []
    for ix, gene in enumerate(genes):
        if gene not in adata_measured.var.index:
            data.append(np.zeros_like(np.array(adata_measured[:, 0].X).flatten()))
        else:
            data.append(np.array(adata_measured[:, gene].X).flatten())

    df = pd.DataFrame(
        data=np.array(data).T, columns=genes, index=adata_measured.obs.index,
    )
    construct_obs_plot(df, adata_measured, suffix="measured")

    df = pd.DataFrame(
        data=np.array(adata_predicted[:, genes].X),
        columns=genes,
        index=adata_predicted.obs.index,
    )
    construct_obs_plot(df, adata_predicted, perc=perc, suffix="predicted")

    fig = plt.figure(figsize=(7, len(genes) * 3.5))
    gs = GridSpec(len(genes), 2, figure=fig)
    
    # non-Visium data：从 obs 的 x/y 列构造坐标
    if 'spatial' not in adata_measured.obsm.keys():
        coords = [[xv, yv] for xv, yv in zip(adata_measured.obs[x].values, adata_measured.obs[y].values)]
        adata_measured.obsm['spatial'] = np.array(coords)
        coords = [[xv, yv] for xv, yv in zip(adata_predicted.obs[x].values, adata_predicted.obs[y].values)]
        adata_predicted.obsm['spatial'] = np.array(coords)

    if ("spatial" not in adata_measured.uns.keys()) and (spot_size is None and scale_factor is None):
        raise ValueError("Spot Size and Scale Factor cannot be None when ad_sp.uns['spatial'] does not exist")
        
    for ix, gene in enumerate(genes):
        ax_m = fig.add_subplot(gs[ix, 0])
        sc.pl.spatial(
            adata_measured,
            spot_size=spot_size,
            scale_factor=scale_factor,
            color=["{} (measured)".format(gene)],
            frameon=False,
            ax=ax_m,
            show=False,
            cmap=cmap,
            alpha_img=alpha_img,
            bw=bw
        )
        ax_m.set_title(f"{gene} (measured)")

        ax_p = fig.add_subplot(gs[ix, 1])
        sc.pl.spatial(
            adata_predicted,
            spot_size=spot_size,
            scale_factor=scale_factor,
            color=["{} (predicted)".format(gene)],
            frameon=False,
            ax=ax_p,
            show=False,
            cmap=cmap,
            alpha_img=alpha_img,
            bw=bw
        )
        ax_p.set_title(f"{gene} (predicted)")
        
    # 清理添加到 obs 的列
    adata_measured.obs.drop(
        ["{} (measured)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )
    adata_predicted.obs.drop(
        ["{} (predicted)".format(gene) for gene in genes],
        inplace=True,
        errors="ignore",
        axis=1,
    )

    # 布局更紧凑
    fig.tight_layout()

    # === 新增：保存图片 ===
    if save_path is not None:
        # 若提供了文件夹，确保存在
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        # 保存（根据后缀决定格式，如 .png/.pdf/.svg）
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    
    if return_figure:
        return fig
    else:
        # 不返回时，为了避免内存占用，主动关闭
        plt.close(fig)


def plot_auc(df_all_genes: DataFrame, test_genes: ndarray | list | None = None) -> None:
    metric_dict, ((pol_xs, pol_ys), (xs, ys)) = eval_metric(df_all_genes, test_genes)

    plt.figure(figsize=(6, 5))

    plt.plot(pol_xs, pol_ys, c="r")
    sns.scatterplot(x=xs, y=ys, alpha=0.5, edgecolors="face")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(0.5)
    plt.xlabel("score")
    plt.ylabel("spatial sparsity")
    plt.tick_params(axis="both", labelsize=8)
    plt.title("Prediction on test transcriptome")

    textstr = "auc_score={}".format(np.round(metric_dict["auc_score"], 3))
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.3)
    # place a text box in upper left in axes coords
    plt.text(0.03, 0.1, textstr, fontsize=11, verticalalignment="top", bbox=props)
