# TI/single_time/transition.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import ot

from anndata import AnnData
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors, KernelDensity


def _normalize_array(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + eps)


def compute_density_and_confidence(
    adata: AnnData,
    spatial_key: str = "spatial",
    density_bandwidth: float = 50.0,
    reliability_k: int = 20,
    reliability_threshold: float = 0.35,
    edge_quantile: float = 0.03,
    hole_neighbor_quantile: float = 0.90,
    inplace: bool = True,
) -> dict:
    """
    Compute local density, local reliability, and identify low-confidence regions.

    Returns:
        {
            "density_scores": ...,
            "density_norm": ...,
            "mean_knn_dist": ...,
            "kth_dist": ...,
            "reliability_norm": ...,
            "edge_mask": ...,
            "hole_mask": ...,
            "low_rel_mask": ...,
            "low_conf_mask": ...
        }
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not found in adata.obsm")

    spatial_coords = np.asarray(adata.obsm[spatial_key], dtype=float)

    # 1) KDE density
    kde = KernelDensity(bandwidth=density_bandwidth, kernel="gaussian")
    kde.fit(spatial_coords)
    density_scores = np.exp(kde.score_samples(spatial_coords))
    density_norm = _normalize_array(density_scores)

    # 2) KNN reliability
    nn = NearestNeighbors(n_neighbors=reliability_k + 1)
    nn.fit(spatial_coords)
    distances, _ = nn.kneighbors(spatial_coords)

    knn_dists = distances[:, 1:]
    kth_dist = distances[:, reliability_k]
    mean_knn_dist = knn_dists.mean(axis=1)

    reliability_raw = 1.0 / (mean_knn_dist + 1e-8)
    reliability_norm = _normalize_array(reliability_raw)

    # 3) edge / hole / low reliability
    x = spatial_coords[:, 0]
    y = spatial_coords[:, 1]

    x_low = np.quantile(x, edge_quantile)
    x_high = np.quantile(x, 1 - edge_quantile)
    y_low = np.quantile(y, edge_quantile)
    y_high = np.quantile(y, 1 - edge_quantile)

    edge_mask = (x <= x_low) | (x >= x_high) | (y <= y_low) | (y >= y_high)

    local_sparse_mask = mean_knn_dist >= np.quantile(mean_knn_dist, hole_neighbor_quantile)
    hole_mask = (kth_dist >= np.quantile(kth_dist, hole_neighbor_quantile)) | local_sparse_mask

    low_rel_mask = reliability_norm < reliability_threshold
    low_conf_mask = edge_mask | hole_mask | low_rel_mask

    result = {
        "density_scores": density_scores,
        "density_norm": density_norm,
        "mean_knn_dist": mean_knn_dist,
        "kth_dist": kth_dist,
        "reliability_norm": reliability_norm,
        "edge_mask": edge_mask,
        "hole_mask": hole_mask,
        "low_rel_mask": low_rel_mask,
        "low_conf_mask": low_conf_mask,
    }

    if inplace:
        adata.obs["cell_density"] = density_norm
        adata.obs["density_score_raw"] = density_scores
        adata.obs["mean_knn_dist"] = mean_knn_dist
        adata.obs["kth_dist"] = kth_dist
        adata.obs["reliability"] = reliability_norm

        adata.obs["edge_mask"] = edge_mask
        adata.obs["hole_mask"] = hole_mask
        adata.obs["low_reliability_mask"] = low_rel_mask
        adata.obs["low_confidence_region"] = low_conf_mask

        adata.obs["edge_label"] = pd.Categorical(
            np.where(edge_mask, "Edge", "Interior"),
            categories=["Interior", "Edge"]
        )
        adata.obs["hole_label"] = pd.Categorical(
            np.where(hole_mask, "HoleLike", "Normal"),
            categories=["Normal", "HoleLike"]
        )
        adata.obs["low_reliability_label"] = pd.Categorical(
            np.where(low_rel_mask, "LowRel", "OK"),
            categories=["OK", "LowRel"]
        )
        adata.obs["low_confidence_label"] = pd.Categorical(
            np.where(low_conf_mask, "LowConf", "OK"),
            categories=["OK", "LowConf"]
        )

    return result

def construct_protected_adaptive_transition_matrix(
    adata: AnnData,
    StPedf_key: str = "StPedf",
    spatial_key: str = "spatial",
    reg: float = 0.005,
    alpha_range: tuple = (0.4, 0.6),
    protected_alpha: float = 0.4,
    density_bandwidth: float = 50.0,
    reliability_k: int = 20,
    reliability_threshold: float = 0.35,
    edge_quantile: float = 0.03,
    hole_neighbor_quantile: float = 0.90,
    obsp_key: str | None = "trans",
    inplace: bool = True,
    plot: bool = False,
    spot_size: float = 50,
    return_cost: bool = False,
):
    """
    Construct a "protected density-adaptive" transition matrix.

    Normal regions:
        High density -> low spatial weight
        Low density -> high spatial weight

    Low-confidence regions (edge / hole-like / poor connectivity):
        Use a fixed lower protected_alpha to avoid over-reliance on spatial distance.
    """
    if StPedf_key not in adata.obsm:
        raise KeyError(f"'{StPedf_key}' not found in adata.obsm")
    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not found in adata.obsm")

    StPedf_emb = np.asarray(adata.obsm[StPedf_key], dtype=float)
    spatial_coords = np.asarray(adata.obsm[spatial_key], dtype=float)

    # Distance matrices
    StPedf_dist = pairwise_distances(StPedf_emb)
    spatial_dist = pairwise_distances(spatial_coords)

    StPedf_dist = StPedf_dist / (StPedf_dist.max() + 1e-10)
    spatial_dist = spatial_dist / (spatial_dist.max() + 1e-10)

    # Density + low-confidence region identification
    qc = compute_density_and_confidence(
        adata=adata,
        spatial_key=spatial_key,
        density_bandwidth=density_bandwidth,
        reliability_k=reliability_k,
        reliability_threshold=reliability_threshold,
        edge_quantile=edge_quantile,
        hole_neighbor_quantile=hole_neighbor_quantile,
        inplace=inplace,
    )

    density_norm = qc["density_norm"]
    low_conf_mask = qc["low_conf_mask"]

    # Raw adaptive weight: high density -> low spatial weight, low density -> high spatial weight
    min_alpha, max_alpha = alpha_range
    raw_spatial_weight = min_alpha + (max_alpha - min_alpha) * (1 - density_norm)

    # Protection for low-confidence regions
    spatial_weight = raw_spatial_weight.copy()
    spatial_weight[low_conf_mask] = protected_alpha
    spatial_weight = np.clip(spatial_weight, 0.0, 1.0)

    if inplace:
        adata.obs["raw_spatial_weight"] = raw_spatial_weight
        adata.obs["spatial_weight"] = spatial_weight

    # Fused cost matrix
    # M[i, j] = alpha_i * spatial_dist[i, j] + (1 - alpha_i) * sedr_dist[i, j]
    M = spatial_weight[:, None] * spatial_dist + (1.0 - spatial_weight[:, None]) * StPedf_dist

    # Sinkhorn
    n = M.shape[0]
    a = np.ones(n) / n
    b = np.ones(n) / n
    Gs = ot.sinkhorn(a, b, M, reg=reg)

    if inplace and obsp_key is not None:
        adata.obsp[obsp_key] = Gs

    if plot:
        plot_transition_diagnostics(
            adata=adata,
            spatial_key=spatial_key,
            spot_size=spot_size,
            protected_alpha=protected_alpha,
        )

    print("========== Protected adaptive transition summary ==========")
    print(f"alpha_range = {alpha_range}")
    print(f"protected_alpha = {protected_alpha}")
    print(f"edge count = {adata.obs['edge_mask'].sum()} / {len(adata)}")
    print(f"hole-like count = {adata.obs['hole_mask'].sum()} / {len(adata)}")
    print(f"low-reliability count = {adata.obs['low_reliability_mask'].sum()} / {len(adata)}")
    print(f"low-confidence count = {adata.obs['low_confidence_region'].sum()} / {len(adata)}")
    print("==========================================================")

    if return_cost:
        return Gs, M
    return Gs


def plot_transition_diagnostics(
    adata: AnnData,
    spatial_key: str = "spatial",
    spot_size: float = 50,
    protected_alpha: float = 0.4,
):
    """
    Visualize density, adaptive weights, and low-confidence regions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color="cell_density",
        ax=axes[0, 0],
        show=False,
        title="Cell density",
        cmap="viridis",
        size=spot_size,
    )

    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color="reliability",
        ax=axes[0, 1],
        show=False,
        title="Reliability",
        cmap="magma",
        size=spot_size,
    )

    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color="raw_spatial_weight",
        ax=axes[0, 2],
        show=False,
        title="Raw adaptive weight",
        cmap="coolwarm",
        size=spot_size,
        vmin=0,
        vmax=1,
    )

    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color="spatial_weight",
        ax=axes[1, 0],
        show=False,
        title=f"Final weight (protected α={protected_alpha})",
        cmap="coolwarm",
        size=spot_size,
        vmin=0,
        vmax=1,
    )

    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color="low_confidence_label",
        ax=axes[1, 1],
        show=False,
        title="Low-confidence region",
        palette={"OK": "lightgrey", "LowConf": "red"},
        size=spot_size,
    )

    sc.pl.embedding(
        adata,
        basis=spatial_key,
        color="hole_label",
        ax=axes[1, 2],
        show=False,
        title="Hole-like region",
        palette={"Normal": "lightgrey", "HoleLike": "blue"},
        size=spot_size,
    )

    plt.tight_layout()
    plt.show()


def construct_adaptive_transition_matrix(
    adata: AnnData,
    StPedf_key: str = 'StPedf',
    spatial_key: str = 'spatial',
    reg: float = 0.1,
    base_alpha: float = 0.5,  
    alpha_range: tuple = (0.3, 0.7), 
    spot_size: float = 100  
):
    """
    Construct a density-adaptive transition matrix without low-confidence protection.
    """

    StPedf_emb = adata.obsm[StPedf_key]
    spatial_coords = adata.obsm[spatial_key]

    StPedf_dist = pairwise_distances(StPedf_emb)
    spatial_dist = pairwise_distances(spatial_coords)

    StPedf_dist /= StPedf_dist.max()
    spatial_dist /= spatial_dist.max()

    from sklearn.neighbors import KernelDensity

    kde = KernelDensity(bandwidth=50, kernel='gaussian')
    kde.fit(spatial_coords)
    density_scores = np.exp(kde.score_samples(spatial_coords))

    density_norm = (density_scores - density_scores.min()) / (
        density_scores.max() - density_scores.min() + 1e-10)

    min_alpha, max_alpha = alpha_range
    adaptive_alphas = min_alpha + (max_alpha - min_alpha) * (1 - density_norm)

    adata.obs['cell_density'] = density_norm
    adata.obs['spatial_weight'] = adaptive_alphas

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sc.pl.embedding(
        adata, 
        basis=spatial_key, 
        color='cell_density', 
        ax=axes[0], 
        show=False, 
        title='Cell Density',
        cmap='viridis',
        size=spot_size
    )

    sc.pl.embedding(
        adata, 
        basis=spatial_key, 
        color='spatial_weight', 
        ax=axes[1], 
        show=False, 
        title='Adaptive Weights',
        cmap='coolwarm',
        size=spot_size,
        vmin=min_alpha,
        vmax=max_alpha
    )

    plt.tight_layout()
    plt.show()

    M = np.zeros_like(spatial_dist)
    for i in range(len(adata)):
        alpha_i = adaptive_alphas[i]
        M[i] = alpha_i * spatial_dist[i] + (1 - alpha_i) * StPedf_dist[i]

    a = b = np.ones(M.shape[0]) / M.shape[0]
    Gs = ot.sinkhorn(a, b, M, reg=reg)

    return Gs