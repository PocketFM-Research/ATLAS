"""
k-NN graph construction, eigengap cluster estimation, and k-means clustering
for entity normalization (Appendix C.2).

Steps:
1. Build symmetric k-NN graph from the combined similarity matrix.
2. Compute graph Laplacian L = D - A.
3. Estimate cluster count k* via eigengap heuristic over non-trivial modes.
4. Run k-means with k* clusters on joint embeddings.
5. Return clusters as lists of indices.
"""

import logging
import math
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def build_knn_graph(S: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Build a symmetric k-nearest-neighbor adjacency matrix from similarity S.

    Args:
        S: (N, N) pairwise similarity matrix.
        k: Number of nearest neighbours per node.

    Returns:
        (N, N) symmetric binary adjacency matrix A.
    """
    N = S.shape[0]
    k = min(k, N - 1)
    A = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        row = S[i].copy()
        row[i] = -np.inf  # exclude self
        nn_idx = np.argpartition(row, -k)[-k:]
        A[i, nn_idx] = 1.0

    # Symmetrize
    A = np.maximum(A, A.T)
    return A


def eigengap_cluster_count(A: np.ndarray, max_clusters: int = 20) -> int:
    """
    Estimate the number of clusters via the eigengap heuristic (Appendix C.2).

    Computes the normalized graph Laplacian, finds its eigenvalues, and
    returns the index of the largest gap in the sorted spectrum.

    Args:
        A: (N, N) symmetric adjacency matrix.
        max_clusters: Upper bound on cluster count.

    Returns:
        Estimated number of clusters k* (minimum 1).
    """
    N = A.shape[0]
    if N <= 1:
        return 1

    # Degree matrix and normalized Laplacian: L = D^{-1/2} (D-A) D^{-1/2}
    D = A.sum(axis=1)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    D_mat = np.diag(D_inv_sqrt)
    L_sym = np.eye(N) - D_mat @ A @ D_mat

    try:
        from scipy.linalg import eigh
        eigenvalues = eigh(L_sym, eigvals_only=True, subset_by_index=[0, min(max_clusters, N - 1)])
    except Exception as e:
        logger.warning("Eigengap computation failed: %s — defaulting to 1 cluster", e)
        return 1

    eigenvalues = np.sort(eigenvalues)

    # Eigengap: find largest gap between consecutive non-trivial eigenvalues
    # Skip the first (always ~0 for a connected graph)
    nontrivial = eigenvalues[1:]  # skip λ_0 ≈ 0
    if len(nontrivial) < 2:
        return 1

    gaps = np.diff(nontrivial)
    k_star = int(np.argmax(gaps)) + 2  # +1 for skipped λ_0, +1 for 1-indexed

    # Additional regularization for small N (avoid over-aggressive merging)
    if N <= 10:
        k_star = max(k_star, max(1, N // 3))

    k_star = max(1, min(k_star, max_clusters, N))
    logger.debug("Eigengap estimated k*=%d (N=%d)", k_star, N)
    return k_star


def kmeans_cluster(embeddings: np.ndarray, k: int, n_init: int = 5) -> np.ndarray:
    """
    Simple k-means clustering returning cluster label per row.

    Uses multiple random inits and returns the best (lowest inertia) result.

    Args:
        embeddings: (N, D) embedding matrix.
        k: Number of clusters.
        n_init: Number of random restarts.

    Returns:
        (N,) integer array of cluster labels.
    """
    N = embeddings.shape[0]
    k = min(k, N)
    if k <= 1:
        return np.zeros(N, dtype=int)

    best_labels = None
    best_inertia = float("inf")
    rng = np.random.default_rng(42)

    for _ in range(n_init):
        # Random initialization
        centers = embeddings[rng.choice(N, k, replace=False)]

        for _ in range(100):  # max iterations
            # Assignment
            dists = np.linalg.norm(
                embeddings[:, None, :] - centers[None, :, :], axis=2
            )
            labels = np.argmin(dists, axis=1)

            # Update centers
            new_centers = np.array([
                embeddings[labels == c].mean(axis=0) if (labels == c).any() else centers[c]
                for c in range(k)
            ])

            if np.allclose(centers, new_centers, atol=1e-6):
                break
            centers = new_centers

        # Inertia
        inertia = sum(
            np.linalg.norm(embeddings[labels == c] - centers[c]) ** 2
            for c in range(k) if (labels == c).any()
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels


def cluster_nodes(
    joint_embeddings: np.ndarray,
    S: np.ndarray,
    k_nn: int = 5,
    max_clusters: int = 20,
) -> List[List[int]]:
    """
    Full clustering pipeline from Appendix C.2.

    Returns list of clusters; singleton clusters are discarded (they are not
    merge candidates).

    Args:
        joint_embeddings: (N, D) joint name+desc embeddings.
        S: (N, N) combined similarity matrix.
        k_nn: k for k-NN graph.
        max_clusters: Maximum cluster count.

    Returns:
        List of clusters (each cluster is a list of node indices).
        Only multi-node clusters are returned.
    """
    N = joint_embeddings.shape[0]
    if N <= 1:
        return []

    A = build_knn_graph(S, k=k_nn)
    k_star = eigengap_cluster_count(A, max_clusters=max_clusters)
    labels = kmeans_cluster(joint_embeddings, k=k_star)

    # Group indices by cluster label
    clusters: List[List[int]] = [[] for _ in range(k_star)]
    for i, label in enumerate(labels):
        clusters[label].append(i)

    # Return only multi-node clusters (these are merge candidates)
    return [c for c in clusters if len(c) > 1]
