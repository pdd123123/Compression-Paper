"""Context set E* from recent frame embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def build_uniform_context(
    history_feats: list[np.ndarray],
    context_size: int,
) -> tuple[list[int], np.ndarray]:
    """Evenly spaced history frames (ORIC-style static context, no clustering)."""
    n = len(history_feats)
    if n == 0:
        return [], np.zeros((0, 1), dtype=np.float32)
    k = min(context_size, n)
    X = np.stack(history_feats, axis=0)
    if k >= n:
        idx = list(range(n))
        return idx, X.astype(np.float32)
    step = max(1, (n - 1) // max(k - 1, 1))
    idx = [min(n - 1, i * step) for i in range(k)]
    return idx, X[idx].astype(np.float32)


def build_diverse_context(
    history_feats: list[np.ndarray],
    context_size: int,
    min_samples: int = 3,
) -> tuple[list[int], np.ndarray]:
    n = len(history_feats)
    if n == 0:
        return [], np.zeros((0, 1), dtype=np.float32)

    X = np.stack(history_feats, axis=0)
    k = min(context_size, n)
    if n < min_samples or k <= 1:
        idx = _farthest_point_indices(X, k)
        centers = X[idx]
        return idx, centers

    km = KMeans(n_clusters=k, n_init=1, max_iter=30, random_state=0)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.astype(np.float32)

    reps: list[int] = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        sub = X[members]
        d = np.linalg.norm(sub - centers[c], axis=1)
        reps.append(int(members[int(np.argmin(d))]))
    return reps, centers


def build_context(
    history_feats: list[np.ndarray],
    context_size: int,
    min_samples: int = 3,
    mode: str = "diverse",
) -> tuple[list[int], np.ndarray]:
    if mode == "uniform":
        return build_uniform_context(history_feats, context_size)
    return build_diverse_context(history_feats, context_size, min_samples)


def _farthest_point_indices(X: np.ndarray, k: int) -> list[int]:
    k = min(k, len(X))
    chosen = [0]
    while len(chosen) < k:
        dists = np.min(
            [np.linalg.norm(X - X[i], axis=1) for i in chosen],
            axis=0,
        )
        nxt = int(np.argmax(dists))
        if nxt in chosen:
            break
        chosen.append(nxt)
    return chosen
