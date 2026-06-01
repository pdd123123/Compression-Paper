"""Greedy diverse top-k frame selection."""

from __future__ import annotations

import numpy as np


def select_diverse_topk(
    scores: np.ndarray,
    n_keep: int,
    feats: np.ndarray | None = None,
    min_temporal_gap: int = 4,
    sim_penalty: float = 0.45,
) -> list[int]:
    n = len(scores)
    if n == 0:
        return []
    n_keep = max(1, min(int(n_keep), n))
    if n_keep >= n:
        return list(range(n))

    # Keeping a large fraction: full diverse greedy is O(n_keep * n * n_keep); use fast path.
    if n_keep > 800 or n_keep > n * 0.25:
        return _select_topk_with_gap(scores, n_keep, min_temporal_gap)

    selected: list[int] = []
    selected_set: set[int] = set()
    gap = max(1, int(min_temporal_gap))

    while len(selected) < n_keep:
        best_i: int | None = None
        best_val = -1e18
        for i in range(n):
            if i in selected_set:
                continue
            if selected and min(abs(i - s) for s in selected) < gap:
                continue
            val = float(scores[i])
            if feats is not None and selected:
                sims = [float(np.dot(feats[i], feats[s])) for s in selected]
                val -= sim_penalty * max(sims)
            if val > best_val:
                best_val = val
                best_i = i
        if best_i is None:
            gap = max(1, gap - 1)
            if gap < 1:
                break
            continue
        selected.append(best_i)
        selected_set.add(best_i)

    if len(selected) < n_keep:
        for i in np.argsort(-scores):
            ii = int(i)
            if ii not in selected_set:
                selected.append(ii)
                selected_set.add(ii)
            if len(selected) >= n_keep:
                break
    return sorted(selected)


def _select_topk_with_gap(
    scores: np.ndarray, n_keep: int, min_temporal_gap: int
) -> list[int]:
    """Score-sorted picks with a minimum index gap (linear-ish, for high retention)."""
    gap = max(1, int(min_temporal_gap))
    order = np.argsort(-scores)
    selected: list[int] = []
    selected_set: set[int] = set()
    for ii in order:
        ii = int(ii)
        if selected and min(abs(ii - s) for s in selected) < gap:
            continue
        selected.append(ii)
        selected_set.add(ii)
        if len(selected) >= n_keep:
            break
    if len(selected) < n_keep:
        for ii in order:
            ii = int(ii)
            if ii not in selected_set:
                selected.append(ii)
                selected_set.add(ii)
            if len(selected) >= n_keep:
                break
    return sorted(selected)
