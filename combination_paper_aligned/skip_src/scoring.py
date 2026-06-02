"""Per-frame importance score and threshold helpers."""

from __future__ import annotations

import numpy as np


def delta_map_proxy(
    feat_t: np.ndarray,
    context_feats: list[np.ndarray],
) -> float:
    if not context_feats:
        return 1.0
    sims = [float(np.dot(feat_t, e)) for e in context_feats]
    return float(1.0 - max(sims))


def novelty_score(
    feat_t: np.ndarray,
    cluster_centers: np.ndarray,
    power: float = 1.0,
) -> float:
    if cluster_centers is None or len(cluster_centers) == 0:
        return 1.0
    dists = np.linalg.norm(cluster_centers - feat_t, axis=1)
    d = float(np.min(dists))
    val = min(1.0, d / 1.414)
    return val**power


def frame_score(
    feat_t: np.ndarray,
    context_feats: list[np.ndarray],
    cluster_centers: np.ndarray,
    class_weight: float,
    novelty_power: float = 1.0,
) -> float:
    dm = delta_map_proxy(feat_t, context_feats)
    nov = novelty_score(feat_t, cluster_centers, novelty_power)
    return dm * nov * class_weight


def calibrate_tau(scores: list[float], target_retention: float) -> float:
    if not scores:
        return 0.0
    target_retention = float(np.clip(target_retention, 0.01, 0.99))
    return float(np.quantile(scores, 1.0 - target_retention))


def score_distribution_stats(scores: np.ndarray) -> dict:
    s = np.asarray(scores, dtype=np.float64)
    if len(s) == 0:
        return {}
    return {
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(np.median(s)),
        "p90": float(np.quantile(s, 0.9)),
        "frac_below_1e-4": float(np.mean(s < 1e-4)),
        "frac_above_0.01": float(np.mean(s > 0.01)),
    }


def estimate_adaptive_tau(
    scores: np.ndarray,
    mad_multiplier: float = 1.2,
    min_tau_quantile: float = 0.45,
    flat_quantile: float = 0.68,
) -> float:
    if len(scores) == 0:
        return 0.0
    s = np.asarray(scores, dtype=np.float64)
    scale = float(np.percentile(s, 95)) + 1e-12
    if scale < 1e-4:
        return float(np.quantile(s, flat_quantile))

    sn = s / scale
    med = float(np.median(sn))
    mad = float(np.median(np.abs(sn - med))) + 1e-12
    tau_n = med + mad_multiplier * mad
    if mad < 0.02:
        tau_n = float(np.quantile(sn, flat_quantile))
    floor_n = float(np.quantile(sn, min_tau_quantile)) * 0.5
    return max(tau_n, floor_n) * scale
