"""Baseline samplers: uniform, non-uniform, content-aware, ORIC-proxy."""

from __future__ import annotations

import cv2
import numpy as np

from .context import build_diverse_context
from .embeddings import FrameEmbedder
from .scoring import delta_map_proxy


def uniform_indices(n_frames: int, keep_every: int) -> list[int]:
    """Fixed-rate: keep 1 of every k frames."""
    k = max(1, keep_every)
    return list(range(0, n_frames, k))


def motion_scores(frames: list[np.ndarray]) -> np.ndarray:
    scores = np.zeros(len(frames), dtype=np.float32)
    prev = None
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            scores[i] = float(np.mean(cv2.absdiff(gray, prev)))
        prev = gray
    return scores


def entropy_scores(frames: list[np.ndarray]) -> np.ndarray:
    scores = np.zeros(len(frames), dtype=np.float32)
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        p = hist.flatten() / (hist.sum() + 1e-8)
        p = p[p > 0]
        scores[i] = float(-np.sum(p * np.log2(p + 1e-12)))
    return scores


def edge_density_scores(frames: list[np.ndarray]) -> np.ndarray:
    scores = np.zeros(len(frames), dtype=np.float32)
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        scores[i] = float(edges.mean() / 255.0)
    return scores


def percentile_keep_indices(scores: np.ndarray, retention_ratio: float) -> list[int]:
    n = len(scores)
    if n == 0:
        return []
    k = max(1, int(round(n * retention_ratio)))
    order = np.argsort(-scores)
    chosen = sorted(order[:k].tolist())
    return chosen


def non_uniform_indices(
    frames: list[np.ndarray],
    retention_ratio: float,
    motion_weight: float = 0.6,
) -> list[int]:
    m = motion_scores(frames)
    e = entropy_scores(frames)
    m = m / (m.max() + 1e-8)
    e = e / (e.max() + 1e-8)
    combined = motion_weight * m + (1 - motion_weight) * e
    return percentile_keep_indices(combined, retention_ratio)


def content_aware_indices(
    frames: list[np.ndarray],
    retention_ratio: float,
) -> list[int]:
    s = edge_density_scores(frames)
    return percentile_keep_indices(s, retention_ratio)


def oric_proxy_indices(
    frames: list[np.ndarray],
    retention_ratio: float,
    history_len: int = 30,
    context_size: int = 5,
    embedder: FrameEmbedder | None = None,
) -> list[int]:
    """
    ORIC-style utility without dual detectors:
    reward ≈ ΔmAP with *uniform* context (paper ORIC limitation).
    """
    emb = embedder or FrameEmbedder()
    n = len(frames)
    rewards = np.zeros(n, dtype=np.float32)
    history: list[np.ndarray] = []

    for i, f in enumerate(frames):
        feat = emb.encode(f)
        # uniform context: evenly spaced past frames
        if len(history) >= 3:
            step = max(1, len(history) // context_size)
            uniform_ctx = history[::step][-context_size:]
        else:
            uniform_ctx = history
        rewards[i] = delta_map_proxy(feat, uniform_ctx)
        history.append(feat)
        if len(history) > history_len:
            history = history[-history_len:]

    return percentile_keep_indices(rewards, retention_ratio)


def semantic_uniform_context_indices(
    frames: list[np.ndarray],
    retention_ratio: float,
    history_len: int = 30,
    context_size: int = 5,
    embedder: FrameEmbedder | None = None,
) -> list[int]:
    """Ablation: clustered E* but no class weights (embedding only)."""
    emb = embedder or FrameEmbedder()
    rewards = []
    history: list[np.ndarray] = []
    for f in frames:
        feat = emb.encode(f)
        rep_idx, centers = build_diverse_context(history, context_size)
        ctx = [history[j] for j in rep_idx if j < len(history)]
        dm = delta_map_proxy(feat, ctx)
        if len(centers):
            d = float(np.min(np.linalg.norm(centers - feat, axis=1)))
            nov = min(1.0, d / 1.414)
        else:
            nov = 1.0
        rewards.append(dm * nov)
        history.append(feat)
        if len(history) > history_len:
            history = history[-history_len:]
    return percentile_keep_indices(np.array(rewards), retention_ratio)
