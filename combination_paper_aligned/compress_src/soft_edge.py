"""Canny edge detection + K-means quantization (soft edge maps)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def canny_mask(
    frame_bgr: np.ndarray,
    low: int = 50,
    high: int = 150,
) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return (edges > 0).astype(np.uint8)


def quantize_edge_colors(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    n_clusters: int,
    centroids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        labels: (H, W) uint8, 0 = non-edge, 1..K = cluster id (1-based)
        centroids: (K, 3) RGB float32 in [0, 255]
        soft_rgb: (H, W, 3) uint8 visualization / model input
    """
    h, w = mask.shape
    edge_idx = np.flatnonzero(mask.ravel())
    if edge_idx.size == 0:
        centroids = np.zeros((n_clusters, 3), dtype=np.float32)
        labels = np.zeros((h, w), dtype=np.uint8)
        soft = np.zeros((h, w, 3), dtype=np.uint8)
        return labels, centroids, soft

    rgb = cv2_bgr_to_rgb(frame_bgr)
    pixels = rgb.reshape(-1, 3)[edge_idx].astype(np.float32)

    if centroids is None:
        k = min(n_clusters, max(1, pixels.shape[0]))
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=0)
        km.fit(pixels)
        centroids = km.cluster_centers_.astype(np.float32)
        cluster_ids = km.predict(pixels)
    else:
        k = centroids.shape[0]
        diff = pixels[:, None, :] - centroids[None, :, :]
        cluster_ids = np.argmin(np.sum(diff * diff, axis=2), axis=1)

    labels = np.zeros(h * w, dtype=np.uint8)
    labels[edge_idx] = (cluster_ids + 1).astype(np.uint8)
    labels = labels.reshape(h, w)

    soft = np.zeros((h, w, 3), dtype=np.uint8)
    edge = labels > 0
    if edge.any():
        soft[edge] = np.clip(centroids[labels[edge] - 1], 0, 255).astype(np.uint8)

    return labels, centroids, soft


def cv2_bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def fit_global_palette(
    frames_bgr: list[np.ndarray],
    n_clusters: int,
    canny_low: int,
    canny_high: int,
    max_pixels: int = 200_000,
) -> np.ndarray:
    """K-means on pooled edge pixels across frames."""
    import cv2

    rng = np.random.default_rng(0)
    samples: list[np.ndarray] = []
    for frame in frames_bgr:
        mask = canny_mask(frame, canny_low, canny_high)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pix = rgb[mask > 0]
        if pix.size:
            samples.append(pix.astype(np.float32))
    if not samples:
        return np.zeros((n_clusters, 3), dtype=np.float32)

    pool = np.concatenate(samples, axis=0)
    if pool.shape[0] > max_pixels:
        idx = rng.choice(pool.shape[0], max_pixels, replace=False)
        pool = pool[idx]

    k = min(n_clusters, pool.shape[0])
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=0)
    km.fit(pool)
    out = np.zeros((n_clusters, 3), dtype=np.float32)
    out[: k] = km.cluster_centers_
    return out


def frame_to_soft_edge(
    frame_bgr: np.ndarray,
    n_clusters: int,
    canny_low: int,
    canny_high: int,
    centroids: np.ndarray | None = None,
    edge_dilate: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import cv2

    mask = canny_mask(frame_bgr, canny_low, canny_high)
    if edge_dilate > 0:
        k = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, k, iterations=int(edge_dilate))
    return quantize_edge_colors(frame_bgr, mask, n_clusters, centroids)
