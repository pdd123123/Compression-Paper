"""PSNR / SSIM and file-size helpers."""

from __future__ import annotations

import numpy as np


def psnr(img_a: np.ndarray, img_b: np.ndarray, max_val: float = 255.0) -> float:
    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(10.0 * np.log10((max_val**2) / mse))


def ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity

        if img_a.ndim == 3:
            return float(
                structural_similarity(
                    img_a, img_b, channel_axis=2, data_range=255
                )
            )
        return float(structural_similarity(img_a, img_b, data_range=255))
    except ImportError:
        return _ssim_simple(img_a, img_b)


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den else 1.0


def transmission_time_sec(size_bytes: float, bitrate_mbps: float = 10.0) -> float:
    """Paper uses 10 Mbps channel."""
    bits = size_bytes * 8
    return bits / (bitrate_mbps * 1e6)
