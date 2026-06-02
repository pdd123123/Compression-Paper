"""Post-processing to reduce blur/blockiness after neural decode."""

from __future__ import annotations

import cv2
import numpy as np


def post_refine(rgb: np.ndarray, soft_rgb: np.ndarray) -> np.ndarray:
    """
    Blend network output with edge-colored guide and mild smoothing.
    """
    base = soft_rgb.astype(np.float32)
    pred = rgb.astype(np.float32)
    gray = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge_w = np.clip(gray * 2.5, 0, 1)[..., None]
    blended = pred * (1.0 - 0.35 * edge_w) + base * (0.35 * edge_w)
    out = cv2.bilateralFilter(blended.astype(np.uint8), d=5, sigmaColor=25, sigmaSpace=25)
    return np.clip(out, 0, 255).astype(np.uint8)


def light_sharpen(rgb: np.ndarray, soft_rgb: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Edge-aware unsharp mask (no heavy bilateral; avoids old post_refine haze)."""
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return rgb
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    sharp = cv2.addWeighted(rgb, 1.0 + s, blur, -s, 0)
    gray = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edge_w = np.clip(gray * 3.0, 0, 1)[..., None]
    out = rgb.astype(np.float32) * (1.0 - 0.5 * edge_w) + sharp.astype(np.float32) * (0.5 * edge_w)
    return np.clip(out, 0, 255).astype(np.uint8)
