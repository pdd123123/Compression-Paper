"""Video read helpers shared by pipeline and sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


def resize_frame_max_height(frame_bgr: np.ndarray, max_height: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if h <= max_height:
        return frame_bgr
    scale = max_height / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(frame_bgr, (new_w, max_height), interpolation=cv2.INTER_AREA)


def iter_video_frames(
    path: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    stride = max(1, int(stride))
    idx = 0
    kept = 0
    while True:
        if stride > 1 and idx % stride != 0:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        yield kept, frame
        kept += 1
        if max_frames is not None and kept >= max_frames:
            break
        idx += 1
    cap.release()


def get_video_meta(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta
