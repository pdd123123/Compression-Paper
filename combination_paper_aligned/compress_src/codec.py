"""Sparse soft-edge bitstream I/O (.seccomp) with temporal delta coding."""

from __future__ import annotations

# Placeholder between JPEG keyframes (hybrid_keys_only mode; no edge data stored).
HYBRID_GAP_PACKET = b"E"

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np


def pack_frame(labels: np.ndarray, prev_labels: np.ndarray | None = None) -> bytes:
    """
    Pack edge pixels. If prev_labels given, encode only changed pixels (xor mask).
    Format per pixel: uint16 y, uint16 x, uint8 label (0 = removed, 1..K = cluster+1)
    """
    if prev_labels is None:
        diff = labels > 0
    else:
        diff = labels != prev_labels

    ys, xs = np.where(diff)
    if ys.size == 0:
        return b""
    vals = labels[ys, xs].astype(np.uint8)
    arr = np.column_stack([ys.astype(np.uint16), xs.astype(np.uint16), vals])
    return arr.tobytes()


def unpack_frame(
    blob: bytes,
    height: int,
    width: int,
    prev_labels: np.ndarray | None = None,
) -> np.ndarray:
    if prev_labels is None:
        labels = np.zeros((height, width), dtype=np.uint8)
    else:
        labels = prev_labels.copy()

    if not blob:
        return labels

    arr = np.frombuffer(blob, dtype=np.uint16).reshape(-1, 3)
    ys, xs, vals = arr[:, 0], arr[:, 1], arr[:, 2]
    if prev_labels is None:
        labels[ys, xs] = vals.astype(np.uint8)
    else:
        labels[ys, xs] = vals.astype(np.uint8)
    return labels


def save_bitstream(
    path: str | Path,
    meta: dict[str, Any],
    centroids: np.ndarray,
    frame_blobs: list[bytes],
    keyframe_interval: int = 30,
    zlib_level: int = 3,
    video_annex: bytes | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "meta": meta,
        "centroids": centroids.astype(np.float16).tolist(),
        "n_frames": len(frame_blobs),
        "keyframe_interval": keyframe_interval,
        "version": meta.get("version", 2),
    }
    if "rgb_keyframe_interval" in meta:
        header["rgb_keyframe_interval"] = meta["rgb_keyframe_interval"]
    if video_annex:
        header["video_annex_bytes"] = len(video_annex)
        meta["playback"] = meta.get("playback", "h264_annex")
    payload = json.dumps(header).encode("utf-8") + b"\n---\n"
    zlvl = int(meta.get("zlib_level", zlib_level))
    zlvl = max(0, min(9, zlvl))
    compressed_frames = [zlib.compress(b, level=zlvl) if b else b"" for b in frame_blobs]
    frame_bytes = b"".join(compressed_frames)
    arrays = {
        "header_len": np.array([len(payload)], dtype=np.int64),
        "frame_lens": np.array([len(c) for c in compressed_frames], dtype=np.int64),
    }
    if video_annex:
        arrays["video_annex"] = np.frombuffer(video_annex, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(payload)
        np.savez(f, **arrays, frames=np.frombuffer(frame_bytes, dtype=np.uint8))


def load_bitstream(
    path: str | Path,
) -> tuple[dict, np.ndarray, list[bytes], bytes | None]:
    path = Path(path)
    with open(path, "rb") as f:
        raw = f.read()

    sep = raw.index(b"\n---\n")
    header = json.loads(raw[:sep].decode("utf-8"))
    rest = raw[sep + 5 :]

    import io

    data = np.load(io.BytesIO(rest), allow_pickle=False)
    frame_lens = data["frame_lens"]
    blob = bytes(data["frames"].tobytes())
    offset = 0
    frame_blobs: list[bytes] = []
    for length in frame_lens:
        ln = int(length)
        chunk = blob[offset : offset + ln]
        offset += ln
        frame_blobs.append(zlib.decompress(chunk) if chunk else b"")

    centroids = np.array(header["centroids"], dtype=np.float32)
    meta = dict(header["meta"])
    if "rgb_keyframe_interval" in header:
        meta["rgb_keyframe_interval"] = header["rgb_keyframe_interval"]
    video_annex = None
    if "video_annex" in data.files:
        video_annex = bytes(data["video_annex"].tobytes())
    if video_annex:
        meta["playback"] = meta.get("playback", "h264_annex")
    return meta, centroids, frame_blobs, video_annex


def labels_to_soft_rgb(labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    soft = np.zeros((h, w, 3), dtype=np.uint8)
    edge = labels > 0
    if edge.any():
        soft[edge] = np.clip(centroids[labels[edge] - 1], 0, 255).astype(np.uint8)
    return soft


def pack_frame_packet(
    labels: np.ndarray | None,
    prev_labels: np.ndarray | None,
    rgb_bgr: np.ndarray | None = None,
    jpeg_quality: int = 82,
) -> bytes:
    """b'E' + edge pack, or b'K' + zlib(jpeg) for full-color anchor frames."""
    if rgb_bgr is not None:
        import cv2

        ok, buf = cv2.imencode(
            ".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        )
        if not ok:
            raise RuntimeError("JPEG keyframe encode failed")
        return b"K" + zlib.compress(buf.tobytes(), 3)
    if labels is None:
        raise ValueError("labels required for edge-only packet")
    return b"E" + pack_frame(labels, prev_labels)


def unpack_frame_packet(
    blob: bytes,
    height: int,
    width: int,
    prev_labels: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns (labels, rgb_bgr). Exactly one is set.
    labels=None on keyframe packets; rgb_bgr=None on edge packets.
    """
    if not blob:
        labels = unpack_frame(b"", height, width, prev_labels)
        return labels, None
    tag = blob[:1]
    body = blob[1:]
    if tag == b"K":
        import cv2

        jpg = zlib.decompress(body)
        bgr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("JPEG keyframe decode failed")
        if bgr.shape[0] != height or bgr.shape[1] != width:
            bgr = cv2.resize(bgr, (width, height))
        return None, bgr
    if tag == b"E":
        labels = unpack_frame(body, height, width, prev_labels)
        return labels, None
    raise ValueError(f"Unknown packet tag: {tag!r}")


def decode_all_labels(
    frame_blobs: list[bytes],
    height: int,
    width: int,
    keyframe_interval: int = 30,
) -> list[np.ndarray]:
    """Reconstruct label maps from delta-coded bitstream."""
    labels_list: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for i, blob in enumerate(frame_blobs):
        if i % keyframe_interval == 0:
            prev = None
        labels = unpack_frame(blob, height, width, prev)
        labels_list.append(labels)
        prev = labels
    return labels_list
