"""End-to-end compress / decompress helpers."""

from __future__ import annotations

import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .codec import (
    HYBRID_GAP_PACKET,
    decode_all_labels,
    labels_to_soft_rgb,
    load_bitstream,
    pack_frame,
    pack_frame_packet,
    save_bitstream,
    unpack_frame_packet,
)
from .metrics import psnr, ssim, transmission_time_sec
from .models.soft_edge_net import SoftEdgeReconstructor
from .pretrained_enhance import enhance_frame, lama_inpaint
from .refine import light_sharpen, post_refine
from .soft_edge import fit_global_palette, frame_to_soft_edge


def iter_video_frames(
    path: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield kept, frame
            kept += 1
            if max_frames is not None and kept >= max_frames:
                break
        idx += 1
    cap.release()


def resize_frame(frame: np.ndarray, max_height: int) -> np.ndarray:
    if max_height <= 0:
        return frame
    h, w = frame.shape[:2]
    if h <= max_height:
        return frame
    scale = max_height / h
    return cv2.resize(frame, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)


def _edge_compute_frame(frame: np.ndarray, edge_max_height: int) -> tuple[np.ndarray, bool]:
    """Run Canny/K-means on a smaller frame when edge_max_height is set."""
    if edge_max_height <= 0 or frame.shape[0] <= edge_max_height:
        return frame, False
    h, w = frame.shape[:2]
    scale = edge_max_height / h
    small = cv2.resize(
        frame,
        (max(1, int(w * scale)), edge_max_height),
        interpolation=cv2.INTER_AREA,
    )
    return small, True


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


def _norm_video_fps(fps: float) -> float:
    if 23.9 <= fps <= 24.1:
        return 24000.0 / 1001.0
    return fps if fps and fps > 1 else 24.0


class _FfmpegVideoWriter:
    """Stream BGR frames to libx264 (smooth CFR playback; avoids OpenCV mp4v judder)."""

    def __init__(self, path: str | Path, width: int, height: int, fps: float, crf: int = 20) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fps = _norm_video_fps(fps)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "cfr",
            "-r",
            f"{fps:.6f}",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, bgr: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin closed")
        self._proc.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def release(self) -> None:
        if self._proc.stdin:
            self._proc.stdin.close()
        code = self._proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg encode failed (exit {code}) for {self.path}")


def _encode_h264_annex(
    input_path: str | Path,
    cfg: dict,
    meta_v: dict,
    max_frames: int | None,
) -> bytes:
    """H.264 MP4 bytes stored in .seccomp (same preset as upload baseline, no faststart = slightly smaller)."""
    from .video_export import encode_h264_mp4

    delivery = cfg.get("delivery", {})
    transmit = cfg.get("transmit", {})
    max_h = int(
        delivery.get("h264_max_height", cfg.get("processing", {}).get("max_height", 720)) or 720
    )
    base_crf = int(delivery.get("h264_crf", 28))
    extra = int(transmit.get("extra_crf", 0) or 0)
    crf = int(transmit.get("h264_crf", base_crf + extra))
    fps = _norm_video_fps(float(meta_v.get("fps") or 24.0))
    preset = str(delivery.get("h264_preset", "fast"))
    return encode_h264_mp4(
        input_path,
        None,
        max_frames or 0,
        max_height=max_h,
        crf=crf,
        fps=fps,
        preset=preset,
        faststart=False,
    )


def _maybe_h264_annex(
    input_path: str | Path,
    cfg: dict,
    meta_v: dict,
    max_frames: int | None,
) -> bytes | None:
    if not cfg.get("delivery", {}).get("h264_annex"):
        return None
    return _encode_h264_annex(input_path, cfg, meta_v, max_frames)


def compress_video(
    input_path: str | Path,
    output_path: str | Path,
    cfg: dict,
) -> dict:
    se = cfg["soft_edge"]
    vm = cfg["video"]
    max_frames = vm.get("max_frames")
    stride = vm.get("frame_stride", 1)

    meta_v = get_video_meta(input_path)
    fps = meta_v["fps"]
    if not fps or fps <= 1 or fps > 120:
        fps = 24.0
    meta_v["fps"] = fps
    max_h = cfg.get("processing", {}).get("max_height", 0) or 0
    rgb_only = bool(se.get("rgb_only", False))
    key_int = se.get("keyframe_interval", 30)
    rgb_kf = int(se.get("rgb_keyframe_interval", 0) or 0)
    jpeg_q = int(se.get("rgb_jpeg_quality", 82))
    hybrid_keys_only = bool(se.get("hybrid_keys_only", False))
    if rgb_only:
        rgb_kf = max(1, rgb_kf or 1)
    if hybrid_keys_only:
        rgb_kf = max(2, int(se.get("rgb_keyframe_interval", 5) or 5))
    zlib_level = int(cfg.get("codec", {}).get("zlib_level", 3))

    if cfg.get("transmit", {}).get("h264_only") or cfg.get("delivery", {}).get("annex_only"):
        n = 0
        out_h, out_w = meta_v["height"], meta_v["width"]
        for _, frame in iter_video_frames(input_path, max_frames=max_frames, stride=stride):
            if n == 0:
                frame = resize_frame(frame, max_h)
                out_h, out_w = frame.shape[:2]
            n += 1
        annex = _encode_h264_annex(input_path, cfg, meta_v, max_frames)
        if not annex:
            raise RuntimeError("h264_only transmit requires delivery.h264_annex settings")
        meta = {
            "width": out_w,
            "height": out_h,
            "fps": float(meta_v["fps"]),
            "n_frames": n,
            "source": str(Path(input_path).name),
            "version": 3,
            "playback": "h264_annex",
            "transmit_h264_only": True,
            "zlib_level": zlib_level,
        }
        centroids = np.zeros((1, 3), dtype=np.float32)
        save_bitstream(output_path, meta, centroids, [], zlib_level=zlib_level, video_annex=annex)
        full_size = Path(input_path).stat().st_size
        total_frames = meta_v["frame_count"] or n
        orig_size = int(full_size * (n / total_frames)) if max_frames and total_frames > 0 and n < total_frames else full_size
        comp_size = Path(output_path).stat().st_size
        saving = (1 - comp_size / orig_size) * 100 if orig_size else 0
        return {
            "original_mb": orig_size / 1024 / 1024,
            "compressed_mb": comp_size / 1024 / 1024,
            "bandwidth_saving_pct": saving,
            "tx_orig_s": transmission_time_sec(orig_size),
            "tx_comp_s": transmission_time_sec(comp_size),
            "n_frames": n,
            "bitstream": str(output_path),
            "h264_annex_mb": len(annex) / 1024 / 1024,
        }

    if hybrid_keys_only:
        blobs: list[bytes] = []
        n = 0
        out_h, out_w = meta_v["height"], meta_v["width"]
        centroids = np.zeros((max(1, int(se.get("num_clusters", 8))), 3), dtype=np.float32)
        for _, frame in tqdm(
            iter_video_frames(input_path, max_frames=max_frames, stride=stride),
            desc="compress",
        ):
            frame = resize_frame(frame, max_h)
            if n == 0:
                out_h, out_w = frame.shape[:2]
            if n % rgb_kf == 0:
                blobs.append(pack_frame_packet(None, None, frame, jpeg_q))
            else:
                blobs.append(HYBRID_GAP_PACKET)
            n += 1
        meta = {
            "width": out_w,
            "height": out_h,
            "storage_downscale": 1,
            "fps": float(meta_v["fps"]),
            "n_frames": n,
            "num_clusters": se.get("num_clusters", 8),
            "global_palette": False,
            "source": str(Path(input_path).name),
            "version": 3,
            "rgb_keyframe_interval": rgb_kf,
            "rgb_jpeg_quality": jpeg_q,
            "hybrid_keys_only": True,
            "zlib_level": zlib_level,
        }
        annex = _maybe_h264_annex(input_path, cfg, meta_v, max_frames)
        if annex:
            meta["playback"] = "h264_annex"
        save_bitstream(
            output_path, meta, centroids, blobs, zlib_level=zlib_level, video_annex=annex
        )
        full_size = Path(input_path).stat().st_size
        total_frames = meta_v["frame_count"] or n
        if max_frames is not None and total_frames > 0 and n < total_frames:
            orig_size = int(full_size * (n / total_frames))
        else:
            orig_size = full_size
        comp_size = Path(output_path).stat().st_size
        saving = (1 - comp_size / orig_size) * 100 if orig_size else 0
        return {
            "original_mb": orig_size / 1024 / 1024,
            "compressed_mb": comp_size / 1024 / 1024,
            "bandwidth_saving_pct": saving,
            "tx_orig_s": transmission_time_sec(orig_size),
            "tx_comp_s": transmission_time_sec(comp_size),
            "n_frames": n,
            "bitstream": str(output_path),
            "h264_annex_mb": (len(annex) / 1024 / 1024) if annex else 0,
        }

    if rgb_only:
        blobs: list[bytes] = []
        n = 0
        out_h, out_w = meta_v["height"], meta_v["width"]
        centroids = np.zeros((max(1, int(se.get("num_clusters", 8))), 3), dtype=np.float32)
        for _, frame in tqdm(
            iter_video_frames(input_path, max_frames=max_frames, stride=stride),
            desc="compress",
        ):
            frame = resize_frame(frame, max_h)
            if n == 0:
                out_h, out_w = frame.shape[:2]
            blobs.append(pack_frame_packet(None, None, frame, jpeg_q))
            n += 1
        meta = {
            "width": out_w,
            "height": out_h,
            "storage_downscale": 1,
            "fps": float(meta_v["fps"]),
            "n_frames": n,
            "num_clusters": se.get("num_clusters", 8),
            "global_palette": False,
            "source": str(Path(input_path).name),
            "version": 3,
            "rgb_keyframe_interval": 1,
            "rgb_jpeg_quality": jpeg_q,
            "rgb_only": True,
            "zlib_level": zlib_level,
        }
        annex = _maybe_h264_annex(input_path, cfg, meta_v, max_frames)
        if annex:
            meta["playback"] = "h264_annex"
        save_bitstream(
            output_path, meta, centroids, blobs, zlib_level=zlib_level, video_annex=annex
        )
        full_size = Path(input_path).stat().st_size
        total_frames = meta_v["frame_count"] or n
        if max_frames is not None and total_frames > 0 and n < total_frames:
            orig_size = int(full_size * (n / total_frames))
        else:
            orig_size = full_size
        comp_size = Path(output_path).stat().st_size
        saving = (1 - comp_size / orig_size) * 100 if orig_size else 0
        return {
            "original_mb": orig_size / 1024 / 1024,
            "compressed_mb": comp_size / 1024 / 1024,
            "bandwidth_saving_pct": saving,
            "tx_orig_s": transmission_time_sec(orig_size),
            "tx_comp_s": transmission_time_sec(comp_size),
            "n_frames": n,
            "bitstream": str(output_path),
            "h264_annex_mb": (len(annex) / 1024 / 1024) if annex else 0,
        }

    frames_for_palette: list[np.ndarray] = []
    if se.get("global_palette", True):
        for _, f in iter_video_frames(input_path, max_frames=min(50, max_frames or 50), stride=stride):
            frames_for_palette.append(resize_frame(f, max_h))
        centroids = fit_global_palette(
            frames_for_palette,
            se["num_clusters"],
            se["canny_low"],
            se["canny_high"],
        )
    else:
        centroids = None

    skip_edge_on_key = bool(se.get("skip_edge_on_rgb_key", True))
    edge_max_h = int(cfg.get("processing", {}).get("edge_max_height", 0) or 0)
    blobs: list[bytes] = []
    prev_labels: np.ndarray | None = None
    n = 0
    out_h, out_w = meta_v["height"], meta_v["width"]
    for _, frame in tqdm(
        iter_video_frames(input_path, max_frames=max_frames, stride=stride),
        desc="compress",
    ):
        frame = resize_frame(frame, max_h)
        if n == 0:
            out_h, out_w = frame.shape[:2]

        if rgb_kf > 0 and n % rgb_kf == 0 and skip_edge_on_key:
            blobs.append(pack_frame_packet(None, None, frame, jpeg_q))
            prev_labels = None
            n += 1
            continue

        edge_frame, upsample_labels = _edge_compute_frame(frame, edge_max_h)
        if not se.get("global_palette", True):
            _, centroids, _ = frame_to_soft_edge(
                edge_frame,
                se["num_clusters"],
                se["canny_low"],
                se["canny_high"],
                None,
                se.get("edge_dilate", 0),
            )
        labels, centroids, _ = frame_to_soft_edge(
            edge_frame,
            se["num_clusters"],
            se["canny_low"],
            se["canny_high"],
            centroids,
            se.get("edge_dilate", 0),
        )
        if upsample_labels:
            labels = cv2.resize(
                labels,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        down = se.get("storage_downscale", 1)
        if down > 1:
            lh, lw = labels.shape
            labels_store = cv2.resize(
                labels,
                (lw // down, lh // down),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            labels_store = labels

        if rgb_kf > 0 and n % rgb_kf == 0:
            blobs.append(pack_frame_packet(None, None, frame, jpeg_q))
            prev_labels = None
        else:
            if n % key_int == 0:
                prev_labels = None
            if rgb_kf > 0:
                blobs.append(pack_frame_packet(labels_store, prev_labels))
            else:
                blobs.append(pack_frame(labels_store, prev_labels))
            prev_labels = labels_store
        n += 1

    meta = {
        "width": out_w,
        "height": out_h,
        "storage_downscale": se.get("storage_downscale", 1),
        "fps": float(meta_v["fps"]),
        "n_frames": n,
        "canny_low": se["canny_low"],
        "canny_high": se["canny_high"],
        "num_clusters": se["num_clusters"],
        "global_palette": se.get("global_palette", True),
        "source": str(Path(input_path).name),
        "version": 3 if rgb_kf > 0 else 2,
        "rgb_keyframe_interval": rgb_kf,
        "rgb_jpeg_quality": jpeg_q,
        "zlib_level": zlib_level,
    }
    annex = _maybe_h264_annex(input_path, cfg, meta_v, max_frames)
    if annex:
        meta["playback"] = "h264_annex"
    save_bitstream(
        output_path, meta, centroids, blobs, zlib_level=zlib_level, video_annex=annex
    )

    full_size = Path(input_path).stat().st_size
    total_frames = meta_v["frame_count"] or n
    if max_frames is not None and total_frames > 0 and n < total_frames:
        orig_size = int(full_size * (n / total_frames))
    else:
        orig_size = full_size
    comp_size = Path(output_path).stat().st_size
    saving = (1 - comp_size / orig_size) * 100 if orig_size else 0
    return {
        "original_mb": orig_size / 1024 / 1024,
        "compressed_mb": comp_size / 1024 / 1024,
        "bandwidth_saving_pct": saving,
        "tx_orig_s": transmission_time_sec(orig_size),
        "tx_comp_s": transmission_time_sec(comp_size),
        "n_frames": n,
        "bitstream": str(output_path),
        "h264_annex_mb": (len(annex) / 1024 / 1024) if annex else 0,
    }


def _temporal_soft_stack(history: list[np.ndarray], tw: int) -> np.ndarray:
    """Pad early frames with the first edge map (same as training dataset)."""
    if not history:
        raise ValueError("empty soft-edge history")
    if len(history) >= tw:
        seq = history[-tw:]
    else:
        seq = [history[0]] * (tw - len(history)) + history
    return np.stack(seq, axis=0)


def _decode_rgb_key(
    blob: bytes,
    store_h: int,
    store_w: int,
    full_h: int,
    full_w: int,
) -> np.ndarray:
    """Decode a standalone JPEG key packet (no label state required)."""
    # JPEG anchors are full resolution; store_h/store_w apply to edge label maps only.
    _, bgr_k = unpack_frame_packet(blob, full_h, full_w, None)
    if bgr_k is None:
        raise ValueError("Expected RGB keyframe packet")
    rgb = cv2.cvtColor(bgr_k, cv2.COLOR_BGR2RGB)
    if rgb.shape[0] != full_h or rgb.shape[1] != full_w:
        rgb = cv2.resize(rgb, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    return rgb


def _keyframe_indices(n_frames: int, rgb_kf: int) -> list[int]:
    return list(range(0, n_frames, rgb_kf))


def _preload_rgb_keys(
    blobs: list[bytes],
    rgb_kf: int,
    store_h: int,
    store_w: int,
    full_h: int,
    full_w: int,
    workers: int = 4,
) -> dict[int, np.ndarray]:
    indices = _keyframe_indices(len(blobs), rgb_kf)
    if not indices:
        return {}

    def _one(ki: int) -> tuple[int, np.ndarray]:
        return ki, _decode_rgb_key(blobs[ki], store_h, store_w, full_h, full_w)

    if workers <= 1 or len(indices) < 4:
        return dict(_one(ki) for ki in indices)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(_one, indices))


def _interp_segment(frame_idx: int, n_frames: int, rgb_kf: int) -> tuple[int, int, float]:
    k0 = (frame_idx // rgb_kf) * rgb_kf
    k1 = k0 + rgb_kf
    if k1 >= n_frames:
        k1 = k0
    if k1 == k0:
        return k0, k0, 0.0
    alpha = (frame_idx - k0) / float(k1 - k0)
    return k0, k1, alpha


def _linear_blend_keys(
    a0: np.ndarray,
    a1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return np.clip(
        a0.astype(np.float32) * (1.0 - alpha) + a1.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def _interp_between_keys(
    frame_idx: int,
    n_frames: int,
    rgb_kf: int,
    key_rgb: dict[int, np.ndarray],
) -> np.ndarray:
    """Linear blend between surrounding full-res JPEG anchors."""
    k0, k1, alpha = _interp_segment(frame_idx, n_frames, rgb_kf)
    if k1 == k0:
        return key_rgb[k0].copy()
    return _linear_blend_keys(key_rgb[k0], key_rgb[k1], alpha)


def _flow_gray_pair(a0: np.ndarray, a1: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if scale < 1.0:
        h, w = a0.shape[:2]
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        s0 = cv2.resize(a0, (nw, nh), interpolation=cv2.INTER_AREA)
        s1 = cv2.resize(a1, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        s0, s1 = a0, a1
    g0 = cv2.cvtColor(s0, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(s1, cv2.COLOR_RGB2GRAY)
    return g0, g1


def _compute_key_flow(
    a0: np.ndarray,
    a1: np.ndarray,
    scale: float,
) -> np.ndarray:
    g0, g1 = _flow_gray_pair(a0, a1, scale)
    flow_small = cv2.calcOpticalFlowFarneback(
        g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    if scale < 1.0:
        h, w = a0.shape[:2]
        flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow *= 1.0 / scale
        return flow
    return flow_small


def _warp_rgb_with_flow(img: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    h, w = img.shape[:2]
    fx = flow[:, :, 0] * t
    fy = flow[:, :, 1] * t
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (x + fx).astype(np.float32)
    map_y = (y + fy).astype(np.float32)
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _flow_interp_between_keys(
    frame_idx: int,
    n_frames: int,
    rgb_kf: int,
    key_rgb: dict[int, np.ndarray],
    flow_cache: dict[tuple[int, int], np.ndarray],
    flow_scale: float = 0.5,
    linear_mix: float = 0.15,
    bidirectional: bool = True,
) -> np.ndarray:
    """Motion-compensated fill between JPEG anchors.

    Bidirectional warp (default) can leave double images on fast cars; use
    bidirectional=False or interp_mode=linear, or denser rgb_keyframe_interval.
    """
    k0, k1, alpha = _interp_segment(frame_idx, n_frames, rgb_kf)
    if k1 == k0:
        return key_rgb[k0].copy()
    a0, a1 = key_rgb[k0], key_rgb[k1]
    lin = _linear_blend_keys(a0, a1, alpha)
    if linear_mix >= 1.0:
        return lin
    seg = (k0, k1)
    if seg not in flow_cache:
        flow_cache[seg] = _compute_key_flow(a0, a1, flow_scale)
    flow = flow_cache[seg]
    if not bidirectional:
        w0 = _warp_rgb_with_flow(a0, flow, alpha)
        mix = float(np.clip(linear_mix, 0.0, 1.0))
        if mix <= 0:
            return w0
        return np.clip(
            w0.astype(np.float32) * (1.0 - mix) + lin.astype(np.float32) * mix,
            0,
            255,
        ).astype(np.uint8)
    w0 = _warp_rgb_with_flow(a0, flow, alpha)
    w1 = _warp_rgb_with_flow(a1, flow, alpha - 1.0)
    rgb = np.clip(
        w0.astype(np.float32) * (1.0 - alpha) + w1.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    if linear_mix > 0:
        rgb = np.clip(
            rgb.astype(np.float32) * (1.0 - linear_mix) + lin.astype(np.float32) * linear_mix,
            0,
            255,
        ).astype(np.uint8)
    return rgb


def _temporal_smooth_rgb(
    rgb: np.ndarray,
    prev: np.ndarray | None,
    strength: float,
    is_key: bool,
    history: list[np.ndarray] | None = None,
    window: int = 1,
) -> np.ndarray:
    """EMA (+ optional short history) to reduce flow shimmer and anchor pops."""
    if strength <= 0:
        return rgb
    if history is not None and window > 1:
        history.append(rgb.copy())
        while len(history) > window:
            history.pop(0)
        if len(history) >= 2:
            acc = history[0].astype(np.float32)
            for h in history[1:]:
                acc = acc * 0.55 + h.astype(np.float32) * 0.45
            rgb = np.clip(acc, 0, 255).astype(np.uint8)
    if prev is None:
        return rgb
    s = strength * (1.2 if is_key else 1.0)
    s = min(s, 0.52)
    return np.clip(
        rgb.astype(np.float32) * (1.0 - s) + prev.astype(np.float32) * s,
        0,
        255,
    ).astype(np.uint8)


def baseline_reconstruct(soft_rgb: np.ndarray) -> np.ndarray:
    """Fast fill: dilate edges + inpaint background."""
    import cv2

    gray = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 0).astype(np.uint8) * 255
    if mask.sum() == 0:
        return soft_rgb
    dilated = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    bgr = cv2.cvtColor(soft_rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, dilated, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def decompress_video(
    bitstream_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path | None,
    cfg: dict,
    device: str | None = None,
) -> dict:
    header_meta, centroids, blobs, video_annex = load_bitstream(bitstream_path)
    dec_cfg = cfg.get("decompress", {})
    playback = str(
        dec_cfg.get("playback", header_meta.get("playback", ""))
    ).lower()
    if video_annex and playback in ("h264_annex", "h264", "auto"):
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(video_annex)
        stats: dict = {
            "output_video": str(out_path),
            "n_frames": int(header_meta.get("n_frames", 0)),
            "playback": "h264_annex",
        }
        cap = cv2.VideoCapture(str(out_path))
        if cap.isOpened():
            n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if n_out > 0:
                stats["n_frames"] = n_out
            cap.release()
        return stats

    h, w = header_meta["height"], header_meta["width"]
    fps = float(header_meta.get("fps") or 24.0)
    cap_path = cfg.get("_source_video")
    if cap_path and Path(cap_path).exists():
        cap_probe = cv2.VideoCapture(str(cap_path))
        if cap_probe.isOpened():
            src_fps = float(cap_probe.get(cv2.CAP_PROP_FPS) or 0.0)
            if src_fps > 1:
                fps = src_fps
        cap_probe.release()
    key_int = header_meta.get("keyframe_interval", 30)
    down = header_meta.get("storage_downscale", 1)
    sh, sw = h // down, w // down
    rgb_kf = int(header_meta.get("rgb_keyframe_interval", 0) or 0)
    fill_mode = str(dec_cfg.get("fill_mode", "neural")).lower()
    use_lama = fill_mode in ("lama", "pretrained_lama")
    pretrained_enhance = str(dec_cfg.get("pretrained_enhance", "") or "")
    enhance_edge_blend = float(dec_cfg.get("enhance_edge_blend", 0.15))
    hybrid_blend = float(dec_cfg.get("hybrid_key_blend", 0.55))
    nn_residual = float(dec_cfg.get("nn_residual_blend", 0.0))
    model_cfg = cfg.get("model", {})
    tw = int(model_cfg.get("temporal_window", 7))
    use_key_interp = rgb_kf > 0 and fill_mode in ("key_interp", "key_interp_nn", "flow")
    interp_mode = str(dec_cfg.get("interp_mode", "flow" if fill_mode == "flow" else "linear"))
    if fill_mode == "flow":
        interp_mode = "flow"
    flow_scale = float(dec_cfg.get("flow_scale", 0.5))
    flow_linear_mix = float(dec_cfg.get("flow_linear_mix", 0.12))
    flow_bidirectional = bool(dec_cfg.get("flow_bidirectional", True))
    flow_cache: dict[tuple[int, int], np.ndarray] = {}

    def _interp_frame(i: int) -> np.ndarray:
        if interp_mode == "flow":
            return _flow_interp_between_keys(
                i,
                len(blobs),
                rgb_kf,
                key_rgb,
                flow_cache,
                flow_scale,
                flow_linear_mix,
                bidirectional=flow_bidirectional,
            )
        return _interp_between_keys(i, len(blobs), rgb_kf, key_rgb)
    if fill_mode == "key_interp_nn":
        nn_residual = max(nn_residual, float(dec_cfg.get("nn_residual_blend", 0.25)))

    ckpt_path = Path(checkpoint) if checkpoint else None
    use_nn = ckpt_path is not None and ckpt_path.is_file()
    if use_lama:
        use_nn = False
    if fill_mode in ("key_interp", "flow"):
        use_nn = False
    elif fill_mode == "key_interp_nn":
        use_nn = use_nn and nn_residual > 0
    model = None
    if use_nn:
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(checkpoint, map_location=dev)
        ckpt_meta = state.get("meta", {})
        model_h = int(ckpt_meta.get("height", h))
        model_w = int(ckpt_meta.get("width", w))
        model = SoftEdgeReconstructor(
            height=model_h,
            width=model_w,
            patch_size=int(model_cfg.get("patch_size", 16)),
            embed_dim=int(model_cfg.get("embed_dim", 256)),
            num_heads=int(model_cfg.get("num_heads", 8)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            temporal_window=tw,
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        model.load_state_dict(state["model"])
        model.to(dev).eval()

    use_refine = cfg.get("decompress", {}).get("post_refine", True)
    use_sharpen = cfg.get("decompress", {}).get("light_sharpen", False)
    sharpen_strength = float(cfg.get("decompress", {}).get("sharpen_strength", 0.35))
    out_fps = _norm_video_fps(fps)
    use_ffmpeg_out = bool(dec_cfg.get("output_ffmpeg", True))
    ffmpeg_crf = int(dec_cfg.get("ffmpeg_crf", 20))
    out: cv2.VideoWriter | _FfmpegVideoWriter | None = None
    if use_ffmpeg_out:
        try:
            out = _FfmpegVideoWriter(output_path, w, h, out_fps, crf=ffmpeg_crf)
        except (FileNotFoundError, OSError) as exc:
            print(f"Warning: ffmpeg pipe unavailable ({exc}); falling back to OpenCV VideoWriter.")
            use_ffmpeg_out = False
    if not use_ffmpeg_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            out = cv2.VideoWriter(str(output_path), fourcc, out_fps, (w, h))

    soft_history: list[np.ndarray] = []
    psnr_vals: list[float] = []
    ssim_vals: list[float] = []
    prev_labels: np.ndarray | None = None
    last_rgb_key: np.ndarray | None = None

    cap = cv2.VideoCapture(cap_path) if cap_path and Path(cap_path).exists() else None
    temporal_smooth = float(dec_cfg.get("temporal_smooth", 0.0))
    temporal_window = int(dec_cfg.get("temporal_smooth_window", 1))
    enhance_keys_only = bool(dec_cfg.get("pretrained_enhance_keyframes_only", False))
    prev_display: np.ndarray | None = None
    display_history: list[np.ndarray] = []

    def _nn_rgb(soft: np.ndarray) -> np.ndarray:
        if use_lama:
            return lama_inpaint(baseline_reconstruct(soft), soft)
        soft_history.append(soft)
        if len(soft_history) > tw:
            del soft_history[:-tw]
        if use_nn and model is not None and soft_history:
            dev = next(model.parameters()).device
            seq = _temporal_soft_stack(soft_history, tw)
            if seq.shape[1] != model_h or seq.shape[2] != model_w:
                seq = np.stack(
                    [
                        cv2.resize(seq[i], (model_w, model_h), interpolation=cv2.INTER_LINEAR)
                        for i in range(seq.shape[0])
                    ],
                    axis=0,
                )
            t = torch.from_numpy(seq).float().permute(0, 3, 1, 2) / 255.0
            t = t.unsqueeze(0).to(dev)
            with torch.no_grad():
                pred = model(t)[0].clamp(0, 1).cpu().numpy()
            out_rgb = (pred.transpose(1, 2, 0) * 255).astype(np.uint8)
            if out_rgb.shape[0] != h or out_rgb.shape[1] != w:
                out_rgb = cv2.resize(out_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            return out_rgb
        return baseline_reconstruct(soft)

    def _write_frame(rgb: np.ndarray, soft: np.ndarray | None, is_exact_key: bool) -> None:
        nonlocal last_rgb_key, prev_display
        if temporal_smooth > 0:
            hist = display_history if temporal_window > 1 else None
            rgb = _temporal_smooth_rgb(
                rgb,
                prev_display,
                temporal_smooth,
                is_exact_key,
                hist,
                temporal_window,
            )
            prev_display = rgb.copy()
        if not is_exact_key and last_rgb_key is not None and hybrid_blend > 0 and soft is not None:
            a = hybrid_blend
            rgb = np.clip(
                rgb.astype(np.float32) * (1 - a) + last_rgb_key.astype(np.float32) * a,
                0,
                255,
            ).astype(np.uint8)
        if not is_exact_key and soft is not None:
            if use_refine and (use_nn or use_lama):
                rgb = post_refine(rgb, soft)
            if use_sharpen and (use_nn or use_lama):
                rgb = light_sharpen(rgb, soft, sharpen_strength)
        if (
            pretrained_enhance
            and pretrained_enhance.lower() not in ("none", "off", "false")
            and (is_exact_key or not enhance_keys_only)
        ):
            rgb = enhance_frame(rgb, soft, pretrained_enhance, enhance_edge_blend)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if isinstance(out, _FfmpegVideoWriter):
            out.write(bgr)
        else:
            out.write(bgr)
        if cap is not None:
            ok, gt = cap.read()
            if ok:
                gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                if gt_rgb.shape[:2] != (h, w):
                    gt_rgb = cv2.resize(gt_rgb, (w, h))
                psnr_vals.append(psnr(gt_rgb, rgb))
                ssim_vals.append(ssim(gt_rgb, rgb))

    if rgb_kf > 0:
        key_workers = int(dec_cfg.get("key_decode_workers", 4))
        fast_interp = use_key_interp and not (use_nn and nn_residual > 0)
        full_fps = bool(header_meta.get("rgb_only")) or rgb_kf == 1

        if full_fps:
            for i, blob in tqdm(enumerate(blobs), total=len(blobs), desc="decompress"):
                rgb = _decode_rgb_key(blob, sh, sw, h, w)
                _write_frame(rgb, None, is_exact_key=True)
        elif fast_interp:
            key_rgb = _preload_rgb_keys(blobs, rgb_kf, sh, sw, h, w, key_workers)
            for i in tqdm(range(len(blobs)), total=len(blobs), desc="decompress"):
                if i % rgb_kf == 0:
                    rgb = key_rgb[i]
                    _write_frame(rgb, None, is_exact_key=True)
                else:
                    rgb = _interp_frame(i)
                    _write_frame(rgb, None, is_exact_key=False)
        else:
            key_rgb = {}
            if use_key_interp:
                key_rgb = _preload_rgb_keys(blobs, rgb_kf, sh, sw, h, w, key_workers)

            for i, blob in tqdm(enumerate(blobs), total=len(blobs), desc="decompress"):
                if i % key_int == 0:
                    prev_labels = None
                if blob[:1] == b"K":
                    _, bgr_k = unpack_frame_packet(blob, h, w, None)
                    labels = None
                else:
                    labels, bgr_k = unpack_frame_packet(blob, sh, sw, prev_labels)
                if bgr_k is not None:
                    rgb = cv2.cvtColor(bgr_k, cv2.COLOR_BGR2RGB)
                    if rgb.shape[0] != h or rgb.shape[1] != w:
                        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
                    if use_key_interp:
                        key_rgb[i] = rgb
                    last_rgb_key = rgb.copy()
                    soft_history.clear()
                    _write_frame(rgb, None, is_exact_key=True)
                elif use_key_interp:
                    rgb = _interp_frame(i)
                    last_rgb_key = key_rgb[(i // rgb_kf) * rgb_kf].copy()
                    labels, _ = unpack_frame_packet(blob, sh, sw, prev_labels)
                    assert labels is not None
                    prev_labels = labels
                    soft = None
                    if use_nn and nn_residual > 0:
                        lab = labels
                        if down > 1:
                            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_NEAREST)
                        soft = labels_to_soft_rgb(lab, centroids)
                        nn_rgb = _nn_rgb(soft)
                        rgb = np.clip(
                            rgb.astype(np.float32)
                            + nn_residual * (nn_rgb.astype(np.float32) - rgb.astype(np.float32)),
                            0,
                            255,
                        ).astype(np.uint8)
                    _write_frame(rgb, soft, is_exact_key=False)
                else:
                    assert labels is not None
                    prev_labels = labels
                    if down > 1:
                        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
                    soft = labels_to_soft_rgb(labels, centroids)
                    rgb = _nn_rgb(soft)
                    _write_frame(rgb, soft, is_exact_key=False)
    else:
        for i, labels in tqdm(
            enumerate(decode_all_labels(blobs, sh, sw, key_int)),
            total=len(blobs),
            desc="decompress",
        ):
            if down > 1:
                labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
            soft = labels_to_soft_rgb(labels, centroids)
            rgb = _nn_rgb(soft)
            _write_frame(rgb, soft, is_exact_key=False)

    if out is not None:
        out.release()
    if cap:
        cap.release()

    stats = {
        "output_video": str(output_path),
        "n_frames": len(blobs),
        "output_fps": out_fps,
        "output_encoder": "ffmpeg_libx264" if use_ffmpeg_out else "opencv",
    }
    if psnr_vals:
        stats["avg_psnr"] = float(np.mean(psnr_vals))
        stats["avg_ssim"] = float(np.mean(ssim_vals))
    return stats
