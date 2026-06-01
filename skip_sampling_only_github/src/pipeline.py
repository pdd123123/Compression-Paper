"""Video I/O and semantic sampling pipeline (run_sample / run_sample_adaptive)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from tqdm import tqdm

from .metrics import retention_ratio, size_reduction_pct, transmission_time_sec
from .progress_ui import PipelineProgress, estimate_processed_frames
from .semantic_sampler import SemanticSkipSampler
from .video_io import get_video_meta, iter_video_frames, resize_frame_max_height

# Re-export for scripts that import from pipeline
__all__ = [
    "iter_video_frames",
    "read_all_frames",
    "get_video_meta",
    "resize_frame_max_height",
    "run_semantic_sampling",
]


def read_all_frames(
    path: str | Path,
    max_frames: int | None = None,
    stride: int = 1,
    show_progress: bool = True,
    progress: PipelineProgress | None = None,
    frame_total: int | None = None,
) -> list[np.ndarray]:
    total = frame_total
    if total is None:
        total = max_frames
    if total is None:
        total = get_video_meta(path).get("frame_count") or None
    it = iter_video_frames(path, max_frames, stride)
    if progress:
        it = progress.iter(it, total=total, desc="read", unit="fr")
    elif show_progress:
        it = tqdm(it, total=total, desc="Read video", unit="fr")
    return [f for _, f in it]


def write_all_frames_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float,
    show_progress: bool = True,
    desc: str = "Write input clip MP4",
    progress: PipelineProgress | None = None,
) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    it: Iterator = frames
    if progress:
        it = progress.iter(frames, total=len(frames), desc="write", unit="fr")
    elif show_progress:
        it = tqdm(frames, desc=desc, unit="fr")
    for frame in it:
        out.write(frame)
    out.release()


def _open_video_writer(
    output_path: str | Path, fps: float, frame_bgr: np.ndarray
) -> cv2.VideoWriter:
    h, w = frame_bgr.shape[:2]
    return cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )


def write_frames_from_source(
    input_path: str | Path,
    output_path: str | Path,
    fps: float,
    *,
    max_frames: int | None = None,
    stride: int = 1,
    keep_indices: set[int] | None = None,
    write_all_processed: bool = False,
    show_progress: bool = True,
    desc: str = "Write MP4",
    progress: PipelineProgress | None = None,
    frame_total: int | None = None,
    write_total: int | None = None,
) -> int:
    """Second pass over video; write kept frames or entire processed clip."""
    it: Iterator = iter_video_frames(input_path, max_frames, stride)
    if progress:
        it = progress.iter(
            it, total=frame_total, desc="write", unit="fr"
        )
    elif show_progress:
        it = tqdm(it, total=frame_total, desc=desc, unit="fr")
    writer: cv2.VideoWriter | None = None
    n_written = 0
    try:
        for proc_idx, frame in it:
            if write_all_processed or (keep_indices is not None and proc_idx in keep_indices):
                if writer is None:
                    writer = _open_video_writer(output_path, fps, frame)
                writer.write(frame)
                n_written += 1
    finally:
        if writer is not None:
            writer.release()
    if n_written == 0:
        raise ValueError("No frames written")
    return n_written


def write_debug_from_source(
    input_path: str | Path,
    output_path: str | Path,
    fps: float,
    kept_set: set[int],
    scores: list[float],
    tau: float | None,
    *,
    max_frames: int | None = None,
    stride: int = 1,
    show_progress: bool = True,
    progress: PipelineProgress | None = None,
) -> None:
    it: Iterator = iter_video_frames(input_path, max_frames, stride)
    if progress:
        it = progress.iter(it, total=len(scores), desc="debug", unit="fr")
    elif show_progress:
        it = tqdm(it, total=len(scores), desc="Write debug MP4", unit="fr")
    writer: cv2.VideoWriter | None = None
    try:
        for proc_idx, frame in it:
            if writer is None:
                writer = _open_video_writer(output_path, fps, frame)
            vis = frame.copy()
            sc = scores[proc_idx] if proc_idx < len(scores) else 0.0
            if proc_idx in kept_set:
                label, color = f"KEEP #{proc_idx}  score={sc:.2e}", (0, 220, 0)
            else:
                label, color = f"SKIP #{proc_idx}  score={sc:.2e}", (0, 0, 220)
            cv2.putText(vis, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            if tau is not None:
                cv2.putText(
                    vis,
                    f"tau={tau:.2e}",
                    (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
            writer.write(vis)
    finally:
        if writer is not None:
            writer.release()


def write_sampled_video(
    frames: list[np.ndarray],
    indices: list[int],
    output_path: str | Path,
    fps: float,
    show_progress: bool = True,
    live_preview: bool = False,
    progress: PipelineProgress | None = None,
) -> None:
    if not indices:
        raise ValueError("No frames selected")
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    it: Iterator = indices
    if progress:
        it = progress.iter(indices, total=len(indices), desc="write", unit="fr")
    elif show_progress:
        it = tqdm(indices, desc="Write sampled MP4", unit="fr")
    for src_i in it:
        out.write(frames[src_i])
        if live_preview:
            vis = frames[src_i].copy()
            cv2.putText(
                vis,
                f"KEPT frame {src_i}",
                (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Skip sampling (q=quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                live_preview = False
    out.release()
    if live_preview:
        cv2.destroyAllWindows()


def write_debug_overlay_video(
    frames: list[np.ndarray],
    kept_set: set[int],
    scores: list[float],
    output_path: str | Path,
    fps: float,
    tau: float | None,
    show_progress: bool = True,
    progress: PipelineProgress | None = None,
) -> None:
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    it: Iterator = enumerate(frames)
    if progress:
        it = progress.iter(
            enumerate(frames), total=len(frames), desc="debug", unit="fr"
        )
    elif show_progress:
        it = tqdm(it, total=len(frames), desc="Write debug MP4", unit="fr")
    for i, frame in it:
        vis = frame.copy()
        sc = scores[i]
        if i in kept_set:
            label, color = f"KEEP #{i}  score={sc:.2e}", (0, 220, 0)
        else:
            label, color = f"SKIP #{i}  score={sc:.2e}", (0, 0, 220)
        cv2.putText(vis, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if tau is not None:
            cv2.putText(
                vis,
                f"tau={tau:.2e}",
                (12, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
        out.write(vis)
    out.release()


def save_manifest(
    path: str | Path,
    method: str,
    indices: list[int],
    meta: dict,
    scores: list[float] | None = None,
) -> None:
    payload = {
        "method": method,
        "indices": indices,
        "n_total": meta.get("n_frames"),
        "n_kept": len(indices),
        "retention_ratio": retention_ratio(len(indices), meta.get("n_frames", 0)),
        "meta": meta,
    }
    if scores is not None:
        payload["scores"] = scores
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _build_pipeline_plan(
    cfg: dict,
    *,
    stream: bool,
    debug_video: bool,
    input_clip_video: bool,
) -> list[str]:
    algo = cfg.get("scoring", {}).get("algorithm", "batch")
    steps: list[str] = []
    if not stream:
        steps.append("Load video into memory")
    if stream and algo == "motion_gated":
        steps.append("Score & select (motion + semantic)")
    elif stream and algo == "online":
        steps.append("Online score & select")
    elif stream:
        steps.append("Score & select frames")
    else:
        steps.append("Score & select frames")
    if input_clip_video:
        steps.append("Write input reference clip")
    steps.append("Write sampled MP4")
    if debug_video:
        steps.append("Write debug MP4")
    steps.append("Save manifest & report")
    return steps


def run_semantic_sampling(
    input_path: str | Path,
    output_video: str | Path,
    manifest_path: str | Path,
    cfg: dict,
    *,
    show_progress: bool = True,
    live_preview: bool = False,
    debug_video: str | Path | None = None,
    input_clip_video: str | Path | None = None,
) -> dict:
    vm = cfg["video"]
    perf = cfg.get("performance", {})
    stream = bool(perf.get("stream_frames", False))
    max_frames = vm.get("max_frames")
    stride = int(vm.get("frame_stride", 1))
    skip_input_clip = bool(perf.get("skip_input_clip", False))

    meta_v = get_video_meta(input_path)
    fps = meta_v["fps"] or 30.0
    frame_total = estimate_processed_frames(vm, meta_v)
    write_input_clip = input_clip_video is not None and not skip_input_clip

    prog = PipelineProgress(show_progress)
    prog.set_plan(
        _build_pipeline_plan(
            cfg,
            stream=stream,
            debug_video=debug_video is not None,
            input_clip_video=write_input_clip,
        )
    )
    if show_progress:
        mode = cfg.get("scoring", {}).get("retention_mode", "target_ratio")
        algo = cfg.get("scoring", {}).get("algorithm", "batch")
        n_txt = str(frame_total) if frame_total else "?"
        print(
            f"Pipeline: {len(prog.labels)} steps | ~{n_txt} frames to process | "
            f"mode={mode} algorithm={algo}",
            flush=True,
        )

    sampler = SemanticSkipSampler(cfg)

    if stream:
        if live_preview and show_progress:
            prog.note("--preview is ignored in stream mode")
        prog.begin()
        indices, scores = sampler.select_indices_streaming(
            input_path,
            vm,
            show_progress=show_progress,
            progress=prog,
            frame_total=frame_total,
        )
        n_total = len(scores)
        prog.done(f"tau={sampler.state.tau}")

        prog.begin("Write sampled MP4")
        n_out = write_frames_from_source(
            input_path,
            output_video,
            fps,
            max_frames=max_frames,
            stride=stride,
            keep_indices=set(indices),
            show_progress=show_progress,
            progress=prog,
            frame_total=frame_total,
        )
        prog.done(f"{n_out} frames written")

        if write_input_clip:
            prog.begin("Write input reference clip")
            n_in = write_frames_from_source(
                input_path,
                input_clip_video,
                fps,
                max_frames=max_frames,
                stride=stride,
                write_all_processed=True,
                show_progress=show_progress,
                progress=prog,
                frame_total=frame_total,
            )
            prog.done(f"{n_in} frames written")

        if debug_video is not None:
            prog.begin("Write debug MP4")
            write_debug_from_source(
                input_path,
                debug_video,
                fps,
                set(indices),
                scores,
                sampler.state.tau,
                max_frames=max_frames,
                stride=stride,
                show_progress=show_progress,
                progress=prog,
            )
            prog.done()
    else:
        prog.begin("Load video into memory")
        frames = read_all_frames(
            input_path,
            max_frames,
            stride,
            show_progress=show_progress,
            progress=prog,
            frame_total=frame_total,
        )
        n_total = len(frames)
        prog.done(f"{n_total} frames loaded")

        prog.begin("Score & select frames")
        indices, scores = sampler.select_indices(
            frames,
            show_progress=show_progress,
            progress=prog,
        )
        prog.done(f"tau={sampler.state.tau}")

        if write_input_clip:
            prog.begin("Write input reference clip")
            write_all_frames_video(
                frames,
                input_clip_video,
                fps,
                show_progress=show_progress,
                progress=prog,
            )
            prog.done(f"{n_total} frames")

        prog.begin("Write sampled MP4")
        write_sampled_video(
            frames,
            indices,
            output_video,
            fps,
            show_progress=show_progress,
            live_preview=live_preview,
            progress=prog,
        )
        prog.done(f"{len(indices)} frames")

        if debug_video is not None:
            prog.begin("Write debug MP4")
            write_debug_overlay_video(
                frames,
                set(indices),
                scores,
                debug_video,
                fps,
                sampler.state.tau,
                show_progress=show_progress,
                progress=prog,
            )
            prog.done(f"{n_total} frames")

    n_total = len(scores)
    prog.begin("Save manifest & report")
    meta = {
        "source": str(Path(input_path).name),
        "n_frames": n_total,
        "fps": fps,
        "tau": sampler.state.tau,
        "retention_mode": cfg["scoring"].get("retention_mode", "target_ratio"),
        "target_retention": cfg["scoring"].get("target_retention_ratio"),
        "actual_retention": retention_ratio(len(indices), n_total),
        "score_stats": getattr(sampler.state, "score_stats", {}),
        "fast_mode": bool(perf.get("fast_mode") or stream),
        "frame_stride": stride,
        "stream_frames": stream,
        "algorithm": cfg.get("scoring", {}).get("algorithm", "batch"),
        "context_mode": cfg.get("context", {}).get("mode", "diverse"),
        "traffic_busy_fraction": getattr(
            sampler.state, "traffic_busy_fraction", 0.0
        ),
    }
    save_manifest(manifest_path, "semantic", indices, meta, scores)
    prog.done(str(manifest_path))
    prog.finish_run()

    full_avi_bytes = Path(input_path).stat().st_size
    input_clip_path = Path(input_clip_video) if input_clip_video else None
    sampled_path = Path(output_video)
    input_clip_bytes = (
        input_clip_path.stat().st_size if input_clip_path and input_clip_path.is_file() else 0
    )
    sampled_bytes = sampled_path.stat().st_size if sampled_path.is_file() else 0
    clip_reduction = (
        (1 - sampled_bytes / input_clip_bytes) * 100 if input_clip_bytes > 0 else 0
    )

    return {
        "method": "semantic",
        "n_total": n_total,
        "n_kept": len(indices),
        "retention_ratio": retention_ratio(len(indices), n_total),
        "size_reduction_pct": size_reduction_pct(len(indices), n_total),
        "tau": sampler.state.tau,
        "fast_mode": meta["fast_mode"],
        "full_source_avi_mb": full_avi_bytes / 1024 / 1024,
        "input_clip_mb": input_clip_bytes / 1024 / 1024,
        "sampled_mb": sampled_bytes / 1024 / 1024,
        "clip_vs_sampled_reduction_pct": clip_reduction,
        "tx_input_clip_s": transmission_time_sec(input_clip_bytes),
        "tx_sampled_s": transmission_time_sec(sampled_bytes),
        "tx_full_avi_s": transmission_time_sec(full_avi_bytes),
        "input_clip_video": str(input_clip_path) if input_clip_path else None,
        "output_video": str(output_video),
        "manifest": str(manifest_path),
        "debug_video": str(debug_video) if debug_video else None,
    }
