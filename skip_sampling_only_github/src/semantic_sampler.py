"""Semantic frame sampler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from .class_weights import ClassWeightTracker, YoloClassDetector
from .video_io import iter_video_frames, resize_frame_max_height
from .context import build_context
from .embeddings import FrameEmbedder
from .scoring import (
    calibrate_tau,
    delta_map_proxy,
    estimate_adaptive_tau,
    frame_score,
    novelty_score,
    score_distribution_stats,
)
from .progress_ui import PipelineProgress
from .selection import select_diverse_topk


@dataclass
class SamplerState:
    history_feats: list[np.ndarray] = field(default_factory=list)
    scores_warmup: list[float] = field(default_factory=list)
    frame_n_objects: list[int] = field(default_factory=list)
    frame_motion: list[float] = field(default_factory=list)
    tau: float | None = None
    n_seen: int = 0
    n_kept: int = 0
    score_stats: dict = field(default_factory=dict)
    traffic_busy_fraction: float = 0.0
    prev_gray: np.ndarray | None = None


class SemanticSkipSampler:
    def __init__(self, cfg: dict):
        emb = cfg["embedding"]
        ctx = cfg["context"]
        cw = cfg["class_weight"]
        sc = cfg["scoring"]
        ad = cfg.get("adaptive", {})
        perf = cfg.get("performance", {})

        self.embed_batch_size = int(perf.get("embed_batch_size", 1))
        self.yolo_imgsz = int(perf.get("yolo_imgsz", 640))
        self.reuse_detections = bool(perf.get("reuse_detections", True))
        self._last_class_ids: list[int] = []

        self.history_len = int(ctx["history_frames"])
        self.context_size = int(ctx["context_size"])
        self.cluster_min = int(ctx.get("cluster_min_samples", 3))
        self.score_detector_interval = int(
            sc.get("score_detector_interval", cw.get("detector_interval", 1))
        )
        self.class_observe_interval = int(cw.get("detector_interval", 5))
        self.retention_mode = str(sc.get("retention_mode", "target_ratio"))
        self.target_retention = float(sc.get("target_retention_ratio", 0.2))
        self.tau_fixed = sc.get("tau")
        self.min_retention = float(ad.get("min_retention_ratio", 0.12))
        self.max_retention = float(ad.get("max_retention_ratio", 0.70))
        self.adaptive_mad_mult = float(ad.get("mad_multiplier", 1.2))
        self.adaptive_min_tau_q = float(ad.get("min_tau_quantile", 0.45))
        self.adaptive_flat_q = float(ad.get("flat_quantile", 0.68))
        self.soft_target_retention = float(ad.get("soft_target_retention", 0.35))
        self.traffic_aware_adaptive = bool(ad.get("traffic_aware", True))
        self.busy_soft_retention_cap = float(ad.get("busy_soft_retention_cap", 0.62))
        self.busy_min_retention_floor = float(ad.get("busy_min_retention_floor", 0.38))
        self.tau_warmup = int(sc.get("tau_warmup_frames", 120))
        self.novelty_power = float(sc.get("novelty_power", 1.0))
        self.min_score = float(sc.get("min_score", 0.0))
        self.selection_mode = str(sc.get("selection_mode", "diverse"))
        self.min_temporal_gap = int(sc.get("min_temporal_gap", 4))
        self.diversity_sim_penalty = float(sc.get("diversity_sim_penalty", 0.45))
        self.detection_score_boost = float(sc.get("detection_score_boost", 0.12))
        self.motion_blend = float(sc.get("motion_blend", 0.35))
        self.vehicle_presence_floor = float(sc.get("vehicle_presence_floor", 0.2))
        self.busy_objects_threshold = int(sc.get("busy_objects_threshold", 2))
        self.busy_motion_threshold = float(sc.get("busy_motion_threshold", 0.025))
        self.busy_score_boost = float(sc.get("busy_score_boost", 0.15))
        self.busy_score_floor = float(sc.get("busy_score_floor", 0.35))
        self.algorithm = str(sc.get("algorithm", "batch"))
        self.motion_gate_fraction = float(sc.get("motion_gate_fraction", 0.4))
        self.semantic_floor_interval = int(sc.get("semantic_floor_interval", 8))
        self.context_mode = str(ctx.get("mode", "diverse"))

        embed_size = int(
            perf.get("embed_input_size", emb.get("input_size", 224))
        )
        self.embedder = FrameEmbedder(
            backbone=emb.get("backbone", "mobilenet_v3_small"),
            device=emb.get("device"),
            input_size=embed_size,
            use_fp16=bool(perf.get("use_fp16", True)),
        )
        self.class_tracker = ClassWeightTracker(
            window_frames=int(cw.get("window_frames", 60)),
            rare_classes=cw.get("rare_classes"),
        )
        self.detector = YoloClassDetector(
            cw.get("yolo_model", "yolov8n.pt"),
            imgsz=self.yolo_imgsz,
            half=bool(perf.get("yolo_half", True)),
        )
        self.state = SamplerState()

    def reset(self) -> None:
        self.state = SamplerState()
        self._last_class_ids = []

    def _instant_motion(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (192, 108), interpolation=cv2.INTER_AREA)
        if self.state.prev_gray is None:
            self.state.prev_gray = gray
            return 1.0
        diff = float(np.mean(cv2.absdiff(gray, self.state.prev_gray))) / 255.0
        self.state.prev_gray = gray
        return diff

    def _update_tau(self) -> None:
        if self.tau_fixed is not None:
            self.state.tau = float(self.tau_fixed)
            return
        if self.state.tau is not None:
            return
        if len(self.state.scores_warmup) >= self.tau_warmup:
            self.state.tau = calibrate_tau(
                self.state.scores_warmup, self.target_retention
            )

    def score_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        feat: np.ndarray | None = None,
        class_ids: list[int] | None = None,
    ) -> tuple[float, dict]:
        if feat is None:
            feat = self.embedder.encode(frame_bgr)

        rep_idx, centers = build_context(
            self.state.history_feats[-self.history_len :],
            self.context_size,
            self.cluster_min,
            mode=self.context_mode,
        )
        hist = self.state.history_feats[-self.history_len :]
        context_feats = [hist[i] for i in rep_idx if i < len(hist)]

        if class_ids is None:
            if frame_idx % self.score_detector_interval == 0:
                class_ids = self.detector.detect_class_ids(frame_bgr)
            elif self.reuse_detections:
                class_ids = list(self._last_class_ids)
            else:
                class_ids = []
        if frame_idx % self.score_detector_interval == 0:
            self._last_class_ids = list(class_ids)
        if frame_idx % self.class_observe_interval == 0:
            self.class_tracker.observe(class_ids)

        wc = self.class_tracker.weight_for_classes(class_ids)
        semantic = frame_score(
            feat, context_feats, centers, wc, self.novelty_power
        )
        motion = self._instant_motion(frame_bgr)

        score = semantic + self.motion_blend * motion
        n_obj = len(class_ids)
        if class_ids:
            score = max(score, self.vehicle_presence_floor)
            if self.detection_score_boost > 0:
                score *= 1.0 + self.detection_score_boost * min(n_obj, 8)
        if (
            n_obj >= self.busy_objects_threshold
            and motion >= self.busy_motion_threshold
        ):
            score = max(score, self.busy_score_floor)
            score *= 1.0 + self.busy_score_boost * min(n_obj, 10)
        elif n_obj >= 1 and motion >= self.busy_motion_threshold * 0.5:
            score *= 1.0 + 0.08 * min(n_obj, 6)
        score = max(score, self.min_score)

        self.state.frame_n_objects.append(n_obj)
        self.state.frame_motion.append(motion)

        detail = {
            "delta_map": delta_map_proxy(feat, context_feats),
            "novelty": novelty_score(feat, centers, self.novelty_power),
            "motion": motion,
            "class_weight": wc,
            "n_objects": len(class_ids),
            "n_context": len(context_feats),
            "semantic": semantic,
            "score": score,
        }

        self.state.history_feats.append(feat)
        if len(self.state.history_feats) > self.history_len * 2:
            self.state.history_feats = self.state.history_feats[-self.history_len :]

        return score, detail

    def should_keep(self, frame_bgr: np.ndarray, frame_idx: int) -> tuple[bool, float, dict]:
        score, detail = self.score_frame(frame_bgr, frame_idx)
        self.state.n_seen += 1

        if self.state.tau is None:
            self.state.scores_warmup.append(score)
            self._update_tau()
            if self.state.tau is None:
                keep = (
                    score >= np.median(self.state.scores_warmup)
                    if self.state.scores_warmup
                    else True
                )
                if keep:
                    self.state.n_kept += 1
                return bool(keep), score, detail

        keep = score > self.state.tau
        if keep:
            self.state.n_kept += 1
        return keep, score, detail

    def select_indices(
        self,
        frames: list[np.ndarray],
        start_idx: int = 0,
        show_progress: bool = True,
        progress: PipelineProgress | None = None,
    ) -> tuple[list[int], list[float]]:
        self.reset()
        scores: list[float] = []
        feats: list[np.ndarray] = []
        if progress:
            progress.begin_sub("Score frames (embed + YOLO)", total=len(frames))
        it: Iterable = enumerate(frames)
        if progress:
            it = progress.iter(enumerate(frames), total=len(frames), desc="embed", unit="fr")
        elif show_progress and frames:
            it = tqdm(enumerate(frames), total=len(frames), desc="Score frames", unit="fr")
        for i, frame in it:
            sc, _ = self.score_frame(frame, start_idx + i)
            scores.append(sc)
            feats.append(self.state.history_feats[-1].copy())

        score_arr = np.array(scores, dtype=np.float64)
        self.state.score_stats = score_distribution_stats(score_arr)
        feat_mat = np.stack(feats, axis=0) if feats else None

        if progress:
            progress.done_sub(f"{len(frames)} frames scored")
            progress.begin_sub("Select frames to keep")
        elif show_progress:
            print("Selecting frames...", flush=True)
        if self.retention_mode == "adaptive":
            kept = self._select_adaptive(score_arr, feat_mat)
        else:
            kept = self._select_target_ratio(score_arr, feat_mat)
        if progress:
            progress.done_sub(
                f"kept {len(kept)}/{len(frames)} "
                f"({100 * len(kept) / max(len(frames), 1):.1f}%)"
            )

        self.state.n_kept = len(kept)
        self.state.n_seen = len(frames)
        return kept, score_arr.tolist()

    def select_indices_streaming(
        self,
        video_path: str | Path,
        video_cfg: dict,
        show_progress: bool = True,
        progress: PipelineProgress | None = None,
        frame_total: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """Score frames from disk; keep embeddings only (low memory)."""
        if self.algorithm == "motion_gated":
            return self._select_indices_motion_gated_streaming(
                video_path, video_cfg, show_progress, progress, frame_total
            )
        if self.algorithm == "online":
            return self._select_indices_online_streaming(
                video_path, video_cfg, show_progress, progress, frame_total
            )
        return self._select_indices_full_streaming(
            video_path, video_cfg, show_progress, progress, frame_total
        )

    def _select_indices_online_streaming(
        self,
        video_path: str | Path,
        video_cfg: dict,
        show_progress: bool = True,
        progress: PipelineProgress | None = None,
        frame_total: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """One pass: warmup then keep frames with score > tau (paper-style streaming)."""
        self.reset()
        max_frames = video_cfg.get("max_frames")
        stride = int(video_cfg.get("frame_stride", 1))
        max_height = int(video_cfg.get("max_height") or 0)
        kept: list[int] = []
        scores: list[float] = []
        if progress:
            progress.begin_sub("Online score & keep", total=frame_total)
        it: Iterable = iter_video_frames(video_path, max_frames, stride)
        if progress:
            it = progress.iter(it, total=frame_total, desc="online", unit="fr")
        elif show_progress:
            it = tqdm(it, total=frame_total, desc="Online score", unit="fr")
        for proc_idx, frame in it:
            if max_height > 0:
                frame = resize_frame_max_height(frame, max_height)
            keep, sc, _ = self.should_keep(frame, proc_idx)
            scores.append(sc)
            if keep:
                kept.append(proc_idx)
        score_arr = np.array(scores, dtype=np.float64)
        self.state.score_stats = score_distribution_stats(score_arr)
        if progress:
            progress.done_sub(
                f"kept {len(kept)}/{len(scores)} "
                f"({100 * len(kept) / max(len(scores), 1):.1f}%)"
            )
        self.state.n_kept = len(kept)
        self.state.n_seen = len(scores)
        return kept, score_arr.tolist()

    def _select_indices_motion_gated_streaming(
        self,
        video_path: str | Path,
        video_cfg: dict,
        show_progress: bool = True,
        progress: PipelineProgress | None = None,
        frame_total: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """Pass 1: motion only. Pass 2: CNN+YOLO only on high-motion candidates."""
        max_frames = video_cfg.get("max_frames")
        stride = int(video_cfg.get("frame_stride", 1))
        max_height = int(video_cfg.get("max_height") or 0)

        self.reset()
        motions: list[float] = []
        if progress:
            progress.begin_sub("Motion scan (pass 1)", total=frame_total)
        it1: Iterable = iter_video_frames(video_path, max_frames, stride)
        if progress:
            it1 = progress.iter(it1, total=frame_total, desc="motion", unit="fr")
        elif show_progress:
            it1 = tqdm(it1, total=frame_total, desc="Motion pass", unit="fr")
        for _, frame in it1:
            if max_height > 0:
                frame = resize_frame_max_height(frame, max_height)
            motions.append(self._instant_motion(frame))

        n = len(motions)
        motion_arr = np.array(motions, dtype=np.float64)
        frac = float(np.clip(self.motion_gate_fraction, 0.05, 1.0))
        thr = float(np.quantile(motion_arr, 1.0 - frac)) if n else 0.0
        floor_iv = max(0, self.semantic_floor_interval)
        candidates = {
            i
            for i in range(n)
            if motion_arr[i] >= thr
            or (floor_iv > 0 and i % floor_iv == 0)
        }
        if progress:
            progress.done_sub(f"{n} frames, {len(candidates)} need CNN")
        elif show_progress:
            print(
                f"Motion gate: {len(candidates)}/{n} frames get semantic scoring",
                flush=True,
            )

        self.reset()
        return self._score_stream_with_candidates(
            video_path,
            video_cfg,
            candidates,
            motion_arr,
            show_progress,
            progress,
            frame_total,
        )

    def _score_stream_with_candidates(
        self,
        video_path: str | Path,
        video_cfg: dict,
        candidates: set[int],
        motion_arr: np.ndarray,
        show_progress: bool,
        progress: PipelineProgress | None = None,
        frame_total: int | None = None,
    ) -> tuple[list[int], list[float]]:
        max_frames = video_cfg.get("max_frames")
        stride = int(video_cfg.get("frame_stride", 1))
        max_height = int(video_cfg.get("max_height") or 0)
        batch_size = max(1, self.embed_batch_size)

        if progress:
            progress.begin_sub("Semantic score (pass 2)", total=frame_total)
        scores: list[float] = []
        feats: list[np.ndarray | None] = []
        batch_frames: list[np.ndarray] = []
        batch_indices: list[int] = []

        def flush_batch() -> None:
            if not batch_frames:
                return
            feat_mat = self.embedder.encode_batch(batch_frames)
            det_by_j: dict[int, list[int]] = {}
            if self.score_detector_interval > 0:
                need_j = [
                    j
                    for j, frame_idx in enumerate(batch_indices)
                    if frame_idx % self.score_detector_interval == 0
                ]
                if need_j:
                    det_frames = [batch_frames[j] for j in need_j]
                    det_lists = self.detector.detect_class_ids_batch(det_frames)
                    for j, cids in zip(need_j, det_lists):
                        det_by_j[j] = cids
            for j, (frame_idx, frame) in enumerate(zip(batch_indices, batch_frames)):
                cids = det_by_j[j] if j in det_by_j else None
                sc, _ = self.score_frame(
                    frame, frame_idx, feat=feat_mat[j], class_ids=cids
                )
                scores.append(sc)
                feats.append(self.state.history_feats[-1].copy())
            batch_frames.clear()
            batch_indices.clear()

        it: Iterable = iter_video_frames(video_path, max_frames, stride)
        if progress:
            it = progress.iter(it, total=frame_total, desc="semantic", unit="fr")
        elif show_progress:
            it = tqdm(it, total=frame_total, desc="Score frames", unit="fr")
        for proc_idx, frame in it:
            if max_height > 0:
                frame = resize_frame_max_height(frame, max_height)
            if proc_idx not in candidates:
                m = float(motion_arr[proc_idx])
                scores.append(m)
                feats.append(None)
                self.state.frame_n_objects.append(0)
                self.state.frame_motion.append(m)
                continue
            batch_frames.append(frame)
            batch_indices.append(proc_idx)
            if len(batch_frames) >= batch_size:
                flush_batch()
        flush_batch()

        score_arr = np.array(scores, dtype=np.float64)
        self.state.score_stats = score_distribution_stats(score_arr)
        feat_mat = None
        if any(f is not None for f in feats):
            dim = next(f for f in feats if f is not None).shape[0]
            stacked = np.zeros((len(feats), dim), dtype=np.float32)
            for i, f in enumerate(feats):
                if f is not None:
                    stacked[i] = f
            feat_mat = stacked

        if progress:
            progress.done_sub(f"{len(scores)} frames scored")
            progress.begin_sub("Select frames to keep")
        elif show_progress:
            print("Selecting frames...", flush=True)
        if self.retention_mode == "adaptive":
            kept = self._select_adaptive(score_arr, feat_mat)
        else:
            kept = self._select_target_ratio(score_arr, feat_mat)

        if progress:
            progress.done_sub(
                f"kept {len(kept)}/{len(scores)} "
                f"({100 * len(kept) / max(len(scores), 1):.1f}%)"
            )
        self.state.n_kept = len(kept)
        self.state.n_seen = len(scores)
        return kept, score_arr.tolist()

    def _select_indices_full_streaming(
        self,
        video_path: str | Path,
        video_cfg: dict,
        show_progress: bool = True,
        progress: PipelineProgress | None = None,
        frame_total: int | None = None,
    ) -> tuple[list[int], list[float]]:
        """Score every processed frame (original streaming path)."""
        self.reset()
        max_frames = video_cfg.get("max_frames")
        stride = int(video_cfg.get("frame_stride", 1))
        max_height = int(video_cfg.get("max_height") or 0)
        batch_size = max(1, self.embed_batch_size)

        if progress:
            progress.begin_sub("Score frames (embed + YOLO)", total=frame_total)
        scores: list[float] = []
        feats: list[np.ndarray] = []
        batch_frames: list[np.ndarray] = []
        batch_indices: list[int] = []

        def flush_batch() -> None:
            if not batch_frames:
                return
            feat_mat = self.embedder.encode_batch(batch_frames)
            det_by_j: dict[int, list[int]] = {}
            if self.score_detector_interval > 0:
                need_j = [
                    j
                    for j, frame_idx in enumerate(batch_indices)
                    if frame_idx % self.score_detector_interval == 0
                ]
                if need_j:
                    det_frames = [batch_frames[j] for j in need_j]
                    det_lists = self.detector.detect_class_ids_batch(det_frames)
                    for j, cids in zip(need_j, det_lists):
                        det_by_j[j] = cids
            for j, (frame_idx, frame) in enumerate(zip(batch_indices, batch_frames)):
                cids = det_by_j[j] if j in det_by_j else None
                sc, _ = self.score_frame(
                    frame, frame_idx, feat=feat_mat[j], class_ids=cids
                )
                scores.append(sc)
                feats.append(self.state.history_feats[-1].copy())
            batch_frames.clear()
            batch_indices.clear()

        it: Iterable = iter_video_frames(video_path, max_frames, stride)
        if progress:
            it = progress.iter(it, total=frame_total, desc="embed", unit="fr")
        elif show_progress:
            it = tqdm(it, total=frame_total, desc="Score frames", unit="fr")
        for proc_idx, frame in it:
            if max_height > 0:
                frame = resize_frame_max_height(frame, max_height)
            batch_frames.append(frame)
            batch_indices.append(proc_idx)
            if len(batch_frames) >= batch_size:
                flush_batch()
        flush_batch()

        score_arr = np.array(scores, dtype=np.float64)
        self.state.score_stats = score_distribution_stats(score_arr)
        feat_mat = np.stack(feats, axis=0) if feats else None

        if progress:
            progress.done_sub(f"{len(scores)} frames scored")
            progress.begin_sub("Select frames to keep")
        elif show_progress:
            print("Selecting frames...", flush=True)
        if self.retention_mode == "adaptive":
            kept = self._select_adaptive(score_arr, feat_mat)
        else:
            kept = self._select_target_ratio(score_arr, feat_mat)
        if progress:
            progress.done_sub(
                f"kept {len(kept)}/{len(scores)} "
                f"({100 * len(kept) / max(len(scores), 1):.1f}%)"
            )

        self.state.n_kept = len(kept)
        self.state.n_seen = len(scores)
        return kept, score_arr.tolist()

    def _select_target_ratio(
        self, score_arr: np.ndarray, feat_mat: np.ndarray | None
    ) -> list[int]:
        n_keep = max(1, int(round(len(score_arr) * self.target_retention)))
        if self.tau_fixed is not None:
            tau = float(self.tau_fixed)
        else:
            tau = calibrate_tau(score_arr.tolist(), self.target_retention)
        self.state.tau = tau

        if self.selection_mode == "diverse":
            return select_diverse_topk(
                score_arr,
                n_keep,
                feat_mat,
                self.min_temporal_gap,
                self.diversity_sim_penalty,
            )

        kept = [i for i, sc in enumerate(score_arr) if sc > tau]
        if len(kept) > n_keep:
            order = np.argsort(-score_arr)[:n_keep]
            return sorted(int(x) for x in order)
        if len(kept) < n_keep:
            order = np.argsort(-score_arr)
            kept = []
            for ii in order:
                kept.append(int(ii))
                if len(kept) >= n_keep:
                    break
            return sorted(kept)
        return sorted(kept)

    def _traffic_busy_fraction(self) -> float:
        objs = self.state.frame_n_objects
        motions = self.state.frame_motion
        if not objs or len(objs) != len(motions):
            return 0.0
        busy = sum(
            1
            for no, mo in zip(objs, motions)
            if no >= self.busy_objects_threshold
            and mo >= self.busy_motion_threshold
        )
        return busy / len(objs)

    def _effective_adaptive_retention_bounds(self, n: int) -> tuple[int, int, int]:
        """Return (n_min, soft_n, n_max) frame counts for adaptive selection."""
        busy_frac = self._traffic_busy_fraction() if self.traffic_aware_adaptive else 0.0
        self.state.traffic_busy_fraction = busy_frac

        min_r = self.min_retention
        soft_r = self.soft_target_retention
        if busy_frac > 0.0:
            soft_r = soft_r + busy_frac * (self.busy_soft_retention_cap - soft_r)
            min_r = min_r + busy_frac * (self.busy_min_retention_floor - min_r)
        soft_r = float(np.clip(soft_r, min_r, self.max_retention - 0.02))

        n_min = max(1, int(round(n * min_r)))
        soft_n = max(n_min, int(round(n * soft_r)))
        n_max = max(soft_n, int(round(n * self.max_retention)))
        return n_min, soft_n, n_max

    def _select_adaptive(
        self, score_arr: np.ndarray, feat_mat: np.ndarray | None
    ) -> list[int]:
        n = len(score_arr)
        if self.tau_fixed is not None:
            tau = float(self.tau_fixed)
        else:
            tau = estimate_adaptive_tau(
                score_arr,
                self.adaptive_mad_mult,
                self.adaptive_min_tau_q,
                self.adaptive_flat_q,
            )
        self.state.tau = tau

        kept = [i for i, sc in enumerate(score_arr) if sc > tau]

        n_min, soft_n, n_max = self._effective_adaptive_retention_bounds(n)

        if len(kept) < soft_n and self.tau_fixed is None:
            soft_r = soft_n / max(n, 1)
            tau = float(np.quantile(score_arr, 1.0 - soft_r))
            self.state.tau = tau
            kept = [i for i, sc in enumerate(score_arr) if sc > tau]
        if len(kept) < soft_n:
            order = np.argsort(-score_arr)
            kept_set = set(kept)
            for ii in order:
                ii = int(ii)
                if ii not in kept_set:
                    kept.append(ii)
                    kept_set.add(ii)
                if len(kept) >= soft_n:
                    break
            kept = sorted(kept)

        if len(kept) < n_min:
            order = np.argsort(-score_arr)
            kept_set = set(kept)
            for ii in order:
                if int(ii) not in kept_set:
                    kept.append(int(ii))
                    kept_set.add(int(ii))
                if len(kept) >= n_min:
                    break
            kept = sorted(kept)
        elif len(kept) > n_max:
            kept = select_diverse_topk(
                score_arr,
                n_max,
                feat_mat,
                self.min_temporal_gap,
                self.diversity_sim_penalty,
            )
        elif self.selection_mode == "diverse" and len(kept) > 1:
            k = len(kept)
            sub_scores = np.array([score_arr[i] for i in kept])
            sub_feats = feat_mat[kept] if feat_mat is not None else None
            order_local = select_diverse_topk(
                sub_scores,
                k,
                sub_feats,
                self.min_temporal_gap,
                self.diversity_sim_penalty,
            )
            kept = sorted(kept[i] for i in order_local)

        return sorted(kept)
