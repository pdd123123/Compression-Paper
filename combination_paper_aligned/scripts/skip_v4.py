from __future__ import annotations

import argparse

V4_EMBEDDING_BACKBONE = "efficientnet_b0"
V4_YOLO_MODEL = "yolov8s.pt"
V4_SCORE_DETECTOR_INTERVAL = 8
V4_CLASS_DETECTOR_INTERVAL = 8
V4_YOLO_IMGSZ = 416
V4_DETECTION_SCORE_BOOST = 0.15
V4_CONTEXT_MODE = "diverse"
V4_HISTORY_FRAMES = 20
V4_CONTEXT_SIZE = 4
V4_CLUSTER_MIN = 3
DEFAULT_MIN_RETENTION = 0.15
DEFAULT_TARGET_RETENTION = 0.30
DEFAULT_MAX_RETENTION = 0.85
DEFAULT_BUSY_SOFT_CAP = 0.82
DEFAULT_BUSY_MIN_FLOOR = 0.55
DEFAULT_BUSY_OBJECTS_THRESHOLD = 3
DEFAULT_FIXED_RETENTION = 0.2


def apply_v4_model_stack(cfg: dict) -> None:
    perf = cfg.setdefault("performance", {})
    perf["stream_frames"] = True
    perf["yolo_imgsz"] = V4_YOLO_IMGSZ
    perf["skip_input_clip"] = False

    emb = cfg.setdefault("embedding", {})
    emb["backbone"] = V4_EMBEDDING_BACKBONE
    emb["embed_dim"] = 1280

    ctx = cfg.setdefault("context", {})
    ctx["mode"] = V4_CONTEXT_MODE
    ctx["history_frames"] = V4_HISTORY_FRAMES
    ctx["context_size"] = V4_CONTEXT_SIZE
    ctx["cluster_min_samples"] = V4_CLUSTER_MIN

    cw = cfg.setdefault("class_weight", {})
    cw["yolo_model"] = V4_YOLO_MODEL
    cw["detector_interval"] = V4_CLASS_DETECTOR_INTERVAL

    sc = cfg.setdefault("scoring", {})
    sc["algorithm"] = "online"
    sc["busy_objects_threshold"] = DEFAULT_BUSY_OBJECTS_THRESHOLD
    sc["score_detector_interval"] = V4_SCORE_DETECTOR_INTERVAL
    sc["detection_score_boost"] = V4_DETECTION_SCORE_BOOST
    sc["vehicle_presence_floor"] = 0.25
    sc["busy_score_floor"] = 0.40
    sc["busy_score_boost"] = 0.18


def apply_v4_adaptive_policy(cfg: dict, args: argparse.Namespace) -> None:
    apply_v4_model_stack(cfg)
    sc = cfg.setdefault("scoring", {})
    sc["retention_mode"] = "adaptive"

    ad = cfg.setdefault("adaptive", {})
    ad["traffic_aware"] = True
    ad["information_aware"] = True
    ad["mad_multiplier"] = 1.0
    ad["flat_quantile"] = 0.72
    ad["min_retention_ratio"] = (
        args.min_retention if args.min_retention is not None else DEFAULT_MIN_RETENTION
    )
    ad["max_retention_ratio"] = (
        args.max_retention if args.max_retention is not None else DEFAULT_MAX_RETENTION
    )
    ad["soft_target_retention"] = (
        args.target_retention
        if args.target_retention is not None
        else DEFAULT_TARGET_RETENTION
    )
    ad["busy_soft_retention_cap"] = DEFAULT_BUSY_SOFT_CAP
    ad["busy_min_retention_floor"] = DEFAULT_BUSY_MIN_FLOOR


def apply_v4_fixed_policy(cfg: dict, retention: float) -> None:
    apply_v4_model_stack(cfg)
    sc = cfg.setdefault("scoring", {})
    sc["retention_mode"] = "target_ratio"
    sc["target_retention_ratio"] = float(retention)


def enrich_adaptive_skip_stats(cfg: dict, stats: dict) -> dict:
    n_total = int(stats.get("n_total", 0))
    n_kept = int(stats.get("n_kept", 0))
    stats["retention_mode"] = "adaptive_v4"
    stats["actual_retention_pct"] = 100.0 * n_kept / max(n_total, 1)
    stats["pipeline_style"] = {
        "algorithm": cfg["scoring"]["algorithm"],
        "context_mode": cfg["context"]["mode"],
        "history_frames": cfg["context"]["history_frames"],
        "context_size": cfg["context"]["context_size"],
        "stream_frames": cfg["performance"]["stream_frames"],
    }
    stats["model_stack"] = {
        "embedding_backbone": cfg["embedding"]["backbone"],
        "yolo_model": cfg["class_weight"]["yolo_model"],
        "score_detector_interval": cfg["scoring"]["score_detector_interval"],
        "yolo_imgsz": cfg["performance"]["yolo_imgsz"],
    }
    stats["retention_policy"] = {
        "min_retention_ratio": cfg["adaptive"]["min_retention_ratio"],
        "max_retention_ratio": cfg["adaptive"]["max_retention_ratio"],
        "soft_target_retention": cfg["adaptive"]["soft_target_retention"],
        "busy_objects_threshold": cfg["scoring"]["busy_objects_threshold"],
        "busy_soft_retention_cap": cfg["adaptive"]["busy_soft_retention_cap"],
        "busy_min_retention_floor": cfg["adaptive"]["busy_min_retention_floor"],
        "information_aware": cfg["adaptive"].get("information_aware", False),
        "traffic_busy_fraction": stats.get("traffic_busy_fraction"),
        "information_fraction": stats.get("information_fraction"),
    }
    return stats


def enrich_fixed_skip_stats(cfg: dict, stats: dict, retention: float) -> dict:
    n_total = int(stats.get("n_total", 0))
    n_kept = int(stats.get("n_kept", 0))
    stats["retention_mode"] = "fixed_v4"
    stats["target_retention_ratio"] = retention
    stats["actual_retention_pct"] = 100.0 * n_kept / max(n_total, 1)
    stats["pipeline_style"] = {
        "algorithm": cfg["scoring"]["algorithm"],
        "context_mode": cfg["context"]["mode"],
        "history_frames": cfg["context"]["history_frames"],
        "context_size": cfg["context"]["context_size"],
        "stream_frames": cfg["performance"]["stream_frames"],
    }
    stats["model_stack"] = {
        "embedding_backbone": cfg["embedding"]["backbone"],
        "yolo_model": cfg["class_weight"]["yolo_model"],
        "score_detector_interval": cfg["scoring"]["score_detector_interval"],
        "yolo_imgsz": cfg["performance"]["yolo_imgsz"],
    }
    return stats
