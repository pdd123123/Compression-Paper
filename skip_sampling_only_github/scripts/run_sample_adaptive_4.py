

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import apply_light_preset, apply_quality_preset, apply_turbo_preset, load_config
from src.output_paths import get_output_dir
from src.pipeline import run_semantic_sampling

# Same model stack as v3
V4_EMBEDDING_BACKBONE = "efficientnet_b0"
V4_YOLO_MODEL = "yolov8s.pt"
V4_SCORE_DETECTOR_INTERVAL = 8
V4_CLASS_DETECTOR_INTERVAL = 8
V4_YOLO_IMGSZ = 416
V4_DETECTION_SCORE_BOOST = 0.15

# K-means / streaming (paper-style)
V4_CONTEXT_MODE = "diverse"
V4_HISTORY_FRAMES = 20
V4_CONTEXT_SIZE = 4
V4_CLUSTER_MIN = 3

# Same adaptive retention as v3
DEFAULT_MIN_RETENTION = 0.15
DEFAULT_TARGET_RETENTION = 0.30
DEFAULT_MAX_RETENTION = 0.85
DEFAULT_BUSY_SOFT_CAP = 0.82
DEFAULT_BUSY_MIN_FLOOR = 0.55
DEFAULT_BUSY_OBJECTS_THRESHOLD = 3


def _open_with_default_app(path: Path) -> None:
    p = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{p}"')
    else:
        os.system(f'xdg-open "{p}"')


def _apply_v4_policy(cfg: dict, args: argparse.Namespace) -> None:
    perf = cfg.setdefault("performance", {})
    perf["stream_frames"] = True
    perf["yolo_imgsz"] = V4_YOLO_IMGSZ

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
    sc["retention_mode"] = "adaptive"
    sc["busy_objects_threshold"] = DEFAULT_BUSY_OBJECTS_THRESHOLD
    sc["score_detector_interval"] = V4_SCORE_DETECTOR_INTERVAL
    sc["detection_score_boost"] = V4_DETECTION_SCORE_BOOST
    sc["vehicle_presence_floor"] = 0.25
    sc["busy_score_floor"] = 0.40
    sc["busy_score_boost"] = 0.18

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive skip sampling v4: online stream + K-means context "
            "(paper-style scoring, v3 adaptive retention)"
        )
    )
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--min-retention", type=float, default=None)
    parser.add_argument("--max-retention", type=float, default=None)
    parser.add_argument("--target-retention", type=float, default=None)
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--debug-video", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--no-input-clip", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quality:
        cfg = apply_quality_preset(cfg)
    elif args.light:
        cfg = apply_light_preset(cfg)
    elif args.turbo:
        cfg = apply_turbo_preset(cfg)

    _apply_v4_policy(cfg, args)

    if args.max_frames is not None:
        cfg["video"]["max_frames"] = args.max_frames
    if args.tau is not None:
        cfg["scoring"]["tau"] = args.tau

    inp = Path(args.input)
    if not inp.is_file():
        print("Error: input not found:", inp, file=sys.stderr)
        sys.exit(1)

    stem = inp.stem
    out_dir = get_output_dir(ROOT, "adaptive_4", args.output_dir)
    out_vid = out_dir / f"{stem}_adaptive4_sampled.mp4"
    manifest = out_dir / f"{stem}_adaptive4_manifest.json"
    debug_path = out_dir / f"{stem}_adaptive4_debug.mp4" if args.debug_video else None

    max_f = cfg["video"].get("max_frames")
    n_label = max_f if max_f else "all"
    input_clip = None
    if not args.no_input_clip:
        if args.debug_video:
            cfg.setdefault("performance", {})["skip_input_clip"] = False
        skip_clip = cfg.get("performance", {}).get("skip_input_clip", False)
        if not skip_clip:
            input_clip = out_dir / f"{stem}_input_{n_label}fr.mp4"

    stats = run_semantic_sampling(
        inp,
        out_vid,
        manifest,
        cfg,
        show_progress=not args.no_progress,
        debug_video=debug_path,
        input_clip_video=input_clip,
    )
    n_total = stats["n_total"]
    pct = 100.0 * stats["n_kept"] / max(n_total, 1)
    stats["retention_mode"] = "adaptive_v4"
    stats["actual_retention_pct"] = pct
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

    report = out_dir / f"{stem}_adaptive4_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Done.")
    print(
        f"Pipeline: online stream + K-means context "
        f"(history={V4_HISTORY_FRAMES}, k={V4_CONTEXT_SIZE})"
    )
    print(f"Models: {V4_EMBEDDING_BACKBONE} + {V4_YOLO_MODEL}")
    info_frac = stats.get("information_fraction")
    if info_frac is not None:
        print(f"Information richness: {100.0 * info_frac:.1f}% of clip")
    print(f"Kept {stats['n_kept']} / {n_total} frames ({pct:.1f}%)")
    print(f"tau = {stats.get('tau')}")
    if input_clip and input_clip.is_file():
        print("Input clip (original):", input_clip)
    print("Sampled video:", out_vid)
    if debug_path and debug_path.is_file():
        print("Debug video:", debug_path)
    print("Report:", report)
    if args.open and out_vid.is_file():
        _open_with_default_app(out_vid)


if __name__ == "__main__":
    main()
