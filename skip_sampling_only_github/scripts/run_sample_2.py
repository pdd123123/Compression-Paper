#!/usr/bin/env python3
"""Fixed-retention semantic skip sampling (sample_2: same stack as adaptive v4).

Shared with run_sample_adaptive_4.py (scoring / detection / pipeline only):
- EfficientNet-B0 + YOLOv8s, online stream, K-means context

Difference from v4:
- retention_mode=target_ratio: user sets --retention (default 20%, paper-style)
- After streaming scores, tau is calibrated on the full clip to hit that ratio closely

Pair with v4 for fair comparison: same score, fixed vs adaptive keep rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.output_paths import get_output_dir
from src.pipeline import run_semantic_sampling

# Shared with run_sample_adaptive_4.py
S2_EMBEDDING_BACKBONE = "efficientnet_b0"
S2_YOLO_MODEL = "yolov8s.pt"
S2_SCORE_DETECTOR_INTERVAL = 8
S2_CLASS_DETECTOR_INTERVAL = 8
S2_YOLO_IMGSZ = 416
S2_DETECTION_SCORE_BOOST = 0.15
S2_BUSY_OBJECTS_THRESHOLD = 3

S2_CONTEXT_MODE = "diverse"
S2_HISTORY_FRAMES = 20
S2_CONTEXT_SIZE = 4
S2_CLUSTER_MIN = 3

DEFAULT_RETENTION = 0.20


def _open_with_default_app(path: Path) -> None:
    p = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{p}"')
    else:
        os.system(f'xdg-open "{p}"')


def _apply_sample2_scoring_policy(cfg: dict, retention: float) -> None:
    """Same models / online+K-means / detection scoring as adaptive v4."""
    perf = cfg.setdefault("performance", {})
    perf["stream_frames"] = True
    perf["yolo_imgsz"] = S2_YOLO_IMGSZ

    emb = cfg.setdefault("embedding", {})
    emb["backbone"] = S2_EMBEDDING_BACKBONE
    emb["embed_dim"] = 1280

    ctx = cfg.setdefault("context", {})
    ctx["mode"] = S2_CONTEXT_MODE
    ctx["history_frames"] = S2_HISTORY_FRAMES
    ctx["context_size"] = S2_CONTEXT_SIZE
    ctx["cluster_min_samples"] = S2_CLUSTER_MIN

    cw = cfg.setdefault("class_weight", {})
    cw["yolo_model"] = S2_YOLO_MODEL
    cw["detector_interval"] = S2_CLASS_DETECTOR_INTERVAL

    sc = cfg.setdefault("scoring", {})
    sc["algorithm"] = "online"
    sc["retention_mode"] = "target_ratio"
    sc["target_retention_ratio"] = retention
    sc["busy_objects_threshold"] = S2_BUSY_OBJECTS_THRESHOLD
    sc["score_detector_interval"] = S2_SCORE_DETECTOR_INTERVAL
    sc["detection_score_boost"] = S2_DETECTION_SCORE_BOOST
    sc["vehicle_presence_floor"] = 0.25
    sc["busy_score_floor"] = 0.40
    sc["busy_score_boost"] = 0.18


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-retention skip sampling (sample_2): same scoring as v4, "
            "paper-style --retention (default 20%)"
        )
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output MP4 path")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--retention",
        type=float,
        default=DEFAULT_RETENTION,
        help=f"Fraction of frames to keep (default {DEFAULT_RETENTION:.0%}, paper-style)",
    )
    parser.add_argument("--tau", type=float, default=None, help="Fixed score threshold (optional)")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Preview kept frames")
    parser.add_argument("--debug-video", action="store_true", help="Write labeled debug MP4")
    parser.add_argument("--open", action="store_true", help="Open outputs when done")
    parser.add_argument("--no-input-clip", action="store_true", help="Skip input reference clip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _apply_sample2_scoring_policy(cfg, args.retention)

    if args.max_frames is not None:
        cfg["video"]["max_frames"] = args.max_frames
    if args.tau is not None:
        cfg["scoring"]["tau"] = args.tau

    inp = Path(args.input)
    if not inp.is_file():
        print("Error: input not found:", inp, file=sys.stderr)
        sys.exit(1)

    stem = inp.stem
    out_dir = get_output_dir(ROOT, "sample_2", args.output_dir)
    out_vid = Path(args.output) if args.output else out_dir / f"{stem}_sample2.mp4"
    manifest = out_dir / f"{stem}_sample2_manifest.json"
    debug_path = out_dir / f"{stem}_sample2_debug.mp4" if args.debug_video else None

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
        live_preview=args.preview,
        debug_video=debug_path,
        input_clip_video=input_clip,
    )
    n_total = stats["n_total"]
    pct = 100.0 * stats["n_kept"] / max(n_total, 1)
    stats["retention_mode"] = "target_ratio"
    stats["target_retention_ratio"] = args.retention
    stats["actual_retention_pct"] = pct
    stats["pipeline_style"] = {
        "algorithm": cfg["scoring"]["algorithm"],
        "context_mode": cfg["context"]["mode"],
        "history_frames": cfg["context"]["history_frames"],
        "context_size": cfg["context"]["context_size"],
    }
    stats["model_stack"] = {
        "embedding_backbone": cfg["embedding"]["backbone"],
        "yolo_model": cfg["class_weight"]["yolo_model"],
        "score_detector_interval": cfg["scoring"]["score_detector_interval"],
        "yolo_imgsz": cfg["performance"]["yolo_imgsz"],
    }

    report = out_dir / f"{stem}_sample2_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Done.")
    print(
        f"Pipeline: online + K-means | target retention {100 * args.retention:.0f}% | "
        f"actual {pct:.1f}%"
    )
    print(f"Models: {S2_EMBEDDING_BACKBONE} + {S2_YOLO_MODEL}")
    print(f"tau = {stats.get('tau')}")
    if input_clip and input_clip.is_file():
        print("Input clip (original):", input_clip)
    print("Sampled video:", out_vid)
    if debug_path and debug_path.is_file():
        print("Debug video:", debug_path)
    print("Report:", report)
    print("Manifest:", manifest)

    if args.open:
        if input_clip and input_clip.is_file():
            _open_with_default_app(input_clip)
        _open_with_default_app(out_vid)
        if debug_path and debug_path.is_file():
            _open_with_default_app(debug_path)


if __name__ == "__main__":
    main()
