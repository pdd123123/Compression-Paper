#!/usr/bin/env python3
"""Run semantic skip sampling with a target retention ratio."""

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


def _open_with_default_app(path: Path) -> None:
    p = str(path.resolve())
    if sys.platform == "win32":
        os.startfile(p)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{p}"')
    else:
        os.system(f'xdg-open "{p}"')


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic skip sampling")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", default=None, help="Output MP4 path")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory")
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--retention",
        type=float,
        default=None,
        help="Fraction of frames to keep (e.g. 0.2)",
    )
    parser.add_argument("--tau", type=float, default=None, help="Fixed score threshold")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Preview kept frames")
    parser.add_argument("--debug-video", action="store_true", help="Write labeled debug MP4")
    parser.add_argument("--open", action="store_true", help="Open outputs when done")
    parser.add_argument("--no-input-clip", action="store_true", help="Skip input reference clip")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive retention (same as run_sample_adaptive.py)",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Slow/high-quality: all frames in RAM, YOLO every frame, diverse selection",
    )
    parser.add_argument(
        "--turbo",
        action="store_true",
        help="Extra speed: stride 3, 540p scoring, YOLO every 16 frames (config/turbo.yaml)",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help="Algorithmic fast path: motion gate + uniform context (config/light.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quality:
        cfg = apply_quality_preset(cfg)
    elif args.light:
        cfg = apply_light_preset(cfg)
    elif args.turbo:
        cfg = apply_turbo_preset(cfg)
    if args.max_frames is not None:
        cfg["video"]["max_frames"] = args.max_frames
    if args.adaptive:
        cfg["scoring"]["retention_mode"] = "adaptive"
    if args.retention is not None and not args.adaptive:
        cfg["scoring"]["target_retention_ratio"] = args.retention
    if args.tau is not None:
        cfg["scoring"]["tau"] = args.tau

    inp = Path(args.input)
    if not inp.is_file():
        print("Error: input not found:", inp, file=sys.stderr)
        sys.exit(1)

    stem = inp.stem
    out_dir = get_output_dir(ROOT, "sample", args.output_dir)
    out_vid = Path(args.output) if args.output else out_dir / f"{stem}_sampled.mp4"
    manifest = out_dir / f"{stem}_sample_manifest.json"
    debug_path = out_dir / f"{stem}_debug.mp4" if args.debug_video else None

    max_f = cfg["video"].get("max_frames")
    n_label = max_f if max_f else "all"
    input_clip = None
    skip_clip = cfg.get("performance", {}).get("skip_input_clip", False)
    if not args.no_input_clip and not skip_clip:
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
    report = out_dir / f"{stem}_sample_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(stats, indent=2))
    if input_clip and input_clip.is_file():
        print("Input clip:", input_clip)
    print("Sampled video:", out_vid)
    if debug_path:
        print("Debug video:", debug_path)
    print("Manifest:", manifest)

    if args.open:
        if input_clip and input_clip.is_file():
            _open_with_default_app(input_clip)
        _open_with_default_app(out_vid)
        if debug_path and debug_path.is_file():
            _open_with_default_app(debug_path)


if __name__ == "__main__":
    main()
