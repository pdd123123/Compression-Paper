#!/usr/bin/env python3
"""Run semantic skip sampling with adaptive retention."""

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
    parser = argparse.ArgumentParser(description="Adaptive semantic skip sampling")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--min-retention", type=float, default=None)
    parser.add_argument("--max-retention", type=float, default=None)
    parser.add_argument("--target-retention", type=float, default=0.35)
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--debug-video", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Slow/high-quality preset (config/quality.yaml)",
    )
    parser.add_argument("--turbo", action="store_true", help="Extra speed preset")
    parser.add_argument("--light", action="store_true", help="Motion-gated semantic scoring")
    parser.add_argument("--no-input-clip", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quality:
        cfg = apply_quality_preset(cfg)
    elif args.light:
        cfg = apply_light_preset(cfg)
    elif args.turbo:
        cfg = apply_turbo_preset(cfg)
    cfg["scoring"]["retention_mode"] = "adaptive"
    if args.max_frames is not None:
        cfg["video"]["max_frames"] = args.max_frames
    if args.tau is not None:
        cfg["scoring"]["tau"] = args.tau
    ad = cfg.setdefault("adaptive", {})
    if args.min_retention is not None:
        ad["min_retention_ratio"] = args.min_retention
    if args.max_retention is not None:
        ad["max_retention_ratio"] = args.max_retention
    ad["soft_target_retention"] = args.target_retention

    inp = Path(args.input)
    if not inp.is_file():
        print("Error: input not found:", inp, file=sys.stderr)
        sys.exit(1)

    stem = inp.stem
    out_dir = get_output_dir(ROOT, "adaptive", args.output_dir)
    out_vid = out_dir / f"{stem}_adaptive_sampled.mp4"
    manifest = out_dir / f"{stem}_adaptive_manifest.json"
    debug_path = out_dir / f"{stem}_adaptive_debug.mp4" if args.debug_video else None

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
        debug_video=debug_path,
        input_clip_video=input_clip,
    )
    n_total = stats["n_total"]
    pct = 100.0 * stats["n_kept"] / max(n_total, 1)
    stats["retention_mode"] = "adaptive"
    stats["actual_retention_pct"] = pct

    report = out_dir / f"{stem}_adaptive_report.json"
    report.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Kept {stats['n_kept']} / {n_total} frames ({pct:.1f}%)")
    print(f"tau = {stats.get('tau')}")
    print("Sampled video:", out_vid)
    print("Report:", report)
    if args.open and out_vid.is_file():
        _open_with_default_app(out_vid)


if __name__ == "__main__":
    main()
