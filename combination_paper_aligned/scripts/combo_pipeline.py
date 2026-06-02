from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from compress_src.codec import load_bitstream
from compress_src.config import load_config as load_compress_config
from compress_src.pipeline import compress_video, decompress_video
from compress_src.video_export import attach_comparison_videos, file_size_mb
from skip_src.config import load_config as load_skip_config
from skip_src.pipeline import run_semantic_sampling

ROOT = Path(__file__).resolve().parents[1]
SKIP_CONFIG = ROOT / "config" / "skip_default.yaml"
COMPRESS_CONFIG = ROOT / "config" / "compress.yaml"
OUTPUT_DIR = ROOT / "outputs"


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--max-frames", type=int, default=None, help="First N frames only")
    p.add_argument("--stem", default=None)
    p.add_argument("--skip-config", default=str(SKIP_CONFIG))
    p.add_argument("--compress-config", default=str(COMPRESS_CONFIG))
    p.add_argument(
        "--baseline-crf",
        type=int,
        default=28,
        help="H.264 baseline CRF for delivery annex (default 28; +extra_crf from config)",
    )
    p.add_argument("--output-dir", "-o", default=str(OUTPUT_DIR))
    p.add_argument("--debug-video", action="store_true")
    p.add_argument("--final-only", action="store_true", help="No original/debug/input clip")
    p.add_argument("--no-progress", action="store_true")


def run_combo_pipeline(
    args: argparse.Namespace,
    *,
    skip_mode: str,
    configure_skip: Callable[[dict, argparse.Namespace], None],
    default_stem_suffix: str,
    enrich_skip_stats: Callable[[dict, dict], dict] | None = None,
) -> None:
    inp = Path(args.input).resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"input not found: {inp}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.stem or f"{inp.stem}_{default_stem_suffix}"
    skip_cfg = load_skip_config(args.skip_config)
    configure_skip(skip_cfg, args)
    skip_cfg.setdefault("video", {})["max_frames"] = args.max_frames
    if args.final_only:
        skip_cfg.setdefault("performance", {})["skip_input_clip"] = True

    sampled = out_dir / f"{stem}_sampled.mp4"
    skip_manifest = out_dir / f"{stem}_skip_manifest.json"
    skip_debug = out_dir / f"{stem}_skip_debug.mp4" if args.debug_video else None
    frame_label = args.max_frames if args.max_frames is not None else "all"
    input_clip = None
    if not args.final_only:
        input_clip = out_dir / f"{stem}_input_{frame_label}fr.mp4"

    print(f"=== Step 1/3: skip sampling ({skip_mode}) ===")
    skip_stats = run_semantic_sampling(
        inp,
        sampled,
        skip_manifest,
        skip_cfg,
        show_progress=not args.no_progress,
        debug_video=skip_debug,
        input_clip_video=input_clip,
    )
    if enrich_skip_stats is not None:
        skip_stats = enrich_skip_stats(skip_cfg, skip_stats)

    n_kept = int(skip_stats.get("n_kept", 0))
    n_total = int(skip_stats.get("n_total", 0))
    skip_report = out_dir / f"{stem}_skip_report.json"
    skip_report.write_text(json.dumps(skip_stats, indent=2), encoding="utf-8")
    print(f"Kept {n_kept} / {n_total} frames ({100.0 * n_kept / max(n_total, 1):.1f}%) -> {sampled}")

    compress_cfg = load_compress_config(args.compress_config)
    compress_cfg.setdefault("paths", {})["output_dir"] = str(out_dir)
    compress_cfg.setdefault("video", {})["max_frames"] = None
    compress_cfg.setdefault("video", {})["frame_stride"] = 1
    compress_cfg.setdefault("delivery", {})["h264_crf"] = args.baseline_crf

    bitstream = out_dir / f"{stem}.seccomp"
    recon = out_dir / f"{stem}_recon.mp4"
    extra_crf = int(compress_cfg.get("transmit", {}).get("extra_crf", 0) or 0)
    delivery_crf = args.baseline_crf + extra_crf

    print(
        f"=== Step 2/3: compress sampled video "
        f"(paper soft-edge + H.264 CRF {delivery_crf}) ==="
    )
    comp_stats = compress_video(sampled, bitstream, compress_cfg)
    _, _, edge_blobs, annex = load_bitstream(bitstream)
    if not edge_blobs:
        raise RuntimeError("bitstream has no soft-edge frames (expected paper_aligned encode)")
    print(
        f"Bitstream: {bitstream} ({file_size_mb(bitstream)} MB) "
        f"edges={len(edge_blobs)} annex={round(len(annex or b'') / 1024 / 1024, 3)} MB"
    )

    print("=== Step 3/3: decompress ===")
    dec_stats = decompress_video(bitstream, recon, None, compress_cfg)
    if args.final_only:
        compare = {"debug_mp4": str(recon)}
    else:
        compare = attach_comparison_videos(
            stem,
            out_dir,
            sampled,
            recon,
            n_kept or args.max_frames or n_total,
            compress_cfg,
        )

    summary = {
        "input": str(inp),
        "output_dir": str(out_dir),
        "stem": stem,
        "skip_mode": skip_mode,
        "max_frames": args.max_frames,
        "skip": {
            "sampled_mp4": str(sampled),
            "manifest": str(skip_manifest),
            "report": str(skip_report),
            "input_clip": str(input_clip) if input_clip.is_file() else None,
            "debug_mp4": str(skip_debug) if skip_debug and skip_debug.is_file() else None,
            "n_kept": n_kept,
            "n_total": n_total,
            "retention_pct": round(100.0 * n_kept / max(n_total, 1), 2),
            "stats": skip_stats,
        },
        "compression_method": "paper_soft_edge + h264_annex (paper_aligned)",
        "compression": {
            "baseline_crf": args.baseline_crf,
            "delivery_crf": delivery_crf,
            "paper_edge_frames": len(edge_blobs),
            "h264_annex_mb": round(len(annex or b"") / 1024 / 1024, 3),
        },
        "compress": comp_stats,
        "decompress": dec_stats,
        "bitstream": str(bitstream),
        "bitstream_mb": file_size_mb(bitstream),
        "recon_mp4": str(recon),
        "sampled_baseline_mp4": compare.get("original_mp4") if not args.final_only else None,
        "compare": compare,
        "skip_config": args.skip_config,
        "compress_config": args.compress_config,
    }
    combo_report = out_dir / f"{stem}_combo_report.json"
    combo_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Done ===")
    if args.final_only:
        print(f"Final video: {recon}")
        print(f"Report:      {combo_report}")
        return

    print(f"Sampled:   {sampled}")
    print(f"Bitstream: {bitstream}")
    print(f"Recon:     {recon}")
    if summary.get("sampled_baseline_mp4"):
        print(f"Baseline:  {summary['sampled_baseline_mp4']}")
    print(f"Report:    {combo_report}")
