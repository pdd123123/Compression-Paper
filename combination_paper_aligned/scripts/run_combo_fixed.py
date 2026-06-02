#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from combo_pipeline import add_common_args, run_combo_pipeline
from skip_v4 import DEFAULT_FIXED_RETENTION, apply_v4_fixed_policy, enrich_fixed_skip_stats


def _configure_fixed(cfg: dict, args: argparse.Namespace) -> None:
    apply_v4_fixed_policy(cfg, args.retention)


def main() -> None:
    p = argparse.ArgumentParser(description="Fixed retention combo")
    add_common_args(p)
    p.add_argument("--retention", type=float, default=DEFAULT_FIXED_RETENTION)
    args = p.parse_args()

    if not 0.0 < args.retention <= 1.0:
        print("Error: --retention must be in (0, 1]", file=sys.stderr)
        sys.exit(1)

    if args.stem is None:
        pct = int(round(args.retention * 100))
        args.stem = f"{Path(args.input).stem}_fixed{pct}pct"

    try:
        run_combo_pipeline(
            args,
            skip_mode="fixed_v4",
            configure_skip=_configure_fixed,
            default_stem_suffix=f"fixed{int(round(args.retention * 100))}pct",
            enrich_skip_stats=lambda cfg, stats: enrich_fixed_skip_stats(cfg, stats, args.retention),
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
