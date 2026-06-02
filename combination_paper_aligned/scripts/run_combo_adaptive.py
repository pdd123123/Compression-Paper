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
from skip_v4 import apply_v4_adaptive_policy, enrich_adaptive_skip_stats


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptive v4 combo")
    add_common_args(p)
    p.add_argument("--min-retention", type=float, default=None)
    p.add_argument("--max-retention", type=float, default=None)
    p.add_argument("--target-retention", type=float, default=None)
    args = p.parse_args()

    try:
        run_combo_pipeline(
            args,
            skip_mode="adaptive_v4",
            configure_skip=apply_v4_adaptive_policy,
            default_stem_suffix="adaptive4",
            enrich_skip_stats=enrich_adaptive_skip_stats,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
