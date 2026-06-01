"""Output folders for run_sample.py and run_sample_adaptive.py."""

from __future__ import annotations

from pathlib import Path

SCRIPT_OUTPUT_SUBDIR = {
    "sample": "sample",
    "adaptive": "adaptive",
}


def get_output_dir(
    project_root: Path,
    script_key: str,
    output_dir: str | Path | None = None,
) -> Path:
    if output_dir is not None:
        p = Path(output_dir)
        if not p.is_absolute():
            p = project_root / p
    else:
        sub = SCRIPT_OUTPUT_SUBDIR.get(script_key, script_key)
        p = project_root / "outputs" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
