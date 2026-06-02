"""Default per-script output subfolders under outputs/."""

from __future__ import annotations

from pathlib import Path

# outputs/sample | compare | evaluate | sweep
SCRIPT_OUTPUT_SUBDIR = {
    "sample": "sample",
    "adaptive": "adaptive",
    "compare": "compare",
    "evaluate": "evaluate",
    "sweep": "sweep",
}


def get_output_dir(
    project_root: Path,
    script_key: str,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Resolve output directory.
    - If output_dir is set: use it (relative paths are under project_root).
    - Else: project_root / outputs / <script_key>/
    """
    if output_dir is not None:
        p = Path(output_dir)
        if not p.is_absolute():
            p = project_root / p
    else:
        sub = SCRIPT_OUTPUT_SUBDIR.get(script_key, script_key)
        p = project_root / "outputs" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
