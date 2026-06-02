from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(path) if path else root / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(root)
    return cfg


def merge_config(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if key.startswith("_"):
            continue
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = merge_config(out[key], val)
        else:
            out[key] = val
    return out


def apply_preset(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    root = Path(cfg.get("_root", Path(__file__).resolve().parents[1]))
    preset_path = root / "config" / f"{name}.yaml"
    if not preset_path.is_file():
        raise FileNotFoundError(f"Config preset not found: {preset_path}")
    overlay = load_config(preset_path)
    merged = merge_config(cfg, overlay)
    merged["_root"] = cfg.get("_root", str(root))
    return merged


def apply_fast_preset(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = apply_preset(cfg, "fast")
    merged.setdefault("performance", {})["fast_mode"] = True
    return merged


def apply_quality_preset(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = apply_preset(cfg, "quality")
    merged.setdefault("performance", {})["fast_mode"] = False
    return merged


def apply_turbo_preset(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = apply_fast_preset(cfg)
    return apply_preset(merged, "turbo")


def apply_light_preset(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = apply_fast_preset(cfg)
    return apply_preset(cfg, "light")
