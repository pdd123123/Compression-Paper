"""Export / encode H.264 for upload baseline and .seccomp annex (same pipeline settings)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def encode_h264_mp4(
    input_path: str | Path,
    output_path: str | Path | None,
    max_frames: int,
    max_height: int = 720,
    crf: int = 28,
    fps: float | None = None,
    preset: str = "fast",
    faststart: bool = False,
) -> bytes:
    """Encode clip to H.264 MP4. Returns file bytes if output_path is None."""
    inp = Path(input_path)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(inp)]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    if max_height and max_height > 0:
        cmd += ["-vf", f"scale=-2:{int(max_height)}"]
    cmd += [
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-fps_mode",
        "cfr",
    ]
    if fps and fps > 1:
        cmd += ["-r", f"{float(fps):.6f}"]
    if faststart:
        cmd += ["-movflags", "+faststart"]

    if output_path is None:
        fd, name = tempfile.mkstemp(suffix=".mp4")
        import os

        os.close(fd)
        out = Path(name)
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

    cmd.append(str(out))
    subprocess.run(cmd, check=True)
    data = out.read_bytes()
    if output_path is None:
        try:
            out.unlink()
        except OSError:
            pass
    return data


def export_original_mp4(
    input_path: str | Path,
    output_path: str | Path,
    max_frames: int,
    max_height: int = 720,
    crf: int = 28,
    fps: float | None = None,
) -> Path:
    """Baseline 'user uploads this MP4' (faststart for normal playback)."""
    encode_h264_mp4(
        input_path,
        output_path,
        max_frames,
        max_height=max_height,
        crf=crf,
        fps=fps,
        preset="fast",
        faststart=True,
    )
    return Path(output_path)


def file_size_mb(path: Path) -> float | None:
    return round(path.stat().st_size / (1024 * 1024), 3) if path.is_file() else None


def attach_comparison_videos(
    stem: str,
    out_dir: Path,
    input_video: Path,
    debug_mp4: Path,
    max_frames: int,
    cfg: dict,
    export_original: bool = True,
) -> dict[str, str]:
    _ = stem
    out_dir.mkdir(parents=True, exist_ok=True)
    original = out_dir / "original.mp4"
    debug = out_dir / "debug.mp4"
    if debug_mp4.resolve() != debug.resolve() and debug_mp4.is_file():
        shutil.copy2(debug_mp4, debug)

    max_h = int(cfg.get("processing", {}).get("max_height", 720) or 720)
    crf = int(
        cfg.get("delivery", {}).get("h264_crf")
        or cfg.get("decompress", {}).get("ffmpeg_crf", 28)
        or 28
    )
    if export_original and not original.is_file():
        export_original_mp4(input_video, original, max_frames, max_h, crf)

    info: dict[str, str] = {
        "debug_mp4": str(debug if debug.is_file() else debug_mp4),
        "original_mp4": str(original),
    }
    for key, p in (("debug_mp4", Path(info["debug_mp4"])), ("original_mp4", original)):
        mb = file_size_mb(p)
        if mb is not None:
            info[f"{key.replace('_mp4', '')}_mb"] = str(mb)
    orig_mb = file_size_mb(original)
    dbg_mb = file_size_mb(Path(info["debug_mp4"]))
    if orig_mb and dbg_mb:
        info["debug_pct_of_original_mp4"] = str(round(100 * dbg_mb / orig_mb, 1))
    return info
