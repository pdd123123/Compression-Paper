"""Step labels and tqdm wrappers for pipeline progress."""

from __future__ import annotations

import time
from typing import Any, Iterable, Iterator

from tqdm import tqdm


def estimate_processed_frames(video_cfg: dict, meta: dict) -> int | None:
    """How many frames will be scored after stride / max_frames limits."""
    stride = max(1, int(video_cfg.get("frame_stride", 1)))
    max_frames = video_cfg.get("max_frames")
    if max_frames is not None:
        return int(max_frames)
    fc = int(meta.get("frame_count") or 0)
    if fc <= 0:
        return None
    return (fc + stride - 1) // stride


class PipelineProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.labels: list[str] = []
        self.step = 0
        self.total_steps = 0
        self._run_t0 = time.perf_counter()
        self._step_t0 = 0.0

    def set_plan(self, labels: list[str]) -> None:
        self.labels = labels
        self.total_steps = len(labels)
        self.step = 0

    def begin(self, label: str | None = None) -> None:
        self.step += 1
        if label is None and self.step <= len(self.labels):
            label = self.labels[self.step - 1]
        label = label or f"Step {self.step}"
        self._step_t0 = time.perf_counter()
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._run_t0
        print(
            f"\n[{self.step}/{self.total_steps}] {label}  "
            f"(total elapsed {elapsed:.0f}s)",
            flush=True,
        )

    def done(self, detail: str = "") -> None:
        if not self.enabled:
            return
        dt = time.perf_counter() - self._step_t0
        suffix = f" — {detail}" if detail else ""
        print(f"      finished in {dt:.1f}s{suffix}", flush=True)

    def note(self, msg: str) -> None:
        if self.enabled:
            print(f"      {msg}", flush=True)

    def begin_sub(self, label: str, *, total: int | None = None) -> None:
        self._step_t0 = time.perf_counter()
        if not self.enabled:
            return
        extra = f", {total} frames" if total is not None else ""
        print(f"      · {label}{extra}", flush=True)

    def done_sub(self, detail: str = "") -> None:
        if not self.enabled:
            return
        dt = time.perf_counter() - self._step_t0
        suffix = f" — {detail}" if detail else ""
        print(f"        done in {dt:.1f}s{suffix}", flush=True)

    def iter(
        self,
        iterable: Iterable[Any],
        *,
        total: int | None = None,
        desc: str | None = None,
        unit: str = "fr",
    ) -> Iterator[Any]:
        if not self.enabled:
            yield from iterable
            return
        label = desc or "progress"
        bar_desc = f"  {label}"
        yield from tqdm(iterable, total=total, desc=bar_desc, unit=unit, leave=True)

    def finish_run(self) -> None:
        if not self.enabled:
            return
        total = time.perf_counter() - self._run_t0
        print(f"\nAll steps done in {total:.1f}s ({total / 60:.1f} min)", flush=True)
