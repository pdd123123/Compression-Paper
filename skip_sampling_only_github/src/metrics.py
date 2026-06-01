"""Size / transmission helpers."""

from __future__ import annotations


def transmission_time_sec(size_bytes: float, bitrate_mbps: float = 10.0) -> float:
    """Transfer time at a given bitrate (default 10 Mbps)."""
    return size_bytes * 8 / (bitrate_mbps * 1e6)


def retention_ratio(n_kept: int, n_total: int) -> float:
    return n_kept / n_total if n_total else 0.0


def size_reduction_pct(n_kept: int, n_total: int) -> float:
    if not n_total:
        return 0.0
    return (1.0 - n_kept / n_total) * 100.0
