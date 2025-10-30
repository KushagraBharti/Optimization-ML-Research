from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from coverage_planning.algs.geometry import tour_length
from coverage_planning.common import constants

EPS = constants.EPS_GEOM
TOL = constants.TOL_NUM

Interval = Tuple[float, float]


@dataclass(frozen=True)
class IntervalCoverage:
    """Snapshot of a coverage interval and indices it spans."""

    start_index: int
    end_index: int
    intervals: Tuple[Interval, ...]


def normalize_intervals(intervals: Iterable[Interval], *, tol: float = TOL) -> List[Interval]:
    """Return canonical, sorted, non-overlapping intervals."""
    sorted_parts = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged: List[Interval] = []
    for a, b in sorted_parts:
        if not merged:
            merged.append((a, b))
            continue
        prev_a, prev_b = merged[-1]
        if a <= prev_b + tol:
            merged[-1] = (prev_a, max(prev_b, b))
        else:
            merged.append((a, b))
    return merged


def subtract_interval(
    intervals: Sequence[Interval],
    cover: Interval,
    *,
    tol: float = TOL,
) -> List[Interval]:
    """Subtract ``cover`` from ``intervals`` and return remaining pieces."""
    p, q = cover
    lo = min(p, q)
    hi = max(p, q)
    if hi <= lo + tol:
        return list(intervals)
    remainder: List[Interval] = []
    for a, b in intervals:
        if b <= lo + tol or a >= hi - tol:
            remainder.append((a, b))
            continue
        if a < lo - tol:
            remainder.append((a, min(lo, b)))
        if b > hi + tol:
            remainder.append((max(hi, a), b))
    return normalize_intervals(remainder, tol=tol)


def intervals_span(intervals: Sequence[Interval], *, tol: float = TOL) -> Interval:
    if not intervals:
        raise ValueError("interval list is empty")
    a = min(seg[0] for seg in intervals)
    b = max(seg[1] for seg in intervals)
    if b <= a + tol:
        raise ValueError("interval span collapses under tolerance")
    return a, b


def locate_interval(
    intervals: Sequence[Interval],
    x: float,
    *,
    tol: float = TOL,
) -> int:
    """Return index of interval containing ``x`` within tolerance, else -1."""
    for idx, (a, b) in enumerate(intervals):
        if a - tol <= x <= b + tol:
            return idx
    return -1


def contiguous_cover(
    intervals: Sequence[Interval],
    p: float,
    q: float,
    *,
    tol: float = TOL,
) -> IntervalCoverage | None:
    """Return indices covered by [p, q] when coverage is contiguous."""
    if not intervals:
        return None
    lo, hi = (p, q) if p <= q else (q, p)
    start_idx = locate_interval(intervals, lo, tol=tol)
    end_idx = locate_interval(intervals, hi, tol=tol)
    if start_idx == -1 or end_idx == -1:
        return None
    for idx in range(start_idx, end_idx):
        _, right = intervals[idx]
        nxt_left, _ = intervals[idx + 1]
        if right < nxt_left - tol:
            return None
    sliced = tuple(intervals[idx] for idx in range(start_idx, end_idx + 1))
    return IntervalCoverage(start_idx=start_idx, end_index=end_idx, intervals=sliced)


def covers_contiguously(
    intervals: Sequence[Interval],
    p: float,
    q: float,
    *,
    tol: float = TOL,
) -> bool:
    return contiguous_cover(intervals, p, q, tol=tol) is not None


def is_within_budget(p: float, q: float, h: float, L: float, *, tol: float = TOL) -> bool:
    return tour_length(p, q, h) <= L + tol


def is_maximal_length(p: float, q: float, h: float, L: float, *, tol: float = TOL) -> bool:
    return abs(tour_length(p, q, h) - L) <= tol


def snap_index(
    value: float,
    candidates: Sequence[float],
    *,
    tol: float,
) -> int:
    """Return index of ``value`` in ``candidates`` within tolerance."""
    for idx, candidate in enumerate(candidates):
        if abs(candidate - value) <= tol:
            return idx
    raise ValueError(f"value {value:.12f} not found within tolerance {tol:.2e}")


def equal_within(x: float, y: float, *, tol: float = TOL) -> bool:
    return abs(x - y) <= tol


__all__ = [
    "EPS",
    "TOL",
    "Interval",
    "IntervalCoverage",
    "normalize_intervals",
    "subtract_interval",
    "intervals_span",
    "locate_interval",
    "contiguous_cover",
    "covers_contiguously",
    "is_within_budget",
    "is_maximal_length",
    "snap_index",
    "equal_within",
]
