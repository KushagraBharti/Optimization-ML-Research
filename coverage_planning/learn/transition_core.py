from __future__ import annotations

from typing import List, Tuple

from coverage_planning.algs.geometry import find_maximal_p
from coverage_planning.common.constants import EPS_GEOM as EPS, TOL_NUM as TOL

Interval = Tuple[float, float]

__all__ = [
    "normalize_intervals",
    "classify_x",
    "covers_no_gap",
    "is_maximal_pair",
    "trim_covered",
]


def normalize_intervals(intervals: List[Interval], eps: float = EPS) -> List[Interval]:
    """Merge and sort intervals; used by DP transition enumerators and featurization."""
    if not intervals:
        return []
    canonical: List[Interval] = []
    prepared = [(min(a, b), max(a, b)) for a, b in intervals]
    prepared.sort(key=lambda seg: seg[0])
    start, end = prepared[0]
    for a, b in prepared[1:]:
        if a <= end + eps:
            end = max(end, b)
        else:
            canonical.append((start, end))
            start, end = a, b
    canonical.append((start, end))
    return canonical


def classify_x(x: float, intervals: List[Interval], eps: float = EPS) -> Tuple[str, int | None]:
    """Classify location of x relative to disjoint coverage intervals; solver-tolerant."""
    if not intervals:
        raise ValueError("classify_x requires at least one interval")
    first_left = intervals[0][0]
    if x < first_left - eps:
        return "before", None
    for idx, (a, b) in enumerate(intervals):
        if a - eps <= x <= b + eps:
            return "inseg", idx
        if idx + 1 < len(intervals):
            next_left = intervals[idx + 1][0]
            # Within tolerance of the next interval boundary should count as inside.
            if next_left - eps <= x <= next_left + eps:
                return "inseg", idx + 1
            if b + eps < x < next_left - eps:
                return "gap", idx
    last_right = intervals[-1][1]
    if x > last_right + eps:
        return "beyond", None
    raise ValueError("classify_x received inconsistent interval ordering or tolerance")


def covers_no_gap(p: float, q: float, intervals: List[Interval], eps: float = EPS) -> bool:
    """Return True when [p,q] lies within contiguous intervals without gaps."""
    if not intervals:
        return False
    lo, hi = (p, q) if p <= q else (q, p)
    kind_lo, idx_lo = classify_x(lo, intervals, eps=eps)
    kind_hi, idx_hi = classify_x(hi, intervals, eps=eps)
    if kind_lo != "inseg" or kind_hi != "inseg":
        return False
    assert idx_lo is not None and idx_hi is not None
    if idx_hi < idx_lo:
        return False
    if idx_lo == idx_hi:
        return True
    for idx in range(idx_lo, idx_hi):
        current_right = intervals[idx][1]
        next_left = intervals[idx + 1][0]
        if current_right < next_left - eps:
            return False
    return True


def is_maximal_pair(p: float, q: float, h: float, L: float, tol: float = 1e-7) -> bool:
    """Check continuous maximality via geometry helper; must match solver tolerance."""
    p_star = find_maximal_p(q, h, L)
    return abs(p_star - p) <= tol


def trim_covered(intervals: List[Interval], p: float, q: float, eps: float = EPS) -> List[Interval]:
    """Trim intervals covered by [p,q]; shared by DP enumerators and featurization."""
    if not intervals:
        return []
    lo, hi = (p, q) if p <= q else (q, p)
    trimmed: List[Interval] = []
    for a, b in intervals:
        if b <= lo + eps or a >= hi - eps:
            trimmed.append((a, b))
            continue
        left_kept = a < lo - eps
        right_kept = b > hi + eps
        if left_kept:
            left_end = min(lo, b)
            if left_end > a + eps:
                trimmed.append((a, left_end))
        if right_kept:
            right_start = max(hi, a)
            if b > right_start + eps:
                trimmed.append((right_start, b))
    return normalize_intervals(trimmed, eps=eps)
