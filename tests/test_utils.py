from __future__ import annotations

import itertools
import math
import random
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

try:
    from coverage_planning.algs.geometry import EPS, find_maximal_p, tour_length
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.algs.geometry import EPS, find_maximal_p, tour_length


__all__ = [
    "rng",
    "gen_disjoint_segments",
    "gen_one_side_segments",
    "gen_single_segment",
    "tour_len_sum",
    "check_tours_feasible",
    "check_cover_exact",
    "check_disjoint",
    "oracle_min_tours_gs",
    "oracle_min_length_one_segment",
    "oracle_min_length_one_side",
    "oracle_min_length_full_line",
    "reflect_segments",
    "scale_instance",
    "scale_tuple_list",
]


TOL = 1e-6


# ---------------------------------------------------------------------------
#  Random generators
# ---------------------------------------------------------------------------
def rng(seed: int) -> random.Random:
    return random.Random(seed)


def _distribute_positive(parts: int, base: float, extra: float, rnd: random.Random) -> List[float]:
    if parts <= 0:
        return []
    weights = [rnd.random() for _ in range(parts)]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return [base + extra / parts] * parts
    return [base + (w / total_weight) * extra for w in weights]


def gen_disjoint_segments(
    rnd: random.Random,
    k: int,
    x_min: float,
    x_max: float,
    min_len: float,
    min_gap: float,
) -> List[Tuple[float, float]]:
    if k <= 0:
        raise ValueError("k must be positive")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")
    base_total = k * min_len + (k + 1) * min_gap
    span = x_max - x_min
    if span < base_total - 1e-9:
        raise ValueError("Interval too small for the requested segments")

    extra_space = span - base_total
    lengths = _distribute_positive(k, min_len, extra_space * 0.6, rnd)
    gaps = _distribute_positive(k + 1, min_gap, extra_space * 0.4, rnd)

    segments: List[Tuple[float, float]] = []
    cursor = x_min + gaps[0]
    for idx in range(k):
        start = cursor
        end = start + lengths[idx]
        segments.append((start, end))
        cursor = end + gaps[idx + 1]
    return segments


def gen_one_side_segments(
    rnd: random.Random,
    k: int,
    x_lo: float,
    x_hi: float,
    min_len: float,
    min_gap: float,
) -> List[Tuple[float, float]]:
    if x_lo < -EPS:
        raise ValueError("x_lo must be non-negative for one-sided generation")
    segs = gen_disjoint_segments(rnd, k, x_lo, x_hi, min_len, min_gap)
    return segs


def gen_single_segment(
    rnd: random.Random,
    x_lo: float,
    x_hi: float,
    min_len: float,
) -> Tuple[float, float]:
    if x_hi <= x_lo + min_len:
        raise ValueError("Interval too small for single segment")
    length = rnd.uniform(min_len, (x_hi - x_lo) * 0.9)
    start = rnd.uniform(x_lo, x_hi - length)
    return start, start + length


# ---------------------------------------------------------------------------
#  Basic helpers
# ---------------------------------------------------------------------------
def tour_len_sum(h: float, tours: Iterable[Tuple[float, float]]) -> float:
    return sum(tour_length(min(p, q), max(p, q), h) for p, q in tours)


def check_tours_feasible(
    h: float,
    L: float,
    tours: Sequence[Tuple[float, float]],
    *,
    tol: float = TOL,
) -> None:
    for idx, (p, q) in enumerate(tours):
        length = tour_length(min(p, q), max(p, q), h)
        if length > L + tol:
            raise AssertionError(
                f"Tour #{idx + 1} length {length:.12f} exceeds limit {L:.12f}"
            )


def check_cover_exact(
    segments: Sequence[Tuple[float, float]],
    tours: Sequence[Tuple[float, float]],
    *,
    tol: float = 1e-7,
) -> None:
    for a, b in segments:
        pieces = []
        for p, q in tours:
            lo, hi = (p, q) if p <= q else (q, p)
            if hi < a - tol or lo > b + tol:
                continue
            pieces.append((max(lo, a), min(hi, b)))
        if not pieces:
            raise AssertionError(f"Segment [{a}, {b}] not covered")
        pieces.sort()
        coverage = pieces[0][0]
        if coverage > a + tol:
            raise AssertionError(f"Gap before {a} (coverage starts at {coverage})")
        coverage = pieces[0][1]
        for start, end in pieces[1:]:
            if start > coverage + tol:
                raise AssertionError("Gap detected within segment coverage")
            coverage = max(coverage, end)
        if coverage < b - tol:
            raise AssertionError(f"Segment [{a}, {b}] not fully covered (ends at {coverage})")


def check_disjoint(
    segments: Sequence[Tuple[float, float]],
    *,
    tol: float = 1e-12,
) -> None:
    if not segments:
        return
    prev_end = segments[0][1]
    if segments[0][1] <= segments[0][0] + tol:
        raise AssertionError("Segments must have positive length")
    for idx in range(1, len(segments)):
        a, b = segments[idx]
        if b <= a + tol:
            raise AssertionError("Segments must have positive length")
        if prev_end >= a - tol:
            raise AssertionError("Segments are not pairwise disjoint")
        prev_end = b


# ---------------------------------------------------------------------------
#  Tiny oracles
# ---------------------------------------------------------------------------
def _candidate_points_one_side(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    points = {0.0}
    for a, b in segments:
        points.add(max(0.0, a))
        points.add(max(0.0, b))
        try:
            p = find_maximal_p(b, h, L)
            if p <= b + 1e-9 and p >= -1e-9:
                points.add(max(0.0, p))
        except ValueError:
            continue
    return sorted(points)


def _candidate_points_full(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    points = {0.0}
    for a, b in segments:
        points.add(a)
        points.add(b)
        for endpoint in (a, b):
            try:
                p = find_maximal_p(endpoint, h, L)
                if p <= endpoint + 1e-9:
                    points.add(p)
            except ValueError:
                continue
    return sorted(points)


def oracle_min_tours_gs(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> int:
    if len(segments) > 3:
        raise ValueError("oracle_min_tours_gs supports at most three segments")
    check_disjoint(sorted(segments, key=lambda s: s[0]))
    points = _candidate_points_full(segments, h, L)
    candidates: List[Tuple[float, float]] = []
    for p in points:
        for q in points:
            if q < p - EPS:
                continue
            length = tour_length(p, q, h)
            if length <= L + 1e-9:
                candidates.append((min(p, q), max(p, q)))

    if not candidates:
        raise ValueError("Instance infeasible under L")

    segments_sorted = sorted(segments, key=lambda s: s[0])
    for r in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, r):
            ok = True
            for a, b in segments_sorted:
                covered = any(
                    p - 1e-9 <= a <= q + 1e-9 and p - 1e-9 <= b <= q + 1e-9
                    for p, q in combo
                )
                if not covered:
                    ok = False
                    break
            if ok:
                return r
    raise ValueError("No feasible cover discovered")


def oracle_min_length_one_segment(
    seg: Tuple[float, float],
    h: float,
    L: float,
) -> float:
    a, b = sorted(seg)
    direct = tour_length(a, b, h)
    if direct <= L + 1e-9:
        return direct
    left = tour_length(a, 0.0, h)
    right = tour_length(0.0, b, h)
    if left <= L + 1e-9 and right <= L + 1e-9:
        return left + right
    raise ValueError("Segment infeasible under L")


def oracle_min_length_one_side(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> float:
    if len(segments) > 3:
        raise ValueError("oracle_min_length_one_side supports at most three segments")
    if any(a < -EPS for a, _ in segments):
        raise ValueError("Segments must satisfy x >= 0")
    segments_sorted = sorted(segments, key=lambda s: s[0])
    check_disjoint(segments_sorted)

    points = _candidate_points_one_side(segments_sorted, h, L)
    candidates = []
    for p in points:
        for q in points:
            if q < p - 1e-9:
                continue
            length = tour_length(p, q, h)
            if length > L + 1e-9:
                continue
            covered = [
                idx
                for idx, (a, b) in enumerate(segments_sorted)
                if p - 1e-9 <= a and q + 1e-9 >= b
            ]
            if not covered:
                continue
            if covered != list(range(covered[0], covered[-1] + 1)):
                continue
            candidates.append((p, q, tuple(covered), length))

    if not candidates:
        raise ValueError("No feasible tour candidates under L")

    @lru_cache(maxsize=None)
    def best_from(idx: int) -> float:
        if idx >= len(segments_sorted):
            return 0.0
        best = math.inf
        prev_end = segments_sorted[idx - 1][1] if idx > 0 else None
        for p, q, covered, length in candidates:
            if covered[0] != idx:
                continue
            if idx > 0 and p < prev_end - 1e-9:
                continue
            rightmost = covered[-1]
            if rightmost + 1 < len(segments_sorted):
                next_left = segments_sorted[rightmost + 1][0]
                if q > next_left + 1e-9:
                    continue
            best = min(best, length + best_from(rightmost + 1))
        if math.isinf(best):
            raise ValueError("No feasible cover from index")
        return best

    return best_from(0)


def oracle_min_length_full_line(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> float:
    if not segments:
        return 0.0
    if len(segments) > 3:
        raise ValueError("oracle_min_length_full_line supports at most three segments")
    segments_sorted = sorted(segments, key=lambda s: s[0])
    check_disjoint(segments_sorted)

    points = _candidate_points_full(segments_sorted, h, L)
    candidates = []
    for p in points:
        for q in points:
            if q < p - 1e-9:
                continue
            length = tour_length(p, q, h)
            if length > L + 1e-9:
                continue
            covered = [
                idx
                for idx, (a, b) in enumerate(segments_sorted)
                if p - 1e-9 <= a and q + 1e-9 >= b
            ]
            if not covered:
                continue
            if covered != list(range(covered[0], covered[-1] + 1)):
                continue
            is_crossing = p < -1e-9 and q > 1e-9
            candidates.append((p, q, tuple(covered), length, is_crossing))

    if not candidates:
        raise ValueError("No feasible tours under L")

    @lru_cache(maxsize=None)
    def best_from(idx: int, used_cross: bool) -> float:
        if idx >= len(segments_sorted):
            return 0.0
        best = math.inf
        prev_end = segments_sorted[idx - 1][1] if idx > 0 else None
        for p, q, covered, length, is_cross in candidates:
            if covered[0] != idx:
                continue
            if idx > 0 and p < prev_end - 1e-9:
                continue
            if is_cross and used_cross:
                continue
            rightmost = covered[-1]
            if rightmost + 1 < len(segments_sorted):
                next_left = segments_sorted[rightmost + 1][0]
                if q > next_left + 1e-9:
                    continue
            best = min(best, length + best_from(rightmost + 1, used_cross or is_cross))
        if math.isinf(best):
            raise ValueError("No feasible cover from index")
        return best

    return best_from(0, False)


# ---------------------------------------------------------------------------
#  Reflection & scaling helpers
# ---------------------------------------------------------------------------
def reflect_segments(segments: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    reflected = [(-b, -a) for a, b in segments]
    return sorted(reflected, key=lambda s: s[0])


def scale_instance(
    segments: Sequence[Tuple[float, float]],
    s: float,
) -> List[Tuple[float, float]]:
    if s <= 0:
        raise ValueError("Scaling factor must be positive")
    return [(a * s, b * s) for a, b in segments]


def scale_tuple_list(
    tours: Sequence[Tuple[float, float]],
    s: float,
) -> List[Tuple[float, float]]:
    if s <= 0:
        raise ValueError("Scaling factor must be positive")
    return [(p * s, q * s) for p, q in tours]
