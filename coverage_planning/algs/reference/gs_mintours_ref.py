"""Reference greedy solver for the minimum-tour coverage objective."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM

EPS = EPS_GEOM

__all__ = ["greedy_min_tours_ref"]


def _validate_disjoint(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return segments sorted by left endpoint after validating shape and gaps."""
    segs = sorted(segments, key=lambda s: s[0])
    for idx, (a, b) in enumerate(segs):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Segment endpoints must be finite")
        if b <= a:
            raise ValueError("Segments must satisfy a < b")
        if idx > 0 and segs[idx - 1][1] >= a - EPS:
            raise ValueError("Segments must be pair-wise disjoint with positive gaps")
    return segs


def _trim_right(U: Deque[Tuple[float, float]], left: float) -> None:
    """Remove or truncate segments covered by a right-ending tour."""
    while U:
        a, b = U[-1]
        if b <= left + EPS:
            break
        if a >= left - EPS:
            U.pop()
        else:
            U[-1] = (a, left)
            break


def _trim_left(U: Deque[Tuple[float, float]], right: float) -> None:
    """Remove or truncate segments covered by a left-ending tour."""
    while U:
        a, b = U[0]
        if a >= right - EPS:
            break
        if b <= right + EPS:
            U.popleft()
        else:
            U[0] = (right, b)
            break


def _find_maximal_p_safe(q: float, h: float, L: float, *, tol: float = 1e-9) -> float:
    """Return the maximal start p <= q such that the tour hits length L within tol."""
    reach = 2.0 * math.hypot(q, h)
    if reach > L + EPS:
        raise ValueError("No feasible p: degenerate tour exceeds L")

    candidate = find_maximal_p(q, h, L)
    if candidate <= q + EPS:
        length = tour_length(candidate, q, h)
        if abs(length - L) <= 1e-7:
            return candidate

    hi = q
    if tour_length(hi, q, h) > L + 1e-9:
        raise ValueError("No feasible p: numerical inconsistency at hi")

    span = max(1.0, 2.0 * L)
    lo = hi - span
    attempts = 0
    while tour_length(lo, q, h) <= L:
        span *= 2.0
        lo = hi - span
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Failed to bracket maximal p")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        length = tour_length(mid, q, h)
        if length > L:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break

    length_hi = tour_length(hi, q, h)
    if not (L - 1e-6 <= length_hi <= L + 1e-6):
        raise RuntimeError("Maximal p bisection failed to reach tolerance")
    if hi > q + 1e-9:
        raise RuntimeError("Computed p exceeds q")
    return hi


def _find_maximal_q_safe(p: float, h: float, L: float, *, tol: float = 1e-9) -> float:
    """Return the maximal end q >= p such that the tour hits length L within tol."""
    reach = 2.0 * math.hypot(p, h)
    if reach > L + EPS:
        raise ValueError("No feasible q: degenerate tour exceeds L")

    base = math.hypot(p, h)
    K_prime = L - base + p
    if K_prime > 0.0:
        q_candidate = (K_prime * K_prime - h * h) / (2.0 * K_prime)
        if q_candidate >= p - EPS:
            length = tour_length(p, q_candidate, h)
            if abs(length - L) <= 1e-7:
                return q_candidate

    lo = p
    if tour_length(p, lo, h) > L + 1e-9:
        raise ValueError("No feasible q: numerical inconsistency at lo")

    hi = p + max(1.0, 2.0 * L)
    attempts = 0
    while tour_length(p, hi, h) < L - 1e-9:
        hi = p + (hi - p) * 2.0
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Failed to bracket maximal q")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        length = tour_length(p, mid, h)
        if length < L:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break

    length_hi = tour_length(p, hi, h)
    if not (L - 1e-6 <= length_hi <= L + 1e-6):
        raise RuntimeError("Maximal q bisection failed to reach tolerance")
    if hi < p - 1e-9:
        raise RuntimeError("Computed q precedes p")
    return hi


def greedy_min_tours_ref(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> Tuple[int, List[Tuple[float, float]]]:
    """Paper-faithful Algorithm 1 (GS) for MinTours on disjoint segments."""
    if not segments:
        return 0, []

    segs = _validate_disjoint(segments)

    farthest = max(max(abs(a), abs(b)) for a, b in segs)
    if 2.0 * math.hypot(farthest, h) > L + EPS:
        raise ValueError(
            "Instance infeasible: battery too small to reach farthest point"
        )

    uncovered: Deque[Tuple[float, float]] = deque(segs)
    tours: List[Tuple[float, float]] = []

    while uncovered:
        leftmost = uncovered[0][0]
        rightmost = uncovered[-1][1]
        if tour_length(leftmost, rightmost, h) <= L + EPS:
            tours.append((leftmost, rightmost))
            uncovered.clear()
            break

        prev_left, prev_right = leftmost, rightmost
        right_is_farther = abs(rightmost) >= abs(leftmost)

        if right_is_farther:
            q = rightmost
            try:
                p = _find_maximal_p_safe(q, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest right endpoint became unreachable") from exc
            tour = (min(p, q), max(p, q))
            tours.append(tour)
            _trim_right(uncovered, left=p)
        else:
            p = leftmost
            try:
                q = _find_maximal_q_safe(p, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest left endpoint became unreachable") from exc
            tour = (min(p, q), max(p, q))
            tours.append(tour)
            _trim_left(uncovered, right=q)

        length = tour_length(tour[0], tour[1], h)
        if not (L - 1e-6 <= length <= L + 1e-6):
            raise RuntimeError("Constructed tour is not maximal within tolerance")

        if uncovered:
            new_left, new_right = uncovered[0][0], uncovered[-1][1]
            if abs(new_left - prev_left) <= EPS and abs(new_right - prev_right) <= EPS:
                raise RuntimeError("GS made no progress")
        else:
            # All material covered; progress guaranteed.
            break

    return len(tours), tours


if __name__ == "__main__":
    # Basic sanity checks
    result = greedy_min_tours_ref([(1.0, 2.0), (5.0, 6.0)], h=4.0, L=200.0)
    assert result[0] == 1 and abs(result[1][0][0] - 1.0) < 1e-9 and abs(result[1][0][1] - 6.0) < 1e-9

    result = greedy_min_tours_ref([(1.0, 2.0), (5.0, 6.0)], h=1.0, L=13.0)
    assert result[0] >= 1

    result = greedy_min_tours_ref([(-6.0, -5.0), (-3.0, -2.0)], h=2.0, L=40.0)
    assert result[0] == 1

    try:
        greedy_min_tours_ref([(0.0, 1.0)], h=1.0, L=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected infeasible instance to raise")
