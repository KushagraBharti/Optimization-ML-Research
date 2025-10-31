"""Reference greedy solver for minimum-length coverage with a single segment."""

from __future__ import annotations

import math
from typing import List, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM

EPS = EPS_GEOM

__all__ = ["greedy_min_length_one_segment_ref"]


# ---------------------------------------------------------------------------
#  Numerical helpers (maximal tour solvers)
# ---------------------------------------------------------------------------
def _find_maximal_p_safe(q: float, h: float, L: float, *, tol: float = 1e-9) -> float:
    """Return the maximal start p <= q such that the tour length equals L within tol."""
    reach = 2.0 * math.hypot(q, h)
    if reach > L + EPS:
        raise ValueError("No feasible p: degenerate tour exceeds L")

    candidate = find_maximal_p(q, h, L)
    if candidate <= q + EPS:
        length = tour_length(candidate, q, h)
        if abs(length - L) <= 1e-7:
            return candidate

    hi = q
    hi_len = tour_length(hi, q, h)
    if hi_len > L + 1e-9:
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
        if abs(length - L) <= tol:
            hi = mid
            break
        if hi - lo <= tol:
            break

    length_hi = tour_length(hi, q, h)
    if not (L - 1e-6 <= length_hi <= L + 1e-6):
        raise RuntimeError("Maximal p bisection failed to reach tolerance")
    if hi > q + 1e-9:
        raise RuntimeError("Computed p exceeds q")
    return hi


def _find_maximal_q_safe(p: float, h: float, L: float, *, tol: float = 1e-9) -> float:
    """Return the maximal end q >= p such that the tour length equals L within tol."""
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
    lo_len = tour_length(p, lo, h)
    if lo_len > L + 1e-9:
        raise ValueError("No feasible q: numerical inconsistency at lo")

    hi = max(p + 1.0, 2.0 * L)
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
        if abs(length - L) <= tol:
            hi = mid
            break
        if hi - lo <= tol:
            break

    length_hi = tour_length(p, hi, h)
    if not (L - 1e-6 <= length_hi <= L + 1e-6):
        raise RuntimeError("Maximal q bisection failed to reach tolerance")
    if hi < p - 1e-9:
        raise RuntimeError("Computed q precedes p")
    return hi


# ---------------------------------------------------------------------------
#  Utility predicates
# ---------------------------------------------------------------------------
def _includes_origin(left: float, right: float) -> bool:
    return left <= 0.0 <= right


def _farther_is_right(left: float, right: float) -> bool:
    return abs(right) >= abs(left)


# ---------------------------------------------------------------------------
#  Main Algorithm 2 (GSP)
# ---------------------------------------------------------------------------
def greedy_min_length_one_segment_ref(
    seg: Tuple[float, float],
    h: float,
    L: float,
) -> Tuple[int, List[Tuple[float, float]]]:
    """Paper-faithful GSP for a single segment on y = h, base at O."""
    a, b = sorted(seg)
    if b - a <= EPS:
        raise ValueError("Segment must satisfy a < b with positive measure")

    farthest = max(abs(a), abs(b))
    if 2.0 * math.hypot(farthest, h) > L + EPS:
        raise ValueError("Infeasible: battery too small to reach farthest endpoint")

    left, right = a, b
    tours: List[Tuple[float, float]] = []

    while True:
        if right - left <= EPS:
            break

        if tour_length(left, right, h) <= L + EPS:
            tours.append((left, right))
            break

        if _includes_origin(left, right):
            side_left = tour_length(left, 0.0, h)
            side_right = tour_length(0.0, right, h)
            if side_left <= L + EPS and side_right <= L + EPS:
                tours.append((left, 0.0))
                tours.append((0.0, right))
                break
            raise ValueError("Infeasible central finishing under L")

        prev_left, prev_right = left, right

        if _farther_is_right(left, right):
            q = right
            try:
                p = _find_maximal_p_safe(q, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest right endpoint became unreachable") from exc
            p = min(p, q)
            tour = (p, q)
            tours.append(tour)
            right = min(right, p)
        else:
            p = left
            try:
                q = _find_maximal_q_safe(p, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest left endpoint became unreachable") from exc
            q = max(q, p)
            tour = (p, q)
            tours.append(tour)
            left = max(left, q)

        tour_len = tour_length(tour[0], tour[1], h)
        if not (L - 1e-6 <= tour_len <= L + 1e-6):
            raise RuntimeError("Constructed tour is not maximal within tolerance")

        if abs(left - prev_left) <= EPS and abs(right - prev_right) <= EPS:
            raise RuntimeError("GSP made no progress under L")

        if right - left <= EPS:
            break

    return len(tours), tours


if __name__ == "__main__":
    def _approx(x: float, y: float, tol: float = 1e-6) -> bool:
        return abs(x - y) <= tol

    # Smoke test 1: symmetric segment, large budget -> single tour
    k, tours = greedy_min_length_one_segment_ref((-2.0, 2.0), h=4.0, L=200.0)
    assert k == 1 and _approx(tours[0][0], -2.0) and _approx(tours[0][1], 2.0)

    # Smoke test 2: asymmetric requiring multiple sweeps then central finish
    k, tours = greedy_min_length_one_segment_ref((-8.0, 9.0), h=4.0, L=30.0)
    assert k >= 2 and _approx(tours[-2][1], 0.0) and _approx(tours[-1][0], 0.0)

    # Smoke test 3: one-sided segment requiring multiple sweeps
    k, tours = greedy_min_length_one_segment_ref((5.0, 11.0), h=2.0, L=22.45)
    assert k >= 2 and tours[0][1] <= 11.0 + EPS

    # Smoke test 4: infeasible due to battery
    try:
        greedy_min_length_one_segment_ref((0.0, 1.0), h=1.0, L=0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected infeasible instance to raise")
