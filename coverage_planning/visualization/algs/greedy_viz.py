"""Instrumented copy of the GS/MinTours greedy reference solver."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM

EPS = EPS_GEOM

__all__ = ["gs"]


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


def _trim_right(uncovered: Deque[Tuple[float, float]], left: float) -> None:
    """Remove or truncate segments covered by a right-ending tour."""
    while uncovered:
        a, b = uncovered[-1]
        if b <= left + EPS:
            break
        if a >= left - EPS:
            uncovered.pop()
        else:
            uncovered[-1] = (a, left)
            break


def _trim_left(uncovered: Deque[Tuple[float, float]], right: float) -> None:
    """Remove or truncate segments covered by a left-ending tour."""
    while uncovered:
        a, b = uncovered[0]
        if a >= right - EPS:
            break
        if b <= right + EPS:
            uncovered.popleft()
        else:
            uncovered[0] = (right, b)
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


def _snapshot(uncovered: Deque[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return a list snapshot of uncovered intervals."""
    return [(float(a), float(b)) for a, b in uncovered]


def gs(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    trace: bool = False,
) -> Tuple[int, List[Tuple[float, float]]] | Tuple[int, List[Tuple[float, float]], List[Dict[str, Any]]]:
    """Instrumented GS/MinTours solver matching the reference output."""
    if not segments:
        result: Tuple[int, List[Tuple[float, float]]] = (0, [])
        if trace:
            return result[0], result[1], []
        return result

    segs = _validate_disjoint(segments)

    farthest = max(max(abs(a), abs(b)) for a, b in segs)
    if 2.0 * math.hypot(farthest, h) > L + EPS:
        raise ValueError("Instance infeasible: battery too small to reach farthest point")

    uncovered: Deque[Tuple[float, float]] = deque(segs)
    tours: List[Tuple[float, float]] = []
    trace_steps: List[Dict[str, Any]] = [] if trace else []
    step_idx = 0

    while uncovered:
        uncovered_before = _snapshot(uncovered)
        leftmost = uncovered[0][0]
        rightmost = uncovered[-1][1]

        if tour_length(leftmost, rightmost, h) <= L + EPS:
            tour = (leftmost, rightmost)
            tours.append(tour)
            uncovered.clear()
            if trace:
                choice = "farthest_right" if abs(rightmost) >= abs(leftmost) else "farthest_left"
                trace_steps.append(
                    {
                        "step_idx": step_idx,
                        "choice": choice,
                        "tour": {"p": float(tour[0]), "q": float(tour[1])},
                        "uncovered_before": uncovered_before,
                        "uncovered_after": [],
                        "diagnostics": {"mode": "full_cover"},
                    }
                )
            break

        prev_left, prev_right = leftmost, rightmost
        right_is_farther = abs(rightmost) >= abs(leftmost)

        if right_is_farther:
            choice = "farthest_right"
            q = rightmost
            try:
                p = _find_maximal_p_safe(q, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest right endpoint became unreachable") from exc
            tour = (min(p, q), max(p, q))
            tours.append(tour)
            _trim_right(uncovered, left=p)
            diag_mode = "maximal_from_q"
        else:
            choice = "farthest_left"
            p = leftmost
            try:
                q = _find_maximal_q_safe(p, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest left endpoint became unreachable") from exc
            tour = (min(p, q), max(p, q))
            tours.append(tour)
            _trim_left(uncovered, right=q)
            diag_mode = "maximal_from_p"

        length = tour_length(tour[0], tour[1], h)
        if not (L - 1e-6 <= length <= L + 1e-6):
            raise RuntimeError("Constructed tour is not maximal within tolerance")

        if uncovered:
            new_left, new_right = uncovered[0][0], uncovered[-1][1]
            if abs(new_left - prev_left) <= EPS and abs(new_right - prev_right) <= EPS:
                raise RuntimeError("GS made no progress")

        if trace:
            trace_steps.append(
                {
                    "step_idx": step_idx,
                    "choice": choice,
                    "tour": {"p": float(tour[0]), "q": float(tour[1])},
                    "uncovered_before": uncovered_before,
                    "uncovered_after": _snapshot(uncovered),
                    "diagnostics": {"mode": diag_mode, "length": float(length)},
                }
            )
        step_idx += 1

    result = (len(tours), [(float(p), float(q)) for p, q in tours])
    if trace:
        return result[0], result[1], trace_steps
    return result

