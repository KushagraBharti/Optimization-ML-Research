"""Instrumented copy of the GSP single-segment minimum-length solver."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM

EPS = EPS_GEOM

__all__ = ["gsp"]


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


def _includes_origin(left: float, right: float) -> bool:
    return left <= 0.0 <= right


def _farther_is_right(left: float, right: float) -> bool:
    return abs(right) >= abs(left)


def gsp(
    seg: Tuple[float, float],
    h: float,
    L: float,
    *,
    trace: bool = False,
) -> Tuple[int, List[Tuple[float, float]]] | Tuple[int, List[Tuple[float, float]], Dict[str, Any]]:
    """Instrumented GSP solver for a single segment."""
    a, b = sorted(seg)
    if b - a <= EPS:
        raise ValueError("Segment must satisfy a < b with positive measure")

    farthest = max(abs(a), abs(b))
    if 2.0 * math.hypot(farthest, h) > L + EPS:
        raise ValueError("Infeasible: battery too small to reach farthest endpoint")

    left, right = a, b
    tours: List[Tuple[float, float]] = []
    steps: List[Dict[str, Any]] = []
    step_idx = 0
    case_type = "multi"

    while True:
        if right - left <= EPS:
            break

        current = {"left": float(left), "right": float(right)}

        if tour_length(left, right, h) <= L + EPS:
            tour = (left, right)
            tours.append(tour)
            case_type = "single" if not steps else case_type
            if trace:
                steps.append(
                    {
                        "step_idx": step_idx,
                        "choice": "span_all",
                        "tour": {"p": float(tour[0]), "q": float(tour[1])},
                        "state_before": current,
                        "state_after": {"left": float(tour[1]), "right": float(tour[0])},
                        "diagnostics": {"mode": "full_cover"},
                    }
                )
            break

        if _includes_origin(left, right):
            side_left = tour_length(left, 0.0, h)
            side_right = tour_length(0.0, right, h)
            if side_left <= L + EPS and side_right <= L + EPS:
                case_type = "central"
                tours.append((left, 0.0))
                tours.append((0.0, right))
                if trace:
                    steps.append(
                        {
                            "step_idx": step_idx,
                            "choice": "farthest_left",
                            "tour": {"p": float(left), "q": 0.0},
                            "state_before": current,
                            "state_after": {"left": 0.0, "right": float(right)},
                            "diagnostics": {"mode": "central_split_left"},
                        }
                    )
                    steps.append(
                        {
                            "step_idx": step_idx + 1,
                            "choice": "farthest_right",
                            "tour": {"p": 0.0, "q": float(right)},
                            "state_before": {"left": 0.0, "right": float(right)},
                            "state_after": {"left": 0.0, "right": 0.0},
                            "diagnostics": {"mode": "central_split_right"},
                        }
                    )
                break
            raise ValueError("Infeasible central finishing under L")

        prev_left, prev_right = left, right

        if _farther_is_right(left, right):
            choice = "farthest_right"
            q = right
            try:
                p = _find_maximal_p_safe(q, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest right endpoint became unreachable") from exc
            p = min(p, q)
            tour = (p, q)
            tours.append(tour)
            right = min(right, p)
            diag = {"mode": "maximal_from_q", "length": float(tour_length(*tour, h))}
        else:
            choice = "farthest_left"
            p = left
            try:
                q = _find_maximal_q_safe(p, h, L)
            except ValueError as exc:
                raise RuntimeError("Farthest left endpoint became unreachable") from exc
            q = max(q, p)
            tour = (p, q)
            tours.append(tour)
            left = max(left, q)
            diag = {"mode": "maximal_from_p", "length": float(tour_length(*tour, h))}

        tour_len = tour_length(tour[0], tour[1], h)
        if not (L - 1e-6 <= tour_len <= L + 1e-6):
            raise RuntimeError("Constructed tour is not maximal within tolerance")

        if abs(left - prev_left) <= EPS and abs(right - prev_right) <= EPS:
            raise RuntimeError("GSP made no progress under L")

        if trace:
            steps.append(
                {
                    "step_idx": step_idx,
                    "choice": choice,
                    "tour": {"p": float(tour[0]), "q": float(tour[1])},
                    "state_before": current,
                    "state_after": {"left": float(left), "right": float(right)},
                    "diagnostics": diag,
                }
            )
        step_idx += 1

    result = (len(tours), [(float(p), float(q)) for p, q in tours])
    if trace:
        trace_payload: Dict[str, Any] = {"case": case_type, "steps": steps}
        return result[0], result[1], trace_payload
    return result

