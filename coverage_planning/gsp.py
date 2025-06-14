import math
from typing import List, Tuple

from .utils import tour_length, find_maximal_p as _closed_form, log, EPS, VERBOSE

__all__ = [
    "greedy_min_length_one_segment",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _maximal_left_endpoint(q: float, h: float, L: float) -> float:
    """Return the **left** endpoint *p* (≤ q) of the maximal‑length tour
    that still ends at *q* and has length exactly *L*.

    We try the closed‑form from the paper; if it degenerates numerically we
    fall back to a tiny binary‑search (guaranteed to converge because the
    length function is strictly decreasing in *p*).  Always returns *p ≤ q*.
    """
    # Attempt the analytical expression first.
    r = math.hypot(q, h)
    K = L - r - q  # see paper (Figure 3 notation)

    if K > EPS:
        denom = 2 * K
        p = (h * h - K * K) / denom
        if p <= q + EPS:  # good – within bounds (allow small overshoot)
            return p

    # Fallback: binary search on p in [min_p, q]
    min_p = -abs(q)  # never need to start further than mirror of q
    lo, hi = min_p, q
    for _ in range(60):
        mid = (lo + hi) / 2
        length = tour_length(mid, q, h)
        if length > L:
            lo = mid  # too long → move right (shrink interval)
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# Main algorithm: Greedy with Projection Point (GSP)
# ---------------------------------------------------------------------------

def greedy_min_length_one_segment(
    seg: Tuple[float, float],
    h: float,
    L: float,
) -> Tuple[int, List[Tuple[float, float]]]:
    """Optimal **MinLength** algorithm for **one horizontal segment**.

    Implements the Greedy‑with‑Projection algorithm (GSP) from the paper:

    1. Repeatedly launch *maximal* tours toward the farthest uncovered point
       until the residual interval straddles the projection point `O'`.
    2. Finish by either a single tour (if it fits) **or** two symmetric tours
       through `O'`, whichever is feasible and shorter.

    Returns ``(k, tours)`` where *k* is the exact number of tours and *tours*
    is a list of ``(p, q)`` pairs with *p ≤ q*, each interval lying inside
    the original segment and each tour’s length ≤ *L* + EPS.
    """
    a, b = seg
    if a > b:
        a, b = b, a  # normalise

    left, right = a, b  # current uncovered interval endpoints
    tours: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Greedy phase: carve off maximal tours from the *farther* side.
    # ------------------------------------------------------------------
    while True:
        # 1. Can one tour finish everything?
        if tour_length(left, right, h) <= L + EPS:
            tours.append((left, right))
            break

        # 2. Can two symmetric tours through O' finish?
        span = max(-left, right)  # required half‑length to cover both ends
        if tour_length(-span, span, h) <= L + EPS:
            tours.append((-span, 0.0))
            tours.append((0.0, span))
            break

        # 3. Greedy step – choose the farther endpoint and fire a maximal tour.
        if right >= -left:  # right side is (weakly) farther
            q = right  # positive
            p = _maximal_left_endpoint(q, h, L)
            tours.append((p, q))
            right = p  # shrink uncovered interval
        else:  # left side is farther (more negative)
            q_neg = left  # negative
            q_pos = -q_neg  # mirror to positive axis
            p_pos = _maximal_left_endpoint(q_pos, h, L)
            p_neg = -p_pos
            tours.append((p_neg, q_neg))  # p_neg ≤ q_neg < 0
            left = p_neg

        # Sanity guard: prevent infinite loops due to precision quirks.
        if len(tours) > 10_000:
            raise RuntimeError("GSP: exceeded 10 000 tours – numerical issues?")

    # ------------------------------------------------------------------
    # Final verification (debug builds only)
    # ------------------------------------------------------------------
    if VERBOSE:
        for idx, (p, q) in enumerate(tours, 1):
            Lpq = tour_length(p, q, h)
            log(f"Tour {idx}: interval=({p:.4f},{q:.4f}), length={Lpq:.6f}")
            if Lpq > L + 1e-6:
                log("  **ERROR: length exceeds battery**")
        covered_min = min(t[0] for t in tours)
        covered_max = max(t[1] for t in tours)
        if covered_min > a - 1e-6 or covered_max < b + 1e-6:
            log("  **ERROR: segment not fully covered**")

    return len(tours), tours
