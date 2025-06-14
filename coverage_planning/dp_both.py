import math
from typing import List, Tuple, Dict

from .utils import tour_length, log, EPS
from .dp_1side import dp_one_side

__all__ = [
    "dp_full_line",
]

# ---------------------------------------------------------------------------
#  Min‑Length on both sides of the projection point O′ (Algorithm 4).
# ---------------------------------------------------------------------------
#  Notation matches the paper:
#     Σ_left(p)   – optimal cost to cover everything *left* of, and including,
#                   candidate p on the (reflected) left side.
#     Σ̃_right(q) – optimal cost to cover everything *right* of, and including,
#                   q on the right side (suffix table).
# ---------------------------------------------------------------------------


# -- helpers ---------------------------------------------------------------

def _split_and_reflect(segments: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Split any segment that straddles x=0 and reflect the left half.

    Returns (left_reflected, right) where both lists lie entirely in x≥0.
    """
    left_ref: List[Tuple[float, float]] = []
    right:    List[Tuple[float, float]] = []

    for a, b in segments:
        if b <= 0:                      # fully left → reflect
            left_ref.append((-b, -a))
        elif a >= 0:                    # fully right
            right.append((a, b))
        else:                           # straddles 0 → split
            if a < 0:
                left_ref.append((0.0 - b, 0.0))       # (−b,0)
            if b > 0:
                right.append((0.0, b))

    # remove zero‑length leftovers & sort
    left_ref  = [(l, r) for (l, r) in left_ref if r - l > EPS]
    right     = [(l, r) for (l, r) in right    if r - l > EPS]
    left_ref.sort(key=lambda s: s[0])
    right.sort(key=lambda s: s[0])
    return left_ref, right


def _suffix_dp_right(right: List[Tuple[float, float]], h: float, L: float) -> Dict[float, float]:
    """Return Σ̃_right: suffix‑cost table for the right side."""
    if not right:
        return {}

    B = right[-1][1]
    mirror = [(B - b, B - a) for (a, b) in right]
    mirror.sort(key=lambda s: s[0])

    C_m, dp_m = dp_one_side(mirror, h, L)  # prefix cost in mirrored world
    suffix: Dict[float, float] = {}
    for c_m, cost in zip(C_m, dp_m):
        q = B - c_m
        suffix[q] = cost
    return suffix


# -- main -----------------------------------------------------------------

def dp_full_line(segments: List[Tuple[float, float]], h: float, L: float) -> float:
    """Exact minimum total flight length for an arbitrary two‑sided instance.

    The helper `dp_one_side` in this codebase has the historical signature:
        (dp_vals: List[float], backptr: Dict[float,str])
    where the keys of *backptr* are exactly the candidate points **C** in
    ascending order and the *dp_vals* list is aligned with that order.

    We therefore reconstruct **C** by sorting the keys of *backptr* whenever
    needed.  If a newer dp_one_side variant returns (C, dp_vals) we detect that
    shape and adapt automatically.  This keeps Algorithm 4 robust to either
    signature without touching the one‑sided module.
    """

    # Phase 0 – preprocessing
    left_ref, right = _split_and_reflect(segments)
    log("Left (reflected):", left_ref)
    log("Right:", right, "\n")

    # One‑sided shortcuts
    ret_r = dp_one_side(right, h, L)
    if not left_ref:
        dp_r_vals = ret_r[0] if isinstance(ret_r[0], list) else ret_r[1]
        return dp_r_vals[-1]

    ret_l = dp_one_side(left_ref, h, L)
    if not right:
        dp_l_vals = ret_l[0] if isinstance(ret_l[0], list) else ret_l[1]
        return dp_l_vals[-1]

    # Phase 1 – prefix tables on each side (handle either return shape)
    if isinstance(ret_l[0], list):
        dp_l_vals, back_l = ret_l
        C_l = sorted(back_l.keys())
    else:  # new shape (C_l, dp_l_vals, ...)
        C_l, dp_l_vals = ret_l[:2]

    if isinstance(ret_r[0], list):
        dp_r_vals, back_r = ret_r
        C_r = sorted(back_r.keys())
    else:
        C_r, dp_r_vals = ret_r[:2]

    Σ_left = {c: v for c, v in zip(C_l, dp_l_vals)}
    cost_no_bridge = dp_l_vals[-1] + dp_r_vals[-1]

    # Phase 2 – suffix table on right
    Σ̃_right = _suffix_dp_right(right, h, L)

    # Phase 3 – evaluate maximal bridge candidates
    best = cost_no_bridge

    for i_p, p in enumerate(C_l):
        # binary search for farthest q with feasible tour_length(p,q) ≤ L
        lo, hi = 0, len(C_r) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if tour_length(p, C_r[mid], h) <= L + EPS:
                lo = mid + 1
            else:
                hi = mid - 1
        if hi < 0:
            continue
        q = C_r[hi]

        # maximality checks
        if hi + 1 < len(C_r) and tour_length(p, C_r[hi + 1], h) <= L + EPS:
            continue  # not right‑maximal
        if i_p > 0 and tour_length(C_l[i_p - 1], q, h) <= L + EPS:
            continue  # not left‑maximal

        cost = dp_l_vals[i_p] + tour_length(p, q, h) + Σ̃_right.get(q, math.inf)
        if cost < best:
            best = cost

    return best
