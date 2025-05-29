# coverage_planning/dp_both.py

from typing import List, Tuple
from .dp_1side import dp_one_side, generate_candidates_one_side
from .utils import tour_length, log

def dp_full_line(
    segments: List[Tuple[float, float]],
    h: float,
    L: float
) -> float:
    """
    Full‐line MinLength DP (Algorithm 4).
    """
    left  = [(-b,-a) for (a,b) in segments if b <  0]
    right = [( a, b) for (a,b) in segments if a >  0]
    mid   = [( a, b) for (a,b) in segments if a <= 0 <= b]

    # crossing‐origin segments → one-sided on mid
    if mid:
        return dp_one_side(mid, h, L)[0][-1]
    # pure one‐side cases
    if not left:
        return dp_one_side(right, h, L)[0][-1]
    if not right:
        return dp_one_side(left,  h, L)[0][-1]

    dp_l, _ = dp_one_side(left,  h, L)
    dp_r, _ = dp_one_side(right, h, L)
    cost_no = dp_l[-1] + dp_r[-1]
    log(f"[DPboth] no-bridge {dp_l[-1]:.4f}+{dp_r[-1]:.4f}={cost_no:.4f}")

    C_l, _ = generate_candidates_one_side(left,  h, L)
    C_r, _ = generate_candidates_one_side(right, h, L)

    best = cost_no
    for i, p in enumerate(C_l):
        lo, hi = 0, len(C_r)-1
        while lo <= hi:
            mid = (lo + hi)//2
            if tour_length(p, C_r[mid], h) <= L:
                lo = mid + 1
            else:
                hi = mid - 1
        if hi >= 0:
            q = C_r[hi]
            cand = dp_l[i] + tour_length(p, q, h) + dp_r[hi]
            log(f"[DPboth] bridge p={p:.4f},q={q:.4f},cost={cand:.4f}")
            if cand < best:
                best = cand

    log(f"[DPboth] final={best:.4f}")
    return best
