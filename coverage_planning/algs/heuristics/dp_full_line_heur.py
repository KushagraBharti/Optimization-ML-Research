# coverage_planning/dp_both.py
"""
Full-line exact MinLength DP (Algorithm 4) using the new one-sided API.
"""

from __future__ import annotations
import math
from typing import List, Tuple

from ..geometry import tour_length, log, EPS
from .dp_one_side_heur import dp_one_side

__all__ = ["dp_full_line"]


# --------------------------------------------------------------------------- #
def _split_reflect(segments: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    
    left, right = [], []
    for a, b in segments:
        if b <= 0:
            left.append((-b, -a))
        elif a >= 0:
            right.append((a, b))
        else:  # straddles 0
            left.append((0.0, -a))
            right.append((0.0, b))
    left = [(l, r) for l, r in left if r - l > EPS]
    right = [(l, r) for l, r in right if r - l > EPS]
    left.sort(key=lambda s: s[0])
    right.sort(key=lambda s: s[0])
    return left, right


# --------------------------------------------------------------------------- #
def dp_full_line(segments: List[Tuple[float, float]], h: float, L: float) -> float:
    
    left_ref, right = _split_reflect(segments)
    log("Left (reflected):", left_ref)
    log("Right:", right, "\n")

    # One-sided shortcuts
    if not left_ref:
        pref, *_ = dp_one_side(right, h, L)
        return pref[-1]
    if not right:
        pref, *_ = dp_one_side(left_ref, h, L)
        return pref[-1]

    # Run DP on each side
    pref_L, _suf_L, C_l = dp_one_side(left_ref, h, L)
    pref_R, suf_R, C_r = dp_one_side(right, h, L)

    Σ_left = {c: cost for c, cost in zip(C_l, pref_L)}
    Σ̃_right = {c: cost for c, cost in zip(C_r, suf_R)}

    best = pref_L[-1] + pref_R[-1]  # no bridge

    # Enumerate maximal bridge endpoints
    for i, p in enumerate(C_l):
        # farthest q feasible with p
        j_hi = max(
            j
            for j, q in enumerate(C_r)
            if tour_length(p, q, h) <= L + EPS
        )
        q = C_r[j_hi]

        # Ensure maximality (cannot extend either side)
        if j_hi + 1 < len(C_r) and tour_length(p, C_r[j_hi + 1], h) <= L + EPS:
            continue
        if i > 0 and tour_length(C_l[i - 1], q, h) <= L + EPS:
            continue

        cost = Σ_left[p] + tour_length(p, q, h) + Σ̃_right[q]
        if cost < best:
            best = cost

    return best
