# coverage_planning/dp_1side.py
"""
One-sided exact MinLength DP (DPOS) (Algorithm 3)
"""

from __future__ import annotations
import math
from typing import List, Tuple

from ..geometry import tour_length, find_maximal_p, sort_segments, log, EPS, VERBOSE

__all__ = ["generate_candidates_one_side", "dp_one_side"]

# Candidate set         
def generate_candidates_one_side(
    segments: List[Tuple[float, float]], h: float, L: float
) -> List[float]:
    """Return the candidate set **C** in ascending order."""
    segs = sort_segments(segments)
    rights = [b for _, b in segs]

    C: list[float] = []
    seen: set[float] = set()

    for b in rights:
        q = b
        while True:
            r = math.hypot(q, h)
            K = L - r - q
            if abs(K) < EPS:
                break
            p = find_maximal_p(q, h, L)
            # stop if we jumped into the gap before the first segment
            if p < segs[0][0] - EPS:
                break
            # stop chain if p lies in a gap
            idx = next(i for i, (_a, bb) in enumerate(segs) if bb >= p - EPS)
            if segs[idx][0] - EPS > p:
                break
            if p not in seen:
                seen.add(p)
                C.append(p)
            q = p

    for b in rights:
        if b not in seen:
            seen.add(b)
            C.append(b)

    C.sort()
    return C



# DP core
def _len(a: float, b: float, h: float) -> float:
    return tour_length(a, b, h)


def dp_one_side(
    segments: List[Tuple[float, float]], h: float, L: float
) -> Tuple[List[float], List[float], List[float]]:
    segs = sort_segments(segments)
    if segs[0][0] < -EPS:
        raise ValueError("dp_one_side expects x ≥ 0")

    for a, b in segs:
        if _len(a, b, h) > L + EPS:
            raise ValueError("Unreachable segment — battery too small")

    C = generate_candidates_one_side(segs, h, L)
    m = len(C)
    idx = {c: i for i, c in enumerate(C)}

    prefix = [math.inf] * m
    a1 = segs[0][0]
    seg_ptr = 0

    # prefix DP 
    for k, c in enumerate(C):
        best = math.inf

        # Case 1
        cost1 = _len(a1, c, h)
        if cost1 <= L + EPS:
            best = cost1

        # Case 2
        while seg_ptr < len(segs) and segs[seg_ptr][1] < c - EPS:
            seg_ptr += 1
        j = seg_ptr
        while j >= 0 and segs[j][0] <= c + EPS:
            a_j = segs[j][0]
            prev = 0.0 if j == 0 else prefix[idx[segs[j - 1][1]]]
            t_cost = _len(a_j, c, h)
            if t_cost <= L + EPS and prev + t_cost < best:
                best = prev + t_cost
            j -= 1
            if j >= 0 and segs[j][1] < segs[j + 1][0] - EPS:
                break  # gap

        # Case 3
        t = k - 1
        while t >= 0 and abs(_len(C[t], c, h) - L) > EPS:
            t -= 1
        if t >= 0 and prefix[t] + L < best:
            best = prefix[t] + L

        prefix[k] = best
        if VERBOSE:
            log(f"Σ*({c:.3f}) = {best:.3f}")

    # suffix DP (constant-time recurrence)
    suffix = [0.0] * m
    suffix[-1] = 0.0
    for k in range(m - 2, -1, -1):
        suffix[k] = suffix[k + 1] + _len(C[k], C[k + 1], h)

    return prefix, suffix, C
