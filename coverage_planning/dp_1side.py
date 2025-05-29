# coverage_planning/dp_1side.py

import math, bisect
from typing import List, Tuple, Dict
from .utils import tour_length, find_maximal_p, sort_segments, EPS, log

def generate_candidates_one_side(
    segments: List[Tuple[float,float]],
    h: float,
    L: float
) -> Tuple[List[float], Dict[float,float]]:
    """
    Build candidate set C and predecessor map via GS‐jumps.
    Guards against K≈0 to avoid division by zero.
    """
    segs = sort_segments(segments)
    endpoints = [b for _,b in segs]
    C = set(endpoints)
    pred: Dict[float,float] = {}
    sorted_b = sorted(endpoints)

    for b in endpoints:
        cur = b
        while True:
            r = math.hypot(cur, h)
            # if 2*r ≈ L, K≈0 → no nontrivial jump
            if abs(L - 2*r) < EPS:
                break
            p = find_maximal_p(cur, h, L)
            if p <= cur + EPS:
                break
            if p not in C:
                C.add(p)
                pred[p] = cur
            # step to next endpoint ≤ p
            idx = bisect.bisect_right(sorted_b, p) - 1
            if idx < 0:
                break
            cur = sorted_b[idx]

    C_sorted = sorted(C)
    return C_sorted, pred

def dp_one_side(
    segments: List[Tuple[float,float]],
    h: float,
    L: float
) -> Tuple[List[float], Dict[float,str]]:
    """
    One‐sided MinLength DP (DPOS). Returns dp-costs and backpointer descriptions.
    """
    segs = sort_segments(segments)
    # unreachable check
    for a,b in segs:
        if tour_length(a,b,h) > L + EPS:
            raise ValueError(f"Segment {(a,b)} unreachable")

    C, pred = generate_candidates_one_side(segs, h, L)
    nC = len(C)
    dp = [math.inf]*nC
    desc: Dict[float,str] = {}
    idx_of = {c:i for i,c in enumerate(C)}
    a1 = segs[0][0]

    log("[DP1] Candidates:", C)
    for i, c in enumerate(C):
        best = math.inf
        best_desc = ""
        # Case 1
        cost1 = tour_length(a1, c, h)
        log(f"[DP1] Case1 a1→{c:.4f} cost={cost1:.4f}")
        if cost1 <= L + EPS:
            best, best_desc = cost1, f"Case1: [{a1:.4f}->{c:.4f}]"
        # Case 2
        for j, (aj, bj) in enumerate(segs):
            if aj - EPS <= c <= bj + EPS:
                prev_cost = dp[idx_of[segs[j-1][1]]] if j>0 else 0.0
                c2 = prev_cost + tour_length(aj, c, h)
                log(f"[DP1] Case2 seg{j} cost2={c2:.4f}")
                if c2 < best:
                    best, best_desc = c2, f"Case2: seg{j}"
        # Case 3
        if c in pred:
            pc = pred[c]
            c3 = dp[idx_of[pc]] + L
            log(f"[DP1] Case3 pred={pc:.4f} cost3={c3:.4f}")
            if c3 < best:
                best, best_desc = c3, f"Case3: pred={pc:.4f}"
        dp[i], desc[c] = best, best_desc
        log(f"[DP1] DP[{c:.4f}] = {best:.4f} via {best_desc}")

    return dp, desc
