# coverage_planning/dp_1side.py

import math
from typing import List, Tuple, Dict
from .utils import tour_length, find_maximal_p, sort_segments, EPS, log, VERBOSE

def generate_candidates_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float
) -> Tuple[List[float], Dict[float, float]]:
    """
    Build candidate set C and predecessor map via GS‐jumps.
    Guards against K≈0 to avoid division by zero.
    """
    segs = sort_segments(segments)
    endpoints = [b for _, b in segs]
    C = set(endpoints)
    pred: Dict[float, float] = {}
    sorted_b = sorted(endpoints)

    for b in endpoints:
        cur = b
        while True:
            r = math.hypot(cur, h)
            if abs(L - 2*r) < EPS:        # K≈0
                break
            p = find_maximal_p(cur, h, L)
            if p <= cur + EPS:
                break
            if p not in C:
                C.add(p)
                pred[p] = cur
            idx = sorted_b.index(cur)
            # step to next endpoint ≤ p
            next_idx = idx - 1
            while next_idx >= 0 and sorted_b[next_idx] > p + EPS:
                next_idx -= 1
            if next_idx < 0:
                break
            cur = sorted_b[next_idx]

    C_sorted = sorted(C)
    return C_sorted, pred


def _print_dp_tour_details(a: float, b: float, h: float) -> float:
    """
    Print the 3-leg O->P->Q->O breakdown with indent for DP logging.
    """
    # Leg 1
    log("      Leg 1: O -> P")
    d1 = math.hypot(a, h)
    log(f"        Distance (0.0, 0.0) -> ({a:.4f}, {h:.4f}) = {d1:.4f}")
    # Leg 2
    log("      Leg 2: P -> Q (horizontal)")
    d2 = abs(b - a)
    log(f"        Horizontal distance x={a:.4f} -> x={b:.4f} = {d2:.4f}")
    # Leg 3
    log("      Leg 3: Q -> O")
    d3 = math.hypot(b, h)
    log(f"        Distance ({b:.4f}, {h:.4f}) -> (0.0, 0.0) = {d3:.4f}")
    total = d1 + d2 + d3
    log(f"        Total length = {d1:.4f} + {d2:.4f} + {d3:.4f} = {total:.4f}\n")
    return total


def dp_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    side_label: str = ""
) -> Tuple[List[float], Dict[float, str]]:
    """
    One‐sided MinLength DP (DPOS).
    Returns (dp-costs array, backpointer descriptions).
    """
    segs = sort_segments(segments)

    # Header
    sep = "=" * 26
    label = side_label.upper() if side_label else "ONE SIDE"
    log(sep)
    log(f"ALGORITHM 3: DPOS ({label})")
    log(sep)
    log(f"Segments on line y = {h}, Battery limit L = {L}")
    log(f"Segments: {segs}\n")

    # Unreachable check
    for a, b in segs:
        if tour_length(a, b, h) > L + EPS:
            raise ValueError(f"Segment {(a,b)} unreachable in one tour")

    # Build candidates and predecessor jumps
    C, pred = generate_candidates_one_side(segs, h, L)
    nC = len(C)
    dp = [math.inf] * nC
    desc: Dict[float, str] = {}
    idx_of = {c: i for i, c in enumerate(C)}
    a1 = segs[0][0]

    log(f"Candidates C: {C}\n")

    # seg_i will track which segment might cover the current c
    seg_i = 0

    for i, c in enumerate(C):
        log(f"Evaluating DP[{c:.2f}] — covering up to x = {c:.2f}")
        best = math.inf
        best_desc = ""

        # Case 1: one full tour a1 -> c
        log(f"    Trying Case 1: One full tour from start of first segment ({a1:.2f}) to current end ({c:.2f})")
        cost1 = _print_dp_tour_details(a1, c, h)
        if cost1 <= L + EPS:
            best, best_desc = cost1, f"Case 1: [{a1:.2f}->{c:.2f}]"

        # Advance seg_i so segs[seg_i].b >= c
        while seg_i < len(segs) and segs[seg_i][1] < c - EPS:
            seg_i += 1

        # Case 2: subinterval tour if c lies in segs[seg_i]
        if seg_i < len(segs):
            a_i, b_i = segs[seg_i]
            if a_i - EPS <= c <= b_i + EPS:
                log(f"    Trying Case 2: Subinterval tour for segment ({a_i:.2f}, {b_i:.2f}) ending at {c:.2f}")
                prev_cost = 0.0
                if seg_i > 0:
                    prev_end = segs[seg_i - 1][1]
                    prev_cost = dp[idx_of[prev_end]]
                cost2 = prev_cost + _print_dp_tour_details(a_i, c, h)
                if cost2 < best:
                    best, best_desc = cost2, f"Case 2: seg{seg_i}"

        # Case 3: jump predecessor + full-length L
        if c in pred:
            pc = pred[c]
            log(f"    Trying Case 3: Jump pred={pc:.2f} + full-length tour")
            cost3 = dp[idx_of[pc]] + L
            log(f"    Case 3 cost = {cost3:.4f}")
            if cost3 < best:
                best, best_desc = cost3, f"Case 3: pred={pc:.2f}"

        dp[i] = best
        desc[c] = best_desc
        log(f"DP[{c:.2f}] = {best:.4f} via {best_desc}\n")

    # Footer
    log(f"Final DP Value = {dp[-1]:.4f} (minimum total length to cover all segments on {label})\n")
    return dp, desc
