# coverage_planning/dp_both.py

import bisect
from typing import List, Tuple
from .dp_1side import dp_one_side, generate_candidates_one_side
from .utils import tour_length, log, VERBOSE, EPS

def dp_full_line(
    segments: List[Tuple[float, float]],
    h: float,
    L: float
) -> float:
    """
    Full-line MinLength DP (Algorithm 4).
    Returns the optimal total travel length covering all segments.
    """
    # Header
    sep = "=" * 26
    log(sep)
    log("ALGORITHM 4: General DP (Both Sides)")
    log(sep)
    log(f"All segments: {segments}")
    log(f"Line y = {h}, Battery limit L = {L}\n")

    # Partition
    left  = [(-b, -a) for (a, b) in segments if b <  0]
    right = [( a,  b) for (a, b) in segments if a >  0]
    mid   = [( a,  b) for (a, b) in segments if a <= 0 <= b]

    # Only shortcut to the one-sided DP if every segment crosses the origin.
    # The previous implementation returned early whenever *any* crossing
    # segment was present, which incorrectly ignored additional left/right
    # segments.  This led to an underestimated cost.
    if mid and not left and not right:
        log("Segments crossing projection (cover via one-sided DP):", mid)
        dp_mid, _ = dp_one_side(mid, h, L, side_label="MID")
        result = dp_mid[-1]
        log(f"Final DP Value = {result:.4f} (all segments cross origin)\n")
        return result

    # Pure one-sided shortcuts (only when the other side *and* mid are empty)
    if not left and not mid:
        log("No left-side segments; reduce to one-sided on RIGHT")
        result = dp_one_side(right, h, L, side_label="RIGHT")[0][-1]
        log(f"Final DP Value = {result:.4f}\n")
        return result
    if not right and not mid:
        log("No right-side segments; reduce to one-sided on LEFT")
        result = dp_one_side(left, h, L, side_label="LEFT")[0][-1]
        log(f"Final DP Value = {result:.4f}\n")
        return result

    # If only one side exists alongside mid segments, merge them and run the
    # one-sided DP on the combined set.
    if not left:
        combined = mid + right
        log("No left-side segments; solve mid+right via one-sided DP")
        result = dp_one_side(combined, h, L, side_label="RIGHT+MID")[0][-1]
        log(f"Final DP Value = {result:.4f}\n")
        return result
    if not right:
        combined = left + mid
        log("No right-side segments; solve mid+left via one-sided DP")
        result = dp_one_side(combined, h, L, side_label="LEFT+MID")[0][-1]
        log(f"Final DP Value = {result:.4f}\n")
        return result

    # Solve each side independently
    log(f"Left-side segments (reflected to +x): {left}")
    dp_l, _ = dp_one_side(left, h, L, side_label="LEFT")
    log(f"  Left-side cost = {dp_l[-1]:.4f}\n")

    log(f"Right-side segments: {right}")
    dp_r, _ = dp_one_side(right, h, L, side_label="RIGHT")
    log(f"  Right-side cost = {dp_r[-1]:.4f}\n")

    # No-bridge cost
    cost_no = dp_l[-1] + dp_r[-1]
    log(f"No-bridge cost = {dp_l[-1]:.4f} + {dp_r[-1]:.4f} = {cost_no:.4f}\n")

    # Candidate sets for bridge trials
    C_l, _ = generate_candidates_one_side(left,  h, L)
    C_r, _ = generate_candidates_one_side(right, h, L)
    log(f"Candidates Left: {C_l}")
    log(f"Candidates Right: {C_r}\n")

    # Try bridges
    best = cost_no
    for i, p in enumerate(C_l):
        # find largest j in C_r with tour_length(p, C_r[j]) <= L
        lo, hi = 0, len(C_r) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if tour_length(p, C_r[mid], h) <= L + EPS:
                lo = mid + 1
            else:
                hi = mid - 1
        if hi >= 0:
            q = C_r[hi]
            cand = dp_l[i] + tour_length(p, q, h) + dp_r[hi]
            log(f"Trying bridge: p={p:.4f}, q={q:.4f} → tour={tour_length(p, q, h):.4f}, total={cand:.4f}")
            if cand < best:
                best = cand

    # Final result
    log(f"\nFinal Optimal Cost = {best:.4f} (with or without bridge)\n")
    return best
