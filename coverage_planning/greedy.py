# coverage_planning/greedy.py

import math
from typing import List, Tuple
from .utils import (
    tour_length,
    find_maximal_p,
    subtract_covered_intervals,
    log,
    VERBOSE,
    EPS,
    sort_segments
)

def _print_tour_details(p: float, q: float, h: float) -> float:
    """
    Print the detailed 3-leg breakdown of O->P->Q->O and return total length.
    """
    # Leg 1: O -> P
    log("  Leg 1: O -> P")
    d1 = math.hypot(p, h)
    log(f"    Distance (0.0, 0.0) -> ({p:.4f}, {h:.4f}) = {d1:.4f}")
    # Leg 2: P -> Q (horizontal)
    log("  Leg 2: P -> Q (horizontal)")
    d2 = abs(q - p)
    log(f"    Horizontal distance x={p:.4f} -> x={q:.4f} = {d2:.4f}")
    # Leg 3: Q -> O
    log("  Leg 3: Q -> O")
    d3 = math.hypot(q, h)
    log(f"    Distance ({q:.4f}, {h:.4f}) -> (0.0, 0.0) = {d3:.4f}")
    total = d1 + d2 + d3
    log(f"  Total length = {d1:.4f} + {d2:.4f} + {d3:.4f} = {total:.4f}\n")
    return total

def greedy_min_tours(
    segments: List[Tuple[float, float]],
    h: float,
    L: float
) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Greedy Strategy (GS): minimize the number of tours covering all segments.
    Handles two-sided by processing right then left.
    Returns (tour_count, list_of_(p,q) tours).
    """
    # --- Header ---
    log("=== Greedy Strategy (GS): Minimize Number of Tours ===")
    log(f"Line y={h}, Battery L={L:.4f}")
    log(f"Segments: {segments}\n")

    tours: List[Tuple[float, float]] = []
    tour_count = 0

    # Partition segments into right (b>=0) and left (b<0) reflected
    sides = [
        ([(a,b) for (a,b) in segments if b >= 0], +1),
        ([( -b, -a) for (a,b) in segments if b <  0], -1),
    ]

    for side_segs, sign in sides:
        if not side_segs:
            continue  # skip empty side

        rem = sort_segments(side_segs)
        # Step A: handle unreachable segments
        reachable = []
        for a, b in rem:
            if tour_length(a, b, h) > L + EPS:
                # this segment alone needs its own tour
                tour_count += 1
                p, q = sign*a, sign*b
                log(f"--- Tour {tour_count} (unreachable single segment) ---")
                log(f"Tour covers interval ({p:.4f}, {q:.4f})\n")
                tours.append((p, q))
            else:
                reachable.append((a, b))
        rem = reachable

        # Step B: greedy cover remaining
        while rem:
            tour_count += 1
            log(f"--- Tour {tour_count} ---")
            # farthest endpoint:
            p0 = rem[0][0]
            q0 = rem[-1][1]
            log(f"Farthest endpoint x={q0:.4f}")

            # Attempt minimal-length covering all:
            log("Attempt minimal-length tour:")
            total_len = _print_tour_details(p0, q0, h)

            if total_len <= L + EPS:
                # done
                log("Using minimal-length tour to finish.")
                p, q = p0, q0
            else:
                # maximal-length via farthest
                log("Minimal-length too long, doing maximal-length tour:")
                q = q0
                p = find_maximal_p(q, h, L)
                log(f"Computed P=({p:.4f},{h:.4f})")
                _print_tour_details(p, q, h)

            covered = (min(p, q), max(p, q))
            log(f"Tour covers interval {covered}")
            rem = subtract_covered_intervals(rem, covered)
            log(f"Remaining: {rem}\n")
            tours.append((sign*p, sign*q))

    # --- Footer ---
    log(f"Total tours: {tour_count}\n")
    return tour_count, tours
