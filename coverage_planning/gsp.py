# coverage_planning/gsp.py

import math
from typing import Tuple, List
from .utils import tour_length, find_maximal_p, log, VERBOSE, EPS

def _print_tour_details(p: float, q: float, h: float) -> float:
    """
    Print the 3-leg breakdown for O->P->Q->O, matching the GS detail style.
    """
    # Leg 1: O -> P
    log("  Leg 1: O -> P")
    d1 = math.hypot(p, h)
    log(f"    Distance (0.0, 0.0) -> ({p:.4f}, {h:.4f}) = {d1:.4f}")
    # Leg 2: P -> Q
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

def greedy_min_length_one_segment(
    seg: Tuple[float,float],
    h: float,
    L: float
) -> Tuple[int, List[Tuple[float,float]]]:
    """
    Greedy-Projection (GSP) for single-segment MinLength.
    Returns (num_tours, list of (p,q) tours).
    """
    a, b = seg
    tours: List[Tuple[float,float]] = []

    # Header
    log("=== GSP: MinLength (1 segment) ===")
    log(f"Line y={h}, Projection O'=(0,{h})")
    log(f"Segment: ({a}, {b}), L={L:.4f}\n")

    # Step 1: try whole segment
    log("Step 1: minimal-length for [a,b]:")
    full_len = _print_tour_details(a, b, h)
    if full_len <= L + EPS:
        tours.append((a, b))
        log(f"Total Tour(s): {len(tours)}\n")
        return len(tours), tours

    # Step 2: maximal-length from b
    log("Step 2: maximal-length from Q=b:")
    p1 = find_maximal_p(b, h, L)
    log(f"P1=({p1:.4f},{h:.4f})")
    _print_tour_details(p1, b, h)
    tours.append((p1, b))

    # Remaining interval
    rem = (a, p1)
    log(f"Remaining interval: {rem}\n")

    # Step 3: minimal-length for leftover
    log("Step 3: minimal-length for leftover:")
    rem_len = _print_tour_details(rem[0], rem[1], h)
    if rem_len <= L + EPS:
        tours.append((rem[0], rem[1]))
        log(f"Total Tour(s): {len(tours)}\n")
        return len(tours), tours

    # Fallback: split at projection O'
    log("Leftover too large, splitting at O':")
    log("Tour 2: left to projection")
    _print_tour_details(rem[0], 0.0, h)
    tours.append((rem[0], 0.0))
    log("Tour 3: projection to right")
    _print_tour_details(0.0, rem[1], h)
    tours.append((0.0, rem[1]))
    log(f"Total Tour(s): {len(tours)}\n")
    return len(tours), tours
