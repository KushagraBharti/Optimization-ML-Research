# coverage_planning/gsp.py

from typing import Tuple, List
from .utils import tour_length, find_maximal_p, EPS, log

def greedy_min_length_one_segment(
    seg: Tuple[float,float],
    h: float,
    L: float
) -> Tuple[int, List[Tuple[float,float]]]:
    """
    GSP for single-segment MinLength. Always returns 1–3 tours.
    """
    a, b = seg
    tours: List[Tuple[float,float]] = []

    # Step 1: whole segment
    full_len = tour_length(a, b, h)
    log(f"[GSP] full length {full_len:.4f}")
    if full_len <= L + EPS:
        tours.append((a, b))
        return 1, tours

    # Step 2: maximal tour through b
    p1 = find_maximal_p(b, h, L)
    tours.append((p1, b))
    log(f"[GSP] maximal to b: p={p1:.4f}")

    # Step 3: leftover [a,p1]
    rem = (a, p1)
    rem_len = tour_length(rem[0], rem[1], h)
    log(f"[GSP] leftover length {rem_len:.4f}")
    if rem_len <= L + EPS:
        tours.append((rem[0], rem[1]))
        return len(tours), tours

    # fallback: split at projection
    tours.append((rem[0], 0.0))
    tours.append((0.0, rem[1]))
    log("[GSP] split fallback")
    return len(tours), tours
