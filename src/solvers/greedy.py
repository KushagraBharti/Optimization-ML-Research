# src/solvers/greedy.py

import math
from typing import List, Tuple

def tour_length(p: float, q: float, h: float = 1.0) -> float:
    """
    Compute the length of the triangular tour O→(p,h)→(q,h)→O.
    O is at (0,0), line is y=h.
    """
    return math.hypot(p, h) + math.hypot(q, h) + (q - p)


def cover_min_tours(
    segments: List[Tuple[float, float]],
    L: float,
    h: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Greedy strategy to cover all disjoint, sorted segments on line y=h
    with as few tours as possible under battery limit L.
    
    segments: sorted list of (a_i, b_i), disjoint, a_i < b_i
    returns list of (p, q) endpoints for each tour
    """
    tours: List[Tuple[float, float]] = []
    n = len(segments)
    i = 0
    while i < n:
        a_i, _ = segments[i]
        # try to extend as far right as possible
        j = i
        while j + 1 < n and tour_length(a_i, segments[j + 1][1], h) <= L:
            j += 1
        # record tour covering segments[i..j]
        p = a_i
        q = segments[j][1]
        tours.append((p, q))
        # advance
        i = j + 1
    return tours
