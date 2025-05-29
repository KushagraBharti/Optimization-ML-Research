# coverage_planning/utils.py

import math
from typing import List, Tuple

# Toggle for debug printing
VERBOSE = False
EPS = 1e-9

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def tour_length(p: float, q: float, h: float) -> float:
    """
    Compute the length of the tour O->(p,h)->(q,h)->O.
    """
    return math.hypot(p, h) + abs(q - p) + math.hypot(q, h)

def find_maximal_p(q: float, h: float, L: float) -> float:
    """
    Closed-form solution for p given q, h, L (see paper).
    """
    r = math.hypot(q, h)
    K = L - r - q
    return (h*h - K*K) / (2*K)

def subtract_covered_intervals(
    segments: List[Tuple[float, float]],
    covered: Tuple[float, float]
) -> List[Tuple[float, float]]:
    """
    Subtract [p,q] from disjoint segments.
    """
    p, q = covered
    updated: List[Tuple[float, float]] = []
    for a, b in segments:
        if b <= p or a >= q:
            updated.append((a, b))
        else:
            if a < p < b:
                updated.append((a, p))
            if a < q < b:
                updated.append((q, b))
    return updated

def sort_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Sort by left endpoint.
    """
    return sorted(segments, key=lambda seg: seg[0])
