# coverage_planning/utils.py
"""
Light-weight geometry helpers shared by all algorithms.
"""

from __future__ import annotations
import math
from typing import List, Tuple

# Global debug switch
VERBOSE: bool = False
EPS: float = 1e-9


def log(*args, **kwargs) -> None:            # pragma: no cover
    if VERBOSE:
        print(*args, **kwargs)

# --------------------------------------------------------------------------- #
#  Geometry                                                                   #
# --------------------------------------------------------------------------- #
def tour_length(p: float, q: float, h: float) -> float:
    """Return length of O→(p,h)→(q,h)→O, no ordering assumed for p,q."""
    return math.hypot(p, h) + abs(q - p) + math.hypot(q, h)


def find_maximal_p(q: float, h: float, L: float) -> float:
    r = math.hypot(q, h)
    K = L - r - q
    if abs(K) > EPS:                       # analytic branch
        p = (h * h - K * K) / (2 * K)
        if p <= q + EPS:
            return p
    # numeric fallback
    lo, hi = 0.0, q
    for _ in range(60):
        mid = (lo + hi) / 2
        if tour_length(mid, q, h) > L:
            lo = mid
        else:
            hi = mid
    return hi


def sort_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return segments sorted by their left endpoint."""
    return sorted(segments, key=lambda s: s[0])
