import math
from typing import List, Tuple
from collections import deque

# Local helpers come from utils.py (kept unchanged)
from .utils import tour_length, find_maximal_p, EPS, log, sort_segments

__all__ = [
    "greedy_min_tours",
]


def _remove_covered_from_right(seg_deque: deque, left: float, right: float) -> None:
    """Erase the portion [left,right] from **right‑hand side** of the deque.

    We assume seg_deque is sorted by left endpoint and `right` coincides with the
    *rightmost* uncovered point.  The routine mutates seg_deque in‑place and runs
    amortised O(#segments removed)."""
    while seg_deque:
        a, b = seg_deque[-1]
        if b < left - EPS:          # segment lies completely left of the tour
            break
        if a >= left - EPS:         # segment fully covered – drop it
            seg_deque.pop()
        else:                       # partial overlap – trim its right end
            seg_deque[-1] = (a, left)
            break


def _remove_covered_from_left(seg_deque: deque, left: float, right: float) -> None:
    """Erase the portion [left,right] from **left‑hand side** of the deque."""
    while seg_deque:
        a, b = seg_deque[0]
        if a > right + EPS:         # segment lies completely right of the tour
            break
        if b <= right + EPS:        # segment fully covered – drop it
            seg_deque.popleft()
        else:                       # partial overlap – trim its left end
            seg_deque[0] = (right, b)
            break


def greedy_min_tours(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> Tuple[int, List[Tuple[float, float]]]:
    """Greedy Strategy (GS) from the paper – *optimal* for **MinTours**.

    Parameters
    ----------
    segments : list of (a,b)   – pairwise‑disjoint closed intervals on the x‑axis
    h        : float           – take‑off height of the drones (y‑coordinate)
    L        : float           – battery budget (must be >= minimal feasible tour)

    Returns
    -------
    (m, tours) where
        m     – minimum number of tours required (exact)
        tours – list of (p,q) giving every tour O→(p,h)→(q,h)→O in the order sent

    Complexity
    -----------
    Runs in O(m + n) time and O(n) memory, matching the paper’s bound.
    """
    if not segments:
        return 0, []

    # Pre‑sort and copy into a deque for O(1) pops from both ends.
    segs = deque(sort_segments(segments))
    tours: List[Tuple[float, float]] = []

    while segs:
        # Fast feasibility test: can *everything left* be covered in a single shot?
        min_x = segs[0][0]
        max_x = segs[-1][1]
        if tour_length(min_x, max_x, h) <= L + EPS:
            # One final minimal‑length tour finishes the job.
            tours.append((min_x, max_x))
            break

        # Otherwise pick the *farthest* still‑uncovered point f.
        if abs(max_x) >= abs(min_x):
            # Farthest is on the right‐hand side.
            f = max_x
            p = find_maximal_p(f, h, L)           # left partner of maximal tour
            left, right = min(p, f), max(p, f)
            tours.append((left, right))
            _remove_covered_from_right(segs, left, right)
        else:
            # Farthest is on the left.
            f = min_x            # negative value, farthest to the left
            # Reflect to the right half‑plane, compute partner, reflect back.
            f_pos = -f
            p_pos = find_maximal_p(f_pos, h, L)
            left, right = -f_pos, -p_pos
            if left > right:
                left, right = right, left
            tours.append((left, right))
            _remove_covered_from_left(segs, left, right)

    return len(tours), tours
