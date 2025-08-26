# coverage_planning/greedy.py
"""
Greedy Strategy (GS) — optimal for MinTours on disjoint segments. (algorithm 1)
"""
from __future__ import annotations
import math
from collections import deque
from typing import List, Tuple

from ..geometry import tour_length, find_maximal_p, sort_segments, EPS

__all__ = ["greedy_min_tours"]


def _validate_disjoint(segments: List[Tuple[float, float]]):
    """Raise if segments overlap or merely touch (paper pre-condition)."""
    segs = sort_segments(segments)
    for i in range(1, len(segs)):
        if segs[i - 1][1] >= segs[i][0] - EPS:
            raise ValueError("Segments must be pair-wise disjoint (positive gap)")


def _trim_right(segs: deque, left: float, right: float):
    while segs:
        a, b = segs[-1]
        if b <= left + EPS:
            break
        if a >= left - EPS:
            segs.pop()
        else:
            segs[-1] = (a, left)
            break


def _trim_left(segs: deque, left: float, right: float):
    while segs:
        a, b = segs[0]
        if a >= right - EPS:
            break
        if b <= right + EPS:
            segs.popleft()
        else:
            segs[0] = (right, b)
            break


def greedy_min_tours(segments: List[Tuple[float, float]], h: float, L: float) -> Tuple[int, List[Tuple[float, float]]]:
    
    if not segments:
        return 0, []

    _validate_disjoint(segments)  # faithful to paper

    segs = deque(sort_segments(segments))
    tours: List[Tuple[float, float]] = []

    while segs:
        min_x, max_x = segs[0][0], segs[-1][1]

        # Final single tour feasible?
        if tour_length(min_x, max_x, h) <= L + EPS:
            tours.append((min_x, max_x))
            break

        # Farthest uncovered point
        if abs(max_x) >= abs(min_x):  # right side
            f = max_x
            p = find_maximal_p(f, h, L)
            # safeguard: progress must be > EPS
            if f - p < EPS or tour_length(p, f, h) > L + EPS:
                p = f
            tours.append((p, f))
            _trim_right(segs, p, f)
        else:  # left side
            f_neg = min_x
            f_pos = -f_neg
            p_pos = find_maximal_p(f_pos, h, L)
            if f_pos - p_pos < EPS or tour_length(p_pos, f_pos, h) > L + EPS:
                p_pos = f_pos
            p_neg = -p_pos
            tours.append((p_neg, f_neg))
            _trim_left(segs, p_neg, f_neg)

    return len(tours), tours
