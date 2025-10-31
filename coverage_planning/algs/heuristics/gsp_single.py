# coverage_planning/gsp.py
"""
Greedy with Projection (GSP) - exact MinLength for a single segment. (algorithm 2)
"""
from __future__ import annotations

import math
from typing import List, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM

EPS = EPS_GEOM

__all__ = ["greedy_min_length_one_segment"]


def _max_left(q: float, h: float, L: float) -> float:
    return find_maximal_p(q, h, L)


def greedy_min_length_one_segment(seg: Tuple[float, float], h: float, L: float) -> Tuple[int, List[Tuple[float, float]]]:
    
    a, b = sorted(seg)
    left, right = a, b
    tours: List[Tuple[float, float]] = []

    while True:
        # 1) can one tour finish?
        if tour_length(left, right, h) <= L + EPS:
            tours.append((left, right))
            break

        # 2) can two symmetric tours finish?
        span = max(-left, right)
        if tour_length(-span, span, h) <= L + EPS:
            tours.extend([(-span, 0.0), (0.0, span)])
            break

        # 3) greedy maximal sweep
        prev_left, prev_right = left, right
        if right >= -left:          # sweep from right
            p = _max_left(right, h, L)
            if right - p < EPS:
                # stagnation -> residual now straddles O' and fits in two tours
                span = max(-left, right)
                tours.extend([(-span, 0.0), (0.0, span)])
                break
            tours.append((p, right))
            right = p
        else:                       # sweep from left
            p_pos = _max_left(-left, h, L)
            if -left - p_pos < EPS:
                span = max(-left, right)
                tours.extend([(-span, 0.0), (0.0, span)])
                break
            tours.append((-p_pos, left))
            left = -p_pos

        # paranoia: must shrink interval
        if abs(prev_left - left) < EPS and abs(prev_right - right) < EPS:
            raise RuntimeError("GSP failed to make progress (numeric bug?)")

    return len(tours), tours
