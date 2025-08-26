# tests/test_dp_both.py
"""
Unit tests for `coverage_planning.dp_both` given the new dp_one_side API.
"""

from __future__ import annotations
import math
import random
import pytest

from coverage_planning.algs.heuristics.dp_one_side_heur import dp_one_side
from coverage_planning.algs.heuristics.dp_full_line_heur import dp_full_line
from coverage_planning.algs.geometry import tour_length, EPS

def _reflect(segs):
    """Reflect left-hand segments to the right half-plane."""
    return [(-b, -a) for (a, b) in segs]


# ---------------------------------------------------------------------------
#  Pure one-side shortcut cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "segments,reflect",
    [
        ([(2.0, 4.0), (6.0, 7.0)], False),
        ([(-5.0, -3.0), (-2.0, -1.0)], True),
    ],
)
def test_one_side_shortcut(segments, reflect):
    h, L = 0.0, 1e6
    full_cost = dp_full_line(segments, h, L)

    oracle_segs = _reflect(segments) if reflect else segments
    prefix, *_ = dp_one_side(oracle_segs, h, L)
    assert math.isclose(full_cost, prefix[-1], rel_tol=1e-9)


# ---------------------------------------------------------------------------
#  Bridge tour spans the whole line
# ---------------------------------------------------------------------------
def test_whole_line_bridge():
    segments = [(-3.0, -1.0), (1.0, 3.0)]
    h = 0.0
    L = tour_length(-3.0, 3.0, h) + 1e-6
    cost = dp_full_line(segments, h, L)
    assert math.isclose(cost, L, abs_tol=1e-6)


# ---------------------------------------------------------------------------
#  Random symmetric instances: cost ≤ one-long-tour bound
# ---------------------------------------------------------------------------
def test_random_symmetric_bound():
    random.seed(2025)
    for _ in range(15):
        segs = []
        for x in range(1, 6):
            w = random.uniform(0.3, 1.2)
            segs.append((-x - w, -x))
            segs.append((x, x + w))
        h = 1.0
        L = max(tour_length(a, b, h) for a, b in segs) * 3
        full_cost = dp_full_line(segs, h, L)

        a_min = min(a for a, _ in segs)
        b_max = max(b for _, b in segs)
        assert full_cost <= tour_length(a_min, b_max, h) + 1e-6
