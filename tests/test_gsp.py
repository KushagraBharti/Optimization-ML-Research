# tests/test_gsp.py
"""
Unit tests for Greedy-with-Projection (GSP) on a single segment.
"""

import math
from coverage_planning.gsp import greedy_min_length_one_segment
from coverage_planning.utils import tour_length


def test_gsp_exact_threshold():
    seg = (0.0, 4.0)
    h = 0.0
    L = tour_length(*seg, h)
    count, tours = greedy_min_length_one_segment(seg, h, L)
    assert count == 1
    assert tours == [seg]


def test_gsp_split_needed():
    """
    Battery 5, h = 0  ⇒ farthest reachable x = 2.5, so covering [0,10]
    provably needs 4 maximal tours.
    """
    seg = (0.0, 10.0)
    h = 0.0
    L = 5.0

    count, tours = greedy_min_length_one_segment(seg, h, L)
    assert count == 4      # optimal
    # coverage sanity
    left = min(p for p, _ in tours)
    right = max(q for _, q in tours)
    assert math.isclose(left, 0.0, abs_tol=1e-6)
    assert math.isclose(right, 10.0, abs_tol=1e-6)
