import math
import pytest
from coverage_planning.dp_both import dp_full_line
from coverage_planning.dp_1side import dp_one_side
from coverage_planning.utils import tour_length

def test_dp_full_line_only_left():
    segments = [(-5.0, -3.0), (-2.0, -1.0)]
    h, L = 0.0, 10.0
    cost = dp_full_line(segments, h, L)
    # should equal one-sided DP on the reflected left segments
    left_reflected = [(-b, -a) for (a, b) in segments]
    dp, _ = dp_one_side(left_reflected, h, L)
    assert abs(cost - dp[-1]) < 1e-6

def test_dp_full_line_bridge():
    segments = [(-3.0, -1.0), (1.0, 3.0)]
    h = 0.0
    # battery long enough to cover from -3 to 3 in one tour
    L = tour_length(-3.0, 3.0, h) + 1e-6
    cost = dp_full_line(segments, h, L)
    expected = tour_length(-3.0, 3.0, h)
    assert abs(cost - expected) < 1e-6
