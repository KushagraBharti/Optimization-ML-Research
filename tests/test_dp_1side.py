import pytest
from coverage_planning.dp_1side import dp_one_side, generate_candidates_one_side
from coverage_planning.utils import tour_length

def test_generate_candidates_single_segment():
    segments = [(2.0, 4.0)]
    h, L = 0.0, 10.0
    C, pred = generate_candidates_one_side(segments, h, L)
    assert C == [4.0]
    assert pred == {}

def test_dp_one_side_single_segment():
    segments = [(2.0, 4.0)]
    h, L = 0.0, 10.0
    dp, desc = dp_one_side(segments, h, L)
    expected = tour_length(2.0, 4.0, h)
    assert abs(dp[-1] - expected) < 1e-6

def test_dp_one_side_two_segments():
    segments = [(0.0, 2.0), (3.0, 5.0)]
    h, L = 0.0, 10.0
    dp, desc = dp_one_side(segments, h, L)
    expected = tour_length(0.0, 5.0, h)
    assert abs(dp[-1] - expected) < 1e-6
