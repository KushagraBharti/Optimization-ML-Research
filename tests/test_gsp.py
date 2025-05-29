import pytest
from coverage_planning.gsp import greedy_min_length_one_segment
from coverage_planning.utils import tour_length

def test_one_tour_possible():
    seg = (0.0, 2.0)
    h = 0.0
    L = tour_length(0, 2, h) + 1e-6
    count, tours = greedy_min_length_one_segment(seg, h, L)
    assert count == 1
    assert tours == [(0.0, 2.0)]

def test_two_tours_needed():
    seg = (0.0, 4.0)
    h = 0.0
    L = 6.0
    count, tours = greedy_min_length_one_segment(seg, h, L)
    assert count == 2
    # first tour must end at segment end
    assert tours[0][1] == 4.0

def test_three_tours_needed():
    seg = (0.0, 10.0)
    h = 0.0
    L = 5.0
    count, tours = greedy_min_length_one_segment(seg, h, L)
    assert count == 3
