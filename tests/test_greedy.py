import math
import pytest
from coverage_planning.greedy import greedy_min_tours
from coverage_planning.utils import tour_length

def test_all_in_one_tour():
    segments = [(0,1), (2,3), (4,5)]
    h = 0.0
    # battery just enough to cover from 0 to 5
    L = tour_length(0, 5, h) + 1e-6
    count, tours = greedy_min_tours(segments, h, L)
    assert count == 1
    p, q = tours[0]
    assert p <= 0 and q >= 5

def test_two_tours_needed():
    segments = [(0,4), (8,12)]
    h = 0.0
    L = 10.0
    count, tours = greedy_min_tours(segments, h, L)
    assert count == 2

def test_both_sides_separate_tours():
    segments = [(-3, -1), (1, 3)]
    h = 0.0
    L = 6.0  # can't bridge from -3 to 3 in one tour
    count, tours = greedy_min_tours(segments, h, L)
    assert count == 2
