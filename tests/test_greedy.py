# tests/test_greedy.py

import pytest
from src.solvers.greedy import cover_min_tours, tour_length

def test_tour_length():
    # For p=0, q=0 on y=1: length = 1 + 1 + 0 = 2
    assert pytest.approx(tour_length(0, 0, h=1.0)) == 2.0

def test_single_segment_fits():
    segments = [(10, 20)]
    L = tour_length(10, 20, h=1.0) + 1e-6
    tours = cover_min_tours(segments, L, h=1.0)
    assert tours == [(10, 20)]

def test_single_segment_too_small():
    segments = [(5, 6)]
    L = 1.0  # too small to even cover that one
    tours = cover_min_tours(segments, L, h=1.0)
    # greedy cannot cover => empty or still tries one but length>L; we expect it to return [(5,6)]
    assert tours == [(5, 6)]

def test_two_segments_mergeable():
    segments = [(0, 1), (2, 3)]
    # battery big enough to cover from 0→3
    L = tour_length(0, 3, h=1.0) + 1e-6
    tours = cover_min_tours(segments, L, h=1.0)
    assert tours == [(0, 3)]

def test_two_segments_separate():
    segments = [(0, 1), (2, 3)]
    # battery only enough for one segment at a time
    L = tour_length(0, 1, h=1.0) + 1e-6
    tours = cover_min_tours(segments, L, h=1.0)
    assert tours == [(0, 1), (2, 3)]
