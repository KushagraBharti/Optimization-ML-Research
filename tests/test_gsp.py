import pytest
from coverage_planning.gsp import greedy_min_length_one_segment
from coverage_planning.utils import tour_length

def test_gsp_exact_threshold():
    seg = (0.0, 4.0)
    h = 0.0
    # L exactly equals full tour length → 1 tour
    L = tour_length(0.0, 4.0, h)
    count, tours = greedy_min_length_one_segment(seg, h, L)
    assert count == 1
    assert tours == [(0.0, 4.0)]

def test_gsp_split_needed():
    seg = (0.0, 10.0)
    h = 0.0
    L = 5.0
    count, tours = greedy_min_length_one_segment(seg, h, L)
    # worst‐case needs exactly 3 tours
    assert count == 3
    # ensure coverage
    covered = sorted(tours)
    assert covered[0][0] == 0.0 and covered[-1][1] == 10.0
