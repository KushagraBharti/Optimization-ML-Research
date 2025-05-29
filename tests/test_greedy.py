import pytest
from coverage_planning.greedy import greedy_min_tours
from coverage_planning.utils import tour_length

def test_greedy_unsorted_segments():
    # unsorted input order
    segs = [(5,6),(1,2),(3,4)]
    h = 0.0
    L = tour_length(1,6,h) + 1e-6
    count, tours = greedy_min_tours(segs, h, L)
    assert count == 1
    # tour must cover range [1,6]
    p,q = tours[0]
    assert p <= 1 and q >= 6

@pytest.mark.parametrize("segs,L,expected", [
    ([(0,1),(1,2),(2,3)], 1.001, 3),    # tiny battery → each segment separately
    ([(0,2),(1,3)], 2.5, 2),            # partial overlap
])
def test_greedy_edge_cases(segs, L, expected):
    h = 0.0
    count, _ = greedy_min_tours(segs, h, L)
    assert count == expected
