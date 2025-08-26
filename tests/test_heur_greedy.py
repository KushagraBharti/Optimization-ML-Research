# tests/test_greedy.py
"""
Tests for Greedy MinTours under *paper assumptions*:
segments must be pair-wise disjoint with a positive gap.
"""
import pytest
from coverage_planning.algs.heuristics.gs_mintours import greedy_min_tours

def test_greedy_unsorted():
    segs = [(5, 6), (1, 2), (3, 4)]
    h, L = 0.0, 50.0
    count, tours = greedy_min_tours(segs, h, L)
    assert count == 1
    p, q = tours[0]
    assert p <= 1.0 and q >= 6.0


@pytest.mark.parametrize(
    "segs,L",
    [
        ([(0, 1), (1, 2), (2, 3)], 1.001),  # touching endpoints → invalid
        ([(0, 2), (1, 3)], 2.5),            # overlapping → invalid
    ],
)
def test_greedy_invalid_instances_raise(segs, L):
    with pytest.raises(ValueError):
        greedy_min_tours(segs, h=0.0, L=L)
