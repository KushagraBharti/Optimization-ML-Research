# tests/test_dp.py

import pytest
import itertools
from src.solvers.dp import cover_min_length, tour_length

def brute_force_cover(segments, L, h=1.0):
    """
    Try all ways to partition segments into tours,
    return minimal total length solution.
    """
    n = len(segments)
    best = None
    best_len = float('inf')
    # generate all ways to split [0..n) into contiguous groups
    for mask in itertools.product([0, 1], repeat=n - 1):
        groups = []
        start = 0
        for i, cut in enumerate(mask, start=1):
            if cut:
                groups.append((start, i))
                start = i
        groups.append((start, n))
        # compute tours
        valid = True
        total = 0
        tlist = []
        for a, b in groups:
            p = segments[a][0]
            q = segments[b - 1][1]
            length = tour_length(p, q, h)
            if length > L:
                valid = False
                break
            total += length
            tlist.append((p, q))
        if valid and total < best_len:
            best_len = total
            best = tlist
    return best or []

@pytest.mark.parametrize("segments", [
    [(0, 1)],
    [(0, 1), (2, 3)],
    [(0, 2), (3, 5), (6, 7)],
])
def test_dp_matches_bruteforce(segments):
    # choose L large enough to cover in one tour
    h = 1.0
    L_full = tour_length(segments[0][0], segments[-1][1], h)
    L = L_full + 0.1
    dp_sol = cover_min_length(segments, L, h)
    brute = brute_force_cover(segments, L, h)
    assert pytest.approx(sum(tour_length(p, q, h) for p, q in dp_sol), rel=1e-6) \
           == sum(tour_length(p, q, h) for p, q in brute)
    # also ensure dp_sol covers all segments
    covered = [(p, q) for p, q in dp_sol]
    assert len(dp_sol) == len(brute)

def test_dp_no_solution():
    segments = [(0, 1), (10, 11)]
    L = 2.0  # too small to cover even one
    sol = cover_min_length(segments, L, h=1.0)
    assert sol == []
