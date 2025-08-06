# tests/test_dp_1side.py
"""
Regression tests for `coverage_planning.dp_1side`
using the three-tuple return signature
    prefix_costs, suffix_costs, candidates = dp_one_side(...)
"""

from __future__ import annotations
import math
import random
import pytest

from coverage_planning.dp_1side import dp_one_side
from coverage_planning.utils import tour_length


# ---------------------------------------------------------------------------
#  Helper
# ---------------------------------------------------------------------------
def _single_long_tour_cost(segs, h):
    a_min = min(a for a, _ in segs)
    b_max = max(b for _, b in segs)
    return tour_length(a_min, b_max, h)


# ---------------------------------------------------------------------------
#  Basic single / double segment sanity
# ---------------------------------------------------------------------------
def test_single_segment():
    segs = [(2.0, 4.0)]
    h, L = 0.0, 10.0
    prefix, suffix, C = dp_one_side(segs, h, L)

    assert C == [4.0]
    exp = tour_length(2.0, 4.0, h)
    assert math.isclose(prefix[-1], exp, abs_tol=1e-6)
    assert suffix[-1] == 0.0            # last suffix is always 0


def test_two_segments_large_L():
    segs = [(0.0, 2.0), (3.0, 5.0)]
    h, L = 0.0, 1e6
    prefix, suffix, _ = dp_one_side(segs, h, L)

    exp_total = _single_long_tour_cost(segs, h)
    assert math.isclose(prefix[-1], exp_total, rel_tol=1e-9)
    # suffix values are conservative upper-bounds; they must never be negative
    assert all(s >= 0.0 for s in suffix)


# ---------------------------------------------------------------------------
#  Infeasible tiny-battery instances must raise
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "segments,L",
    [
        ([(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)], 0.9),
        ([(10.0, 12.5), (15.0, 18.0)], 1.5),
    ],
)
def test_value_error_on_unreachable_segment(segments, L):
    with pytest.raises(ValueError):
        dp_one_side(segments, h=0.0, L=L)


# ---------------------------------------------------------------------------
#  Randomised regression: DP never exceeds one-long-tour cost
# ---------------------------------------------------------------------------
def test_random_instances_upper_bound():
    random.seed(0xC0FFEE)
    for _ in range(30):
        n = random.randint(2, 8)
        xs = sorted(random.uniform(0, 20) for _ in range(2 * n))
        segs = [(xs[2 * i], xs[2 * i + 1]) for i in range(n)]
        h = 1.0
        L = _single_long_tour_cost(segs, h) + 5.0      # generous battery
        prefix, *_ = dp_one_side(segs, h, L)
        assert prefix[-1] <= _single_long_tour_cost(segs, h) + 1e-7
