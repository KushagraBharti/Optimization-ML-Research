import pytest
from coverage_planning.dp_both import dp_full_line
from coverage_planning.dp_1side import dp_one_side
from coverage_planning.utils import tour_length

def test_dp_full_line_only_left():
    segments = [(-5.0, -3.0), (-2.0, -1.0)]
    h, L = 0.0, 10.0
    cost = dp_full_line(segments, h, L)
    # Should equal one-sided DP on the reflected left segments
    reflected = [(-b, -a) for (a, b) in segments]
    dp_ref, _ = dp_one_side(reflected, h, L)
    assert abs(cost - dp_ref[-1]) < 1e-6

def test_dp_full_line_bridge():
    segments = [(-3.0, -1.0), (1.0, 3.0)]
    h = 0.0
    # battery long enough to cover from -3 to 3 in one tour
    L = tour_length(-3.0, 3.0, h) + 1e-6
    cost = dp_full_line(segments, h, L)
    expected = tour_length(-3.0, 3.0, h)
    assert abs(cost - expected) < 1e-6

@pytest.mark.parametrize("segs, reflect", [
    # right-only reduces to one-sided on the same coords
    ([(2.0, 4.0), (6.0, 7.0)], False),
    # left-only reduces to one-sided on the reflected coords
    ([(-5.0, -3.0), (-2.0, -1.0)], True),
])
def test_dp_full_line_one_side_shortcuts(segs, reflect):
    h, L = 0.0, 1e6
    cost = dp_full_line(segs, h, L)
    # Compare against the correct one-sided oracle
    if reflect:
        ref = [(-b, -a) for (a, b) in segs]
        dp_ref, _ = dp_one_side(ref, h, L)
    else:
        dp_ref, _ = dp_one_side(segs, h, L)
    assert abs(cost - dp_ref[-1]) < 1e-6


def test_dp_full_line_with_mid_and_others():
    """Ensure mid segments don't cause other segments to be ignored."""
    segments = [(-1.0, 1.0), (2.0, 4.0)]
    h = 0.0
    L = tour_length(-1.0, 4.0, h) + 1e-6
    cost = dp_full_line(segments, h, L)
    # Optimal is one tour covering the full range
    expected = tour_length(-1.0, 4.0, h)
    assert abs(cost - expected) < 1e-6
