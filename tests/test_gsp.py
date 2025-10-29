from __future__ import annotations

import math

import pytest

try:
    from hypothesis import assume, given, settings, strategies as st
    from hypothesis.strategies import composite

    HAVE_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAVE_HYPOTHESIS = False
    assume = None  # type: ignore[assignment]
    given = settings = None  # type: ignore[assignment]
    st = None  # type: ignore[assignment]

try:
    from coverage_planning.algs.reference import gsp
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.algs.reference import gsp

from coverage_planning.common.constants import EPS_GEOM
from coverage_planning.algs.geometry import tour_length
from tests.test_utils import (
    check_cover_exact,
    check_tours_feasible,
    oracle_min_length_one_segment,
    reflect_segments,
    rng,
    tour_len_sum,
)


# ---------------------------------------------------------------------------
#  Unit tests
# ---------------------------------------------------------------------------
def test_gsp_one_tour_finish(tol: float) -> None:
    seg = (2.0, 6.5)
    h = 2.5
    direct = tour_length(seg[0], seg[1], h)
    L = direct + 2.0
    count, tours = gsp(seg, h=h, L=L)
    assert count == 1
    assert len(tours) == 1
    check_cover_exact([seg], tours)
    check_tours_feasible(h, L, tours, tol=tol)
    p, q = tours[0]
    assert math.isclose(p, seg[0], abs_tol=EPS_GEOM)
    assert math.isclose(q, seg[1], abs_tol=EPS_GEOM)


def test_gsp_two_tours_at_zero(tol: float) -> None:
    seg = (-7.0, 5.0)
    h = 2.0
    L = 21.5
    count, tours = gsp(seg, h=h, L=L)
    assert count == 2
    check_cover_exact([seg], tours)
    check_tours_feasible(h, L, tours, tol=tol)
    tour_set = {(round(p, 6), round(q, 6)) for p, q in tours}
    assert (-7.0, 0.0) in tour_set or (0.0, -7.0) in tour_set
    assert (0.0, 5.0) in tour_set or (5.0, 0.0) in tour_set


def test_gsp_multiple_sweeps_one_side(tol: float) -> None:
    seg = (0.5560129491841641, 6.7559704318079685)
    h = 2.900186347909552
    L = 14.761716560455461
    count, tours = gsp(seg, h=h, L=L)
    assert count == 3
    check_cover_exact([seg], tours)
    check_tours_feasible(h, L, tours, tol=tol)
    for p, q in tours[:2]:
        length = tour_length(min(p, q), max(p, q), h)
        assert math.isclose(length, L, abs_tol=1e-6)


def test_gsp_knife_edge_behavior(tol: float) -> None:
    seg = (-4.0, 3.5)
    h = 1.2
    farthest = max(abs(seg[0]), abs(seg[1]))
    L = 2.0 * math.hypot(farthest, h) + 1e-7
    try:
        count, tours = gsp(seg, h=h, L=L)
        check_cover_exact([seg], tours)
        check_tours_feasible(h, L, tours, tol=tol)
    except ValueError as exc:
        message = str(exc).lower()
        assert "battery" in message or "central" in message


def test_gsp_invalid_segment() -> None:
    seg = (1.5, 1.5)
    h = 2.0
    L = 10.0
    with pytest.raises(ValueError):
        gsp(seg, h=h, L=L)


def test_gsp_extremely_asymmetric_segment(tol: float) -> None:
    seg = (-25.0, -0.2)
    h = 2.5
    L = tour_length(seg[0], seg[1], h) + 5.0
    count, tours = gsp(seg, h=h, L=L)
    assert count == 1
    check_cover_exact([seg], tours)
    check_tours_feasible(h, L, tours, tol=tol)


# ---------------------------------------------------------------------------
#  Hypothesis strategies
# ---------------------------------------------------------------------------
if HAVE_HYPOTHESIS:

    @composite
    def gsp_instances(draw):
        h = draw(st.floats(min_value=1.0, max_value=5.0))
        mode = draw(st.sampled_from(["one", "two"]))
        rnd = rng(draw(st.integers(min_value=0, max_value=999999)))
        if mode == "one":
            seg = (
                rnd.uniform(-8.0, 6.0),
                rnd.uniform(6.5, 14.0),
            )
            seg = tuple(sorted(seg))
            direct = tour_length(seg[0], seg[1], h)
            factor = draw(st.floats(min_value=1.02, max_value=1.4))
            L = factor * direct + draw(st.floats(min_value=0.2, max_value=4.0))
        else:
            left = -draw(st.floats(min_value=3.0, max_value=9.0))
            right = draw(st.floats(min_value=2.5, max_value=8.0))
            seg = (left, right)
            left_len = tour_length(left, 0.0, h)
            right_len = tour_length(0.0, right, h)
            direct = tour_length(left, right, h)
            factor = draw(st.floats(min_value=1.02, max_value=1.2))
            L = factor * max(left_len, right_len)
            assume(direct > L + 0.5)
        return seg, h, L, mode


    @settings(max_examples=50)
    @given(gsp_instances())
    def test_gsp_property_expected_tour_structure(data, tol: float) -> None:
        seg, h, L, mode = data
        count, tours = gsp(seg, h=h, L=L)
        check_cover_exact([seg], tours)
        check_tours_feasible(h, L, tours, tol=tol)
        total_length = tour_len_sum(h, tours)
        oracle = oracle_min_length_one_segment(seg, h, L)
        assert math.isclose(total_length, oracle, abs_tol=1e-5)
        if mode == "one":
            assert count == 1
        else:
            assert count == 2
            hits_zero = any(
                min(p, q) <= 0.0 + EPS_GEOM <= max(p, q) for p, q in tours
            )
            assert hits_zero

else:  # pragma: no cover

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_gsp_property_expected_tour_structure(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")


# ---------------------------------------------------------------------------
#  Metamorphic tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [0.25, 2.0, 5.0])
def test_gsp_scaling(scale: float, tol: float) -> None:
    seg = (-6.0, 4.0)
    h = 3.0
    L = 28.0
    base_count, base_tours = gsp(seg, h=h, L=L)
    base_cost = tour_len_sum(h, base_tours)
    scaled_seg = [(p * scale, q * scale) for p, q in [seg]][0]
    scaled_h = h * scale
    scaled_L = L * scale
    scaled_count, scaled_tours = gsp(
        scaled_seg, h=scaled_h, L=scaled_L
    )
    scaled_cost = tour_len_sum(scaled_h, scaled_tours)
    assert scaled_count == base_count
    assert math.isclose(scaled_cost, base_cost * scale, rel_tol=0.0, abs_tol=1e-5)
    check_cover_exact([scaled_seg], scaled_tours)
    check_tours_feasible(scaled_h, scaled_L, scaled_tours, tol=tol)


def test_gsp_reflection(tol: float) -> None:
    seg = (-8.5, 6.0)
    h = 2.5
    L = 26.0
    count, tours = gsp(seg, h=h, L=L)
    cost = tour_len_sum(h, tours)
    reflected_seg = reflect_segments([seg])[0]
    ref_count, ref_tours = gsp(reflected_seg, h=h, L=L)
    ref_cost = tour_len_sum(h, ref_tours)
    assert ref_count == count
    assert math.isclose(cost, ref_cost, abs_tol=1e-5)
    check_cover_exact([reflected_seg], ref_tours)
    check_tours_feasible(h, L, ref_tours, tol=tol)
