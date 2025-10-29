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
    from coverage_planning.algs.reference import gs
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.algs.reference import gs

from coverage_planning.common.constants import EPS_GEOM
from coverage_planning.algs.geometry import tour_length
from tests.test_utils import (
    check_cover_exact,
    check_tours_feasible,
    gen_disjoint_segments,
    oracle_min_tours_gs,
    reflect_segments,
    rng,
    scale_instance,
)


# ---------------------------------------------------------------------------
#  Unit tests
# ---------------------------------------------------------------------------
def test_gs_single_tour_finish(tol: float) -> None:
    segments = [(2.0, 3.5), (5.0, 6.5)]
    h = 2.0
    L = 40.0
    count, tours = gs(segments, h=h, L=L)
    assert count == 1
    assert len(tours) == 1
    check_cover_exact(segments, tours)
    check_tours_feasible(h, L, tours, tol=tol)
    p, q = tours[0]
    assert math.isclose(p, segments[0][0], abs_tol=EPS_GEOM)
    assert math.isclose(q, segments[-1][1], abs_tol=EPS_GEOM)


def test_gs_multiple_tours_and_maximality(tol: float) -> None:
    segments = [(-9.0, -7.0), (-5.0, -4.0), (3.0, 4.0), (6.5, 7.2)]
    h = 3.0
    L = 25.0
    count, tours = gs(segments, h=h, L=L)
    assert count >= 2
    check_cover_exact(segments, tours)
    check_tours_feasible(h, L, tours, tol=tol)
    for p, q in tours[:-1]:
        length = tour_length(p, q, h)
        assert math.isclose(length, L, rel_tol=0.0, abs_tol=1e-5)


def test_gs_tie_break_prefers_right(tol: float) -> None:
    segments = [(-5.0, -4.0), (4.0, 5.0)]
    h = 2.5
    L = 20.5
    count, tours = gs(segments, h=h, L=L)
    assert count == 2
    check_cover_exact(segments, tours)
    check_tours_feasible(h, L, tours, tol=tol)
    first_tour = tours[0]
    assert first_tour[1] >= segments[-1][1] - EPS_GEOM


def test_gs_infeasible_due_to_farthest_endpoint() -> None:
    segments = [(-5.0, -3.5), (3.0, 4.0)]
    h = 1.5
    L = 2.0 * math.hypot(5.0, h) - 1.0
    with pytest.raises(ValueError) as exc_info:
        gs(segments, h=h, L=L)
    assert "battery" in str(exc_info.value).lower()


def test_gs_invalid_overlapping_segments() -> None:
    segments = [(0.0, 2.0), (1.5, 3.0)]
    h = 2.0
    L = 30.0
    with pytest.raises(ValueError) as exc_info:
        gs(segments, h=h, L=L)
    assert "disjoint" in str(exc_info.value).lower()


def test_gs_left_sweep_progress(tol: float) -> None:
    segments = [(-10.0, -8.5), (-6.5, -4.5), (2.5, 3.5), (5.0, 6.0)]
    h = 2.5
    L = 23.0
    first_left = min(a for a, _ in segments)
    count, tours = gs(segments, h=h, L=L)
    assert count >= 2
    check_cover_exact(segments, tours)
    check_tours_feasible(h, L, tours, tol=tol)
    covered = {(p, q) for p, q in tours}
    first_tour = tours[0]
    remaining = [
        seg
        for seg in segments
        if not (first_tour[0] - EPS_GEOM <= seg[0] and seg[1] <= first_tour[1] + EPS_GEOM)
    ]
    if remaining:
        new_left = min(a for a, _ in remaining)
        assert new_left > first_left + EPS_GEOM


# ---------------------------------------------------------------------------
#  Hypothesis strategies
# ---------------------------------------------------------------------------
if HAVE_HYPOTHESIS:

    @composite
    def gs_instances(draw):
        k = draw(st.integers(min_value=2, max_value=5))
        x_min = draw(st.floats(min_value=-15.0, max_value=-2.0))
        x_max = draw(st.floats(min_value=2.0, max_value=15.0))
        assume(x_max - x_min > 5.0)
        seed = draw(st.integers(min_value=0, max_value=999999))
        h = draw(st.floats(min_value=1.0, max_value=6.0))
        rnd = rng(seed)
        segments = gen_disjoint_segments(rnd, k, x_min, x_max, min_len=0.6, min_gap=0.6)
        farthest = max(max(abs(a), abs(b)) for a, b in segments)
        factor = draw(st.floats(min_value=1.05, max_value=1.5))
        margin = draw(st.floats(min_value=0.5, max_value=3.0))
        L = factor * 2.0 * math.hypot(farthest, h) + margin
        return segments, h, L


    @settings(max_examples=40)
    @given(gs_instances())
    def test_gs_property_feasible_instances(data, tol: float) -> None:
        segments, h, L = data
        count, tours = gs(segments, h=h, L=L)
        assert count == len(tours)
        check_cover_exact(segments, tours)
        check_tours_feasible(h, L, tours, tol=tol)


    @settings(max_examples=40)
    @given(gs_instances())
    def test_gs_monotonicity_in_L(data, tol: float) -> None:
        segments, h, L = data
        base_count, base_tours = gs(segments, h=h, L=L)
        check_cover_exact(segments, base_tours)
        check_tours_feasible(h, L, base_tours, tol=tol)
        L_scaled = 1.2 * L
        larger_count, larger_tours = gs(segments, h=h, L=L_scaled)
        assert larger_count <= base_count
        check_cover_exact(segments, larger_tours)
        check_tours_feasible(h, L_scaled, larger_tours, tol=tol)


    @settings(max_examples=60)
    @given(
        st.integers(min_value=1, max_value=9999),
        st.integers(min_value=2, max_value=3),
        st.floats(min_value=1.0, max_value=4.0),
    )
    def test_gs_oracle_cross_check(seed: int, k: int, h: float, tol: float) -> None:
        rnd = rng(seed)
        segments = gen_disjoint_segments(
            rnd, k, x_min=-8.0, x_max=8.0, min_len=0.8, min_gap=0.8
        )
        farthest = max(max(abs(a), abs(b)) for a, b in segments)
        L = 1.15 * 2.0 * math.hypot(farthest, h) + 1.0
        oracle = oracle_min_tours_gs(segments, h, L)
        count, tours = gs(segments, h=h, L=L)
        assert count == oracle
        check_cover_exact(segments, tours)
        check_tours_feasible(h, L, tours, tol=tol)

else:  # pragma: no cover

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_gs_property_feasible_instances(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_gs_monotonicity_in_L(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_gs_oracle_cross_check(seed: int, k: int, h: float, tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")


# ---------------------------------------------------------------------------
#  Metamorphic tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [0.25, 2.0, 10.0])
def test_gs_scaling_invariance(scale: float, tol: float) -> None:
    segments = [(-6.0, -4.5), (-2.0, -1.0), (1.5, 3.0)]
    h = 2.2
    L = 19.0
    base_count, base_tours = gs(segments, h=h, L=L)
    scaled_segments = scale_instance(segments, scale)
    scaled_h = h * scale
    scaled_L = L * scale
    scaled_count, scaled_tours = gs(
        scaled_segments, h=scaled_h, L=scaled_L
    )
    assert scaled_count == base_count
    check_cover_exact(scaled_segments, scaled_tours)
    check_tours_feasible(scaled_h, scaled_L, scaled_tours, tol=tol)


def test_gs_reflection_invariance(tol: float) -> None:
    segments = [(-7.5, -6.0), (-3.2, -2.5), (1.0, 2.0), (4.5, 6.0)]
    h = 3.0
    L = 26.0
    count, tours = gs(segments, h=h, L=L)
    reflected_segments = reflect_segments(segments)
    ref_count, ref_tours = gs(reflected_segments, h=h, L=L)
    assert ref_count == count
    lengths = sorted(tour_length(p, q, h) for p, q in tours)
    ref_lengths = sorted(tour_length(p, q, h) for p, q in ref_tours)
    assert all(math.isclose(x, y, abs_tol=tol) for x, y in zip(lengths, ref_lengths))
    check_cover_exact(reflected_segments, ref_tours)
    check_tours_feasible(h, L, ref_tours, tol=tol)
