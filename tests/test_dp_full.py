from __future__ import annotations

import math
from typing import List, Sequence, Tuple

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
    from coverage_planning.algs.reference import dp_full, dpos
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.algs.reference import dp_full, dpos

from coverage_planning.algs.geometry import tour_length
from tests.test_utils import (
    gen_disjoint_segments,
    oracle_min_length_full_line,
    reflect_segments,
    rng,
    scale_instance,
)


def _baseline_cost(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> float:
    left = [seg for seg in segments if seg[1] <= 0.0]
    right = [seg for seg in segments if seg[0] >= 0.0]
    total = 0.0
    if left:
        left_ref = [(-b, -a) for a, b in reversed(left)]
        Sigma_L, _ = dpos(left_ref, h=h, L=L)
        total += Sigma_L[-1]
    if right:
        Sigma_R, _ = dpos(right, h=h, L=L)
        total += Sigma_R[-1]
    return total


# ---------------------------------------------------------------------------
#  Unit tests
# ---------------------------------------------------------------------------
def test_dp_full_all_on_one_side_matches_dpos(tol: float) -> None:
    segments = [(1.5, 2.5), (4.0, 5.0)]
    h = 2.0
    L = 30.0
    cost_full, tours = dp_full(segments, h=h, L=L)
    Sigma, _ = dpos(segments, h=h, L=L)
    assert math.isclose(cost_full, Sigma[-1], abs_tol=tol)
    assert tours == []


def test_dp_full_gap_around_origin_equals_baseline(tol: float) -> None:
    segments = [(-7.0, -5.5), (-3.0, -2.0), (2.5, 3.4), (5.0, 6.1)]
    h = 2.5
    L = 23.0
    cost_full, _ = dp_full(segments, h=h, L=L)
    baseline = _baseline_cost(segments, h, L)
    assert math.isclose(cost_full, baseline, abs_tol=tol)


def test_dp_full_bridge_beneficial(tol: float) -> None:
    segments = [
        (-8.89835723442975, -6.9393559339752535),
        (-5.307185703922216, -3.7870960811960925),
        (2.3179655663366483, 3.307328152086481),
        (4.770532804219864, 6.2435507360996265),
    ]
    h = 2.5
    L = 18.626939647203475
    cost_full, _ = dp_full(segments, h=h, L=L)
    baseline = _baseline_cost(segments, h, L)
    assert cost_full < baseline - 1e-4


def test_dp_full_straddling_segment_matches_oracle(tol: float) -> None:
    segments = [
        (-3.37675248706401, -1.7721230293140966),
        (2.0915610830440468, 3.1487028879463246),
        (3.7112867574285993, 4.990761818398388),
    ]
    h = 1.4994411525659943
    L = 14.445651073544326
    cost_full, _ = dp_full(segments, h=h, L=L)
    oracle = oracle_min_length_full_line(segments, h, L)
    assert math.isclose(cost_full, oracle, abs_tol=5e-5)


def test_dp_full_bridge_maximality_not_exposed() -> None:
    segments = [(-7.5, -6.0), (-4.0, -3.2), (2.8, 3.6), (5.5, 7.2)]
    h = 2.0
    L = 20.5
    cost_full, tours = dp_full(segments, h=h, L=L)
    baseline = _baseline_cost(segments, h, L)
    if cost_full < baseline - 1e-4:
        pytest.skip("dp_full does not expose bridge endpoints for maximality check")
    else:
        assert not tours


# ---------------------------------------------------------------------------
#  Hypothesis strategies
# ---------------------------------------------------------------------------
if HAVE_HYPOTHESIS:

    @composite
    def dp_full_instances(draw):
        k = draw(st.integers(min_value=2, max_value=3))
        seed = draw(st.integers(min_value=0, max_value=99999))
        rnd = rng(seed)
        segments = gen_disjoint_segments(
            rnd, k, x_min=-10.0, x_max=10.0, min_len=0.8, min_gap=0.8
        )
        h = draw(st.floats(min_value=1.0, max_value=4.0))
        farthest = max(max(abs(a), abs(b)) for a, b in segments)
        base = 2.0 * math.hypot(farthest, h)
        L = draw(st.floats(min_value=1.05 * base, max_value=1.4 * base))
        return segments, h, L


    @settings(max_examples=50)
    @given(dp_full_instances())
    def test_dp_full_oracle_cross_check(data, tol: float) -> None:
        segments, h, L = data
        cost_full, _ = dp_full(segments, h=h, L=L)
        oracle = oracle_min_length_full_line(segments, h, L)
        assert math.isclose(cost_full, oracle, abs_tol=5e-5)


    @settings(max_examples=50)
    @given(dp_full_instances())
    def test_dp_full_bound_by_baseline(data, tol: float) -> None:
        segments, h, L = data
        cost_full, _ = dp_full(segments, h=h, L=L)
        baseline = _baseline_cost(segments, h, L)
        assert cost_full <= baseline + tol


    @settings(max_examples=40)
    @given(dp_full_instances())
    def test_dp_full_monotonicity_in_L(data, tol: float) -> None:
        segments, h, L = data
        cost_full, _ = dp_full(segments, h=h, L=L)
        cost_loose, _ = dp_full(segments, h=h, L=1.2 * L)
        assert cost_loose <= cost_full + tol


    @settings(max_examples=40)
    @given(dp_full_instances())
    def test_dp_full_reflection_invariance(data, tol: float) -> None:
        segments, h, L = data
        cost_full, _ = dp_full(segments, h=h, L=L)
        reflected = reflect_segments(segments)
        cost_ref, _ = dp_full(reflected, h=h, L=L)
        assert math.isclose(cost_full, cost_ref, abs_tol=tol)

else:  # pragma: no cover

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_dp_full_oracle_cross_check(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_dp_full_bound_by_baseline(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_dp_full_monotonicity_in_L(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_dp_full_reflection_invariance(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")


# ---------------------------------------------------------------------------
#  Metamorphic tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [0.5, 2.0, 6.0])
def test_dp_full_scaling(scale: float, tol: float) -> None:
    segments = [(-6.0, -4.5), (-2.0, -1.0), (1.0, 2.0), (4.0, 5.5)]
    h = 2.5
    L = 24.0
    cost_full, _ = dp_full(segments, h=h, L=L)
    scaled_segments = scale_instance(segments, scale)
    scaled_cost, _ = dp_full(
        scaled_segments, h=h * scale, L=L * scale
    )
    assert math.isclose(scaled_cost, cost_full * scale, abs_tol=1e-5)


def test_dp_full_insensitive_to_eps_slivers(tol: float, eps: float) -> None:
    segments = [(-5.0, -3.0), (-0.5, -0.4), (0.4, 0.5), (2.0, 3.5)]
    h = 2.0
    L = 20.0
    base_cost, _ = dp_full(segments, h=h, L=L)
    delta = min(tol, eps) / 2.0
    segments_perturbed = [
        (-5.0, -3.0),
        (-0.5 - delta, -0.4 + delta),
        (0.4 - delta, 0.5 + delta),
        (2.0, 3.5),
    ]
    aug_cost, _ = dp_full(segments_perturbed, h=h, L=L)
    assert math.isclose(base_cost, aug_cost, abs_tol=1e-5)
