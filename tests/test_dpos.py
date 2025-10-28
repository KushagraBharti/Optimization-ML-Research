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
    from coverage_planning.algs.reference import dp_one_side_ref
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.algs.reference import dp_one_side_ref

from coverage_planning.algs.geometry import tour_length, find_maximal_p
from tests.test_utils import (
    gen_one_side_segments,
    oracle_min_length_one_side,
    rng,
    scale_instance,
)


def _find_idx(candidates: Sequence[float], value: float, tol: float = 1e-8) -> int:
    for idx, cand in enumerate(candidates):
        if abs(cand - value) <= tol:
            return idx
    raise AssertionError(f"{value:.9f} not present in candidates {candidates}")


# ---------------------------------------------------------------------------
#  Unit tests
# ---------------------------------------------------------------------------
def test_dpos_single_segment(tol: float) -> None:
    segments = [(0.0, 4.0)]
    h = 2.5
    L = 40.0
    Sigma, C = dp_one_side_ref(segments, h=h, L=L)
    assert len(Sigma) == len(C) == 1
    idx = _find_idx(C, segments[0][1])
    expected = tour_length(segments[0][0], segments[0][1], h)
    assert math.isclose(Sigma[idx], expected, abs_tol=tol)


def test_dpos_two_segments_sum_lengths(tol: float) -> None:
    segments = [(0.0, 2.0), (3.5, 5.0)]
    h = 2.0
    L = 16.0
    Sigma, C = dp_one_side_ref(segments, h=h, L=L)
    b_last = segments[-1][1]
    idx = _find_idx(C, b_last)
    expected = oracle_min_length_one_side(segments, h, L)
    assert math.isclose(Sigma[idx], expected, abs_tol=5e-5)


def test_dpos_case_three_transition(tol: float) -> None:
    segments = [
        (0.4963467851637333, 2.052148061669569),
        (3.2626003190253767, 4.733105703427381),
        (6.925295878470212, 7.62555150539123),
    ]
    h = 1.3343911582849786
    L = 16.165898616030113
    Sigma, C = dp_one_side_ref(segments, h=h, L=L)
    q = segments[-1][1]
    idx_q = _find_idx(C, q)
    p_star = find_maximal_p(q, h, L)
    assert segments[0][0] - tol <= p_star <= segments[0][1] + tol
    idx_p = _find_idx(C, p_star)

    option_case2 = math.inf
    right_indices = {
        segments[i][1]: _find_idx(C, segments[i][1]) for i in range(len(segments))
    }
    for j in range(1, len(segments)):
        length = tour_length(segments[j][0], q, h)
        if length > L + tol:
            continue
        prev = 0.0 if j == 0 else Sigma[right_indices[segments[j - 1][1]]]
        option_case2 = min(option_case2, prev + length)

    expected = min(L + Sigma[idx_p], option_case2)
    assert math.isclose(Sigma[idx_q], expected, abs_tol=5e-5)


def test_dpos_candidates_include_right_endpoints(tol: float) -> None:
    segments = [(0.0, 2.0), (3.0, 3.6), (5.5, 6.2)]
    h = 1.5
    L = 15.0
    Sigma, C = dp_one_side_ref(segments, h=h, L=L)
    for _, b in segments:
        _find_idx(C, b)
    q = segments[-1][1]
    p_star = find_maximal_p(q, h, L)
    if p_star >= 0.0:
        _find_idx(C, p_star)


def test_dpos_invalid_segment_length() -> None:
    segments = [(0.0, 5.0)]
    h = 1.0
    L = tour_length(0.0, 5.0, h) - 1.0
    with pytest.raises(ValueError):
        dp_one_side_ref(segments, h=h, L=L)


# ---------------------------------------------------------------------------
#  Hypothesis strategies
# ---------------------------------------------------------------------------
if HAVE_HYPOTHESIS:

    @composite
    def dpos_instances(draw):
        k = draw(st.integers(min_value=1, max_value=3))
        seed = draw(st.integers(min_value=0, max_value=99999))
        rnd = rng(seed)
        segments = gen_one_side_segments(
            rnd, k, x_lo=0.0, x_hi=15.0, min_len=0.8, min_gap=0.8
        )
        h = draw(st.floats(min_value=1.0, max_value=4.0))
        max_atomic = max(tour_length(a, b, h) for a, b in segments)
        factor = draw(st.floats(min_value=1.05, max_value=1.4))
        L = factor * max_atomic + draw(st.floats(min_value=0.3, max_value=2.0))
        return segments, h, L


    @settings(max_examples=50)
    @given(dpos_instances())
    def test_dpos_oracle_cross_check(data, tol: float) -> None:
        segments, h, L = data
        Sigma, C = dp_one_side_ref(segments, h=h, L=L)
        b_last = segments[-1][1]
        idx = _find_idx(C, b_last)
        oracle = oracle_min_length_one_side(segments, h, L)
        assert math.isclose(Sigma[idx], oracle, abs_tol=5e-5)


    @settings(max_examples=50)
    @given(dpos_instances())
    def test_dpos_monotonicity_in_L(data, tol: float) -> None:
        segments, h, L = data
        Sigma, C = dp_one_side_ref(segments, h=h, L=L)
        idx = _find_idx(C, segments[-1][1])
        larger = 1.2 * L
        Sigma2, C2 = dp_one_side_ref(segments, h=h, L=larger)
        idx2 = _find_idx(C2, segments[-1][1])
        assert Sigma2[idx2] <= Sigma[idx] + tol

else:  # pragma: no cover

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_dpos_oracle_cross_check(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")

    @pytest.mark.skip(reason="Hypothesis not installed")
    def test_dpos_monotonicity_in_L(tol: float) -> None:  # type: ignore[ref-assign]
        pytest.skip("Hypothesis not installed")


def test_dpos_prefix_costs_unchanged_with_extra_segment(tol: float) -> None:
    segments = [(0.0, 2.0), (3.2, 4.0)]
    extra = (6.5, 7.3)
    h = 2.0
    L = 18.0
    Sigma, C = dp_one_side_ref(segments, h=h, L=L)
    segments_extended = segments + [extra]
    Sigma_ext, C_ext = dp_one_side_ref(segments_extended, h=h, L=L)
    for seg in segments:
        idx = _find_idx(C, seg[1])
        idx_ext = _find_idx(C_ext, seg[1])
        assert math.isclose(Sigma[idx], Sigma_ext[idx_ext], abs_tol=tol)


# ---------------------------------------------------------------------------
#  Metamorphic tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [0.5, 3.0, 9.0])
def test_dpos_scaling(scale: float, tol: float) -> None:
    segments = [(0.0, 2.2), (3.4, 4.5)]
    h = 1.8
    L = 20.0
    Sigma, C = dp_one_side_ref(segments, h=h, L=L)
    idx = _find_idx(C, segments[-1][1])
    scaled_segments = scale_instance(segments, scale)
    Sigma_scaled, C_scaled = dp_one_side_ref(
        scaled_segments, h=h * scale, L=L * scale
    )
    idx_scaled = _find_idx(C_scaled, scaled_segments[-1][1])
    assert math.isclose(
        Sigma_scaled[idx_scaled], Sigma[idx] * scale, rel_tol=0.0, abs_tol=1e-5
    )


def test_dpos_large_battery_coarsens(tol: float) -> None:
    segments = [(0.0, 1.0), (2.0, 3.0), (4.5, 5.5)]
    h = 2.0
    base = tour_length(segments[0][0], segments[-1][1], h)
    L = base * 1.5
    Sigma, C = dp_one_side_ref(segments, h=h, L=L)
    idx = _find_idx(C, segments[-1][1])
    assert math.isclose(Sigma[idx], base, abs_tol=1e-5)
