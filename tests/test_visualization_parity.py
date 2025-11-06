import math
from typing import Iterable, List, Sequence, Tuple

import pytest

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference.dp_full_line_ref import dp_full_line_with_plan
from coverage_planning.algs.reference.dp_one_side_ref import dp_one_side_with_plan as dp_one_side_with_plan_ref
from coverage_planning.algs.reference.dp_one_side_ref import (
    reconstruct_one_side_plan as reconstruct_one_side_plan_ref,
)
from coverage_planning.algs.reference.gsp_single_ref import greedy_min_length_one_segment_ref
from coverage_planning.algs.reference.gs_mintours_ref import greedy_min_tours_ref
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.visualization.algs.dp_full_line_viz import dp_full_with_plan
from coverage_planning.visualization.algs.dp_one_side_viz import (
    dpos_with_plan,
    reconstruct_one_side_plan,
)
from coverage_planning.visualization.algs.gsp_viz import gsp
from coverage_planning.visualization.algs.greedy_viz import gs


def _sorted_tours(tours: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return sorted((float(p), float(q)) for p, q in tours)


def _assert_tours_close(
    lhs: Sequence[Tuple[float, float]],
    rhs: Sequence[Tuple[float, float]],
    *,
    tol: float = 1e-7,
) -> None:
    lhs_sorted = _sorted_tours(lhs)
    rhs_sorted = _sorted_tours(rhs)
    assert len(lhs_sorted) == len(rhs_sorted)
    for (p1, q1), (p2, q2) in zip(lhs_sorted, rhs_sorted):
        assert math.isclose(p1, p2, abs_tol=tol)
        assert math.isclose(q1, q2, abs_tol=tol)


GS_CASES = [
    ([(0.0, 2.0)], 2.0, 40.0),
    ([(1.0, 2.0), (5.0, 6.5)], 1.5, 18.0),
    ([(-6.0, -4.0), (-2.0, -1.0), (2.5, 3.5)], 3.0, 60.0),
]


@pytest.mark.parametrize("segments,h,L", GS_CASES)
def test_gs_parity(segments, h, L) -> None:
    k_ref, tours_ref = greedy_min_tours_ref(segments, h, L)
    k_viz, tours_viz, trace = gs(segments, h, L, trace=True)

    assert k_ref == k_viz
    _assert_tours_close(tours_ref, tours_viz, tol=EPS_GEOM * 10)
    assert isinstance(trace, list)
    assert len(trace) >= len(tours_viz)


GSP_CASES = [
    ((-2.0, 2.0), 4.0, 200.0),
    ((-8.0, 9.0), 4.0, 35.0),
    ((5.0, 11.0), 2.0, 25.0),
]


@pytest.mark.parametrize("segment,h,L", GSP_CASES)
def test_gsp_parity(segment, h, L) -> None:
    k_ref, tours_ref = greedy_min_length_one_segment_ref(segment, h, L)
    k_viz, tours_viz, trace = gsp(segment, h, L, trace=True)

    assert k_ref == k_viz
    _assert_tours_close(tours_ref, tours_viz, tol=EPS_GEOM * 10)
    assert trace["case"] in {"single", "central", "multi"}
    assert isinstance(trace["steps"], list)


DPOS_CASES = [
    ([(0.0, 4.0)], 2.5, 30.0),
    ([(0.0, 3.0), (5.0, 7.0)], 2.0, 30.0),
    ([(1.0, 2.5), (4.0, 5.5), (8.0, 9.75)], 3.0, 40.0),
]


@pytest.mark.parametrize("segments,h,L", DPOS_CASES)
def test_dpos_parity(segments, h, L) -> None:
    Sigma_ref, C_ref, plan_ref = dp_one_side_with_plan_ref(segments, h, L)
    tours_ref = reconstruct_one_side_plan_ref(C_ref, plan_ref)

    Sigma_viz, C_viz, plan_viz, trace = dpos_with_plan(segments, h, L, trace=True)
    tours_viz = reconstruct_one_side_plan(C_viz, plan_viz)

    assert len(C_ref) == len(C_viz)
    for left, right in zip(C_ref, C_viz):
        assert math.isclose(left, right, abs_tol=TOL_NUM)
    assert math.isclose(Sigma_ref[-1], Sigma_viz[-1], abs_tol=TOL_NUM)
    _assert_tours_close(tours_ref, tours_viz, tol=EPS_GEOM * 10)
    assert len(trace["candidates"]) == len(C_viz)
    for recorded, actual in zip(trace["candidates"], C_viz):
        assert math.isclose(recorded, actual, abs_tol=TOL_NUM)
    assert len(trace["per_candidate"]) == len(C_viz)


DP_FULL_CASES = [
    ([(-5.0, -3.0), (-1.5, -0.5), (1.0, 2.0), (4.0, 5.0)], 2.5, 40.0),
    ([(-6.0, -3.5), (2.5, 3.5), (6.0, 7.5)], 3.0, 50.0),
    ([(-4.0, -2.0), (0.5, 1.5), (3.5, 4.5), (6.0, 7.0)], 2.0, 45.0),
]


@pytest.mark.parametrize("segments,h,L", DP_FULL_CASES)
def test_dp_full_parity(segments, h, L) -> None:
    cost_ref, tours_ref, meta_ref = dp_full_line_with_plan(segments, h, L)
    cost_viz, tours_viz, meta_viz, trace = dp_full_with_plan(segments, h, L, trace=True)

    assert math.isclose(cost_ref, cost_viz, abs_tol=TOL_NUM)
    _assert_tours_close(tours_ref, tours_viz, tol=EPS_GEOM * 10)

    total_length = sum(tour_length(p, q, h) for p, q in tours_viz)
    assert math.isclose(total_length, cost_viz, abs_tol=TOL_NUM)

    assert meta_ref["best_mode"] == meta_viz["best_mode"]
    assert trace["final_mode"] == meta_viz["best_mode"]
    assert isinstance(trace["bridge_attempts"], list)
    if meta_viz["best_mode"] == "bridge":
        assert trace["bridge_attempts"], "bridge solution expected attempts recorded"
