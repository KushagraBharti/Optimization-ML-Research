from __future__ import annotations

import math
from typing import List, Tuple

import pytest

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import dpos
from coverage_planning.algs.reference.dp_full_line_ref import dp_full_line_with_plan
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from tests.test_utils import (
    check_cover_exact,
    gen_disjoint_segments,
    rng,
)


def _sample_two_sided_instance(seed: int) -> Tuple[List[Tuple[float, float]], float, float]:
    rnd = rng(seed)

    left_segments = gen_disjoint_segments(
        rnd,
        k=rnd.randint(1, 3),
        x_min=-rnd.uniform(25.0, 60.0),
        x_max=-rnd.uniform(5.0, 2.0),
        min_len=0.9,
        min_gap=0.8,
    )
    right_segments = gen_disjoint_segments(
        rnd,
        k=rnd.randint(1, 3),
        x_min=rnd.uniform(1.0, 3.0),
        x_max=rnd.uniform(15.0, 40.0),
        min_len=0.9,
        min_gap=0.8,
    )
    segments = sorted(left_segments + right_segments, key=lambda seg: seg[0])

    h = rnd.uniform(2.5, 6.0)
    farthest = max(max(abs(a), abs(b)) for a, b in segments)
    per_seg = max(tour_length(a, b, h) for a, b in segments)
    baseline = max(per_seg, 2.0 * math.hypot(farthest, h))
    L = baseline * rnd.uniform(1.1, 1.6)
    return segments, h, L


def _count_crossing_tours(tours: List[Tuple[float, float]]) -> int:
    return sum(1 for p, q in tours if p < -EPS_GEOM and q > EPS_GEOM)


@pytest.mark.parametrize("seed", list(range(12)))
def test_dp_full_line_plan_roundtrip(seed: int) -> None:
    segments, h, L = _sample_two_sided_instance(seed)

    cost, tours, meta = dp_full_line_with_plan(segments, h, L)
    assert tours, "Expected at least one tour from plan extraction"

    total = sum(tour_length(p, q, h) for p, q in tours)
    assert math.isclose(total, cost, rel_tol=1e-9, abs_tol=1e-7)

    check_cover_exact(segments, tours, tol=EPS_GEOM + TOL_NUM)
    for p, q in tours:
        assert tour_length(p, q, h) <= L + 1e-6

    # baseline check for bridge usefulness
    left = [seg for seg in segments if seg[1] <= 0.0]
    right = [seg for seg in segments if seg[0] >= 0.0]
    baseline = 0.0
    if left:
        left_ref = [(-b, -a) for a, b in reversed(left)]
        Sigma_L, _ = dpos(left_ref, h, L)
        baseline += Sigma_L[-1]
    if right:
        Sigma_R, _ = dpos(right, h, L)
        baseline += Sigma_R[-1]

    crossings = _count_crossing_tours(tours)
    if meta["best_mode"] == "bridge":
        assert crossings == 1, "Bridge solution must include exactly one cross-origin tour"
        assert cost <= baseline + 1e-6
    else:
        assert crossings == 0
        assert cost <= baseline + 1e-6
