from __future__ import annotations

import math
import random
from typing import List, Tuple

import pytest

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference.dp_one_side_ref import (
    dp_one_side_with_plan,
    reconstruct_one_side_plan,
)
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from tests.test_utils import (
    check_cover_exact,
    gen_one_side_segments,
    rng,
)


def _sample_instance(seed: int) -> Tuple[List[Tuple[float, float]], float, float]:
    rnd = rng(seed)
    segments = gen_one_side_segments(
        rnd,
        k=rnd.randint(2, 5),
        x_lo=0.0,
        x_hi=rnd.uniform(15.0, 40.0),
        min_len=0.8,
        min_gap=0.6,
    )
    h = rnd.uniform(2.0, 6.0)
    # Ensure generous battery so the solver is feasible.
    span_cost = tour_length(segments[0][0], segments[-1][1], h)
    L = span_cost * rnd.uniform(1.15, 1.6)
    return segments, h, L


@pytest.mark.parametrize("seed", list(range(10)))
def test_dp_one_side_plan_roundtrip(seed: int) -> None:
    segments, h, L = _sample_instance(seed)
    Sigma, C, plan = dp_one_side_with_plan(segments, h, L)

    tours = reconstruct_one_side_plan(C, plan)
    assert tours, "Plan reconstruction yielded no tours"

    total = sum(tour_length(p, q, h) for p, q in tours)
    assert math.isclose(total, Sigma[-1], rel_tol=1e-9, abs_tol=1e-7)

    # Coverage verification up to geometry tolerance.
    check_cover_exact(segments, tours, tol=EPS_GEOM + TOL_NUM)

    # Every tour must obey the battery constraint.
    for p, q in tours:
        assert tour_length(p, q, h) <= L + 1e-6
