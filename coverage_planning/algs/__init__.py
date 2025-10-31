"""Algorithm package entry points with reference solvers exposed by default."""

from __future__ import annotations

import coverage_planning.algs.heuristics as heuristics
import coverage_planning.algs.reference as reference
from coverage_planning.algs.heuristics import (
    dp_full_line as legacy_dp_full,
    dp_one_side as legacy_dpos,
    greedy_min_length_one_segment as legacy_gsp,
    greedy_min_tours as legacy_gs,
)
from coverage_planning.algs.reference import (
    FullLinePlanContext,
    TailPlan,
    dp_full,
    dp_full_with_plan,
    dp_one_side_tail_with_plan,
    dp_one_side_with_plan,
    dpos,
    find_maximal_bridge_p,
    gsp,
    gs,
    reconstruct_one_side_plan,
    reconstruct_tail_plan,
)

__all__ = [
    "gs",
    "gsp",
    "dpos",
    "dp_full",
    "dp_full_with_plan",
    "dp_one_side_with_plan",
    "dp_one_side_tail_with_plan",
    "reconstruct_one_side_plan",
    "reconstruct_tail_plan",
    "FullLinePlanContext",
    "TailPlan",
    "find_maximal_bridge_p",
    "reference",
    "heuristics",
    "legacy_gs",
    "legacy_gsp",
    "legacy_dpos",
    "legacy_dp_full",
]
