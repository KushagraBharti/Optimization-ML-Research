"""Paper-faithful reference solvers for coverage planning."""

from __future__ import annotations

from coverage_planning.algs.reference.dp_full_line_ref import (
    FullLinePlanContext,
    TailPlan,
    dp_full_line_ref,
    dp_full_line_with_plan,
    dp_one_side_tail_with_plan,
    find_maximal_bridge_p,
    reconstruct_tail_plan,
)
from coverage_planning.algs.reference.dp_one_side_ref import (
    dp_one_side_ref,
    dp_one_side_with_plan,
    reconstruct_one_side_plan,
)
from coverage_planning.algs.reference.greedy_ref import greedy_min_tours_ref
from coverage_planning.algs.reference.gsp_ref import greedy_min_length_one_segment_ref

gs = greedy_min_tours_ref
gsp = greedy_min_length_one_segment_ref
dpos = dp_one_side_ref
dp_full = dp_full_line_ref
dp_full_with_plan = dp_full_line_with_plan

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
]
