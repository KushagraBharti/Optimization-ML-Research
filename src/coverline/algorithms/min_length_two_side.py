from __future__ import annotations

from dataclasses import dataclass

from ..candidates import build_candidate_sets_one_side
from ..geometry import feasible_tour, tour_length
from ..model import EPS, Instance, Solution, Tour
from .dpos_one_side import solve_min_length_one_side_dpos


@dataclass(frozen=True)
class TwoSideCaseResult:
    name: str
    solution: Solution


def _combine_solutions(name: str, left: Solution, middle: Tour | None, right: Solution) -> Solution:
    tours = list(left.tours)
    if middle is not None:
        tours.append(middle)
    tours.extend(right.tours)
    return Solution.from_tours(
        tours,
        metadata={
            "algorithm": "min_length_two_side",
            "case": name,
            "left_tours": left.tour_count,
            "right_tours": right.tour_count,
            "middle": None if middle is None else (middle.left, middle.right),
        },
    )


def _independent_case(instance: Instance) -> TwoSideCaseResult:
    left = solve_min_length_one_side_dpos(instance.clipped(right=0.0))
    right = solve_min_length_one_side_dpos(instance.clipped(left=0.0))
    combined = _combine_solutions("independent", left, None, right)
    return TwoSideCaseResult(name="independent", solution=combined)


def _interior_case(instance: Instance) -> TwoSideCaseResult | None:
    left_instance = instance.clipped(right=0.0)
    right_instance = instance.clipped(left=0.0)
    if left_instance.is_empty or right_instance.is_empty:
        return None

    c_left = [x for x in build_candidate_sets_one_side(left_instance).union if x < -EPS]
    c_right = [x for x in build_candidate_sets_one_side(right_instance).union if x > EPS]
    if not c_left or not c_right:
        return None

    left_cache: dict[float, Solution] = {}
    right_cache: dict[float, Solution] = {}
    best: Solution | None = None
    best_pair: tuple[float, float] | None = None

    for p in c_left:
        for q in c_right:
            if not feasible_tour(p, q, instance.h, instance.L):
                continue
            if p not in left_cache:
                left_cache[p] = solve_min_length_one_side_dpos(instance.clipped(right=p))
            if q not in right_cache:
                right_cache[q] = solve_min_length_one_side_dpos(instance.clipped(left=q))

            center_len = tour_length(p, q, instance.h)
            center_tour = Tour(
                left=p,
                right=q,
                length=center_len,
                maximal=abs(center_len - instance.L) <= 1e-6,
                tag="two_side_interior",
            )
            candidate = _combine_solutions(
                "interior_projection",
                left_cache[p],
                center_tour,
                right_cache[q],
            )
            if best is None or candidate.total_length < best.total_length - EPS:
                best = candidate
                best_pair = (p, q)

    if best is None:
        return None
    meta = dict(best.metadata)
    meta["pair"] = best_pair
    meta["c_left"] = len(c_left)
    meta["c_right"] = len(c_right)
    best = Solution(
        tours=best.tours,
        total_length=best.total_length,
        tour_count=best.tour_count,
        metadata=meta,
    )
    return TwoSideCaseResult(name="interior_projection", solution=best)


def solve_min_length_two_side(instance: Instance) -> Solution:
    if instance.is_empty:
        return Solution.from_tours([], metadata={"algorithm": "min_length_two_side"})
    if instance.one_side():
        sol = solve_min_length_one_side_dpos(instance)
        meta = dict(sol.metadata)
        meta["algorithm"] = "min_length_two_side"
        meta["case"] = "one_side_reduction"
        return Solution(
            tours=sol.tours,
            total_length=sol.total_length,
            tour_count=sol.tour_count,
            metadata=meta,
        )

    candidates: list[TwoSideCaseResult] = [_independent_case(instance)]
    interior = _interior_case(instance)
    if interior is not None:
        candidates.append(interior)

    best = min(candidates, key=lambda r: r.solution.total_length)
    meta = dict(best.solution.metadata)
    meta["case_candidates"] = [c.name for c in candidates]
    return Solution(
        tours=best.solution.tours,
        total_length=best.solution.total_length,
        tour_count=best.solution.tour_count,
        metadata=meta,
    )


def solve_min_length_two_side_reference(instance: Instance) -> Solution:
    """Reference pair-enumeration solver for small instances."""
    return solve_min_length_two_side(instance)
