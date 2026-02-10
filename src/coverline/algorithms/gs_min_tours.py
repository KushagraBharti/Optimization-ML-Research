from __future__ import annotations

from dataclasses import dataclass

from ..coverage import subtract_interval_from_segments
from ..geometry import (
    can_cover_all_with_one_tour,
    choose_farthest_endpoint,
    solve_maximal_left_endpoint,
    solve_maximal_right_endpoint,
    tour_length,
)
from ..model import EPS, Instance, Solution, Tour


@dataclass(frozen=True)
class GreedyStep:
    remaining_before: tuple[tuple[float, float], ...]
    chosen_tour: tuple[float, float]
    farthest: float
    one_tour_finish: bool


def _remaining_signature(segments) -> tuple[tuple[float, float], ...]:
    return tuple((s.a, s.b) for s in segments)


def _solve_min_tours_gs(instance: Instance, mode: str) -> Solution:
    if instance.is_empty:
        return Solution.from_tours([], metadata={"algorithm": mode, "steps": []})

    remaining = instance.segments
    tours: list[Tour] = []
    steps: list[GreedyStep] = []

    while remaining:
        min_x = remaining[0].a
        max_x = remaining[-1].b

        if can_cover_all_with_one_tour(min_x, max_x, instance.h, instance.L):
            length = tour_length(min_x, max_x, instance.h)
            t = Tour(
                left=min_x,
                right=max_x,
                length=length,
                maximal=abs(length - instance.L) <= EPS,
                tag="gs-finish",
            )
            steps.append(
                GreedyStep(
                    remaining_before=_remaining_signature(remaining),
                    chosen_tour=(min_x, max_x),
                    farthest=choose_farthest_endpoint(min_x, max_x, instance.h),
                    one_tour_finish=True,
                )
            )
            tours.append(t)
            break

        farthest = choose_farthest_endpoint(min_x, max_x, instance.h)
        if abs(farthest - max_x) <= EPS:
            right = max_x
            left = solve_maximal_left_endpoint(right=right, h=instance.h, L=instance.L)
        else:
            left = min_x
            right = solve_maximal_right_endpoint(left=left, h=instance.h, L=instance.L)

        length = tour_length(left, right, instance.h)
        t = Tour(
            left=min(left, right),
            right=max(left, right),
            length=length,
            maximal=abs(length - instance.L) <= 1e-6,
            tag="gs-max",
        )
        steps.append(
            GreedyStep(
                remaining_before=_remaining_signature(remaining),
                chosen_tour=(t.left, t.right),
                farthest=farthest,
                one_tour_finish=False,
            )
        )
        tours.append(t)
        remaining = subtract_interval_from_segments(remaining, t.left, t.right)

    return Solution.from_tours(
        tours,
        metadata={
            "algorithm": mode,
            "steps": [step.__dict__ for step in steps],
        },
    )


def solve_min_tours_gs(instance: Instance) -> Solution:
    """Greedy strategy for MinTours; binary-search variant."""
    return _solve_min_tours_gs(instance, mode="gs_min_tours_log")


def solve_min_tours_gs_linear(instance: Instance) -> Solution:
    """Greedy strategy for MinTours; linear-sweep variant."""
    # The state updates are identical; this entrypoint mirrors the linear variant API.
    return _solve_min_tours_gs(instance, mode="gs_min_tours_linear")
