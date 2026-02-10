from __future__ import annotations

from ..geometry import (
    can_cover_interval_with_one_tour,
    choose_farthest_endpoint,
    solve_maximal_left_endpoint,
    solve_maximal_right_endpoint,
    tour_length,
)
from ..model import EPS, Instance, Solution, Tour


def solve_min_length_one_segment_gsp(instance: Instance) -> Solution:
    """Projection-aware greedy strategy for one-segment MinLength."""
    if instance.is_empty:
        return Solution.from_tours([], metadata={"algorithm": "gsp_one_segment"})
    if len(instance.segments) != 1:
        raise ValueError("GSP is defined for exactly one segment.")

    a = instance.segments[0].a
    b = instance.segments[0].b
    tours: list[Tour] = []
    steps: list[dict[str, float | str]] = []
    guard = 0
    reached_projection = False

    while b >= a + EPS:
        guard += 1
        if guard > 10_000:
            raise RuntimeError("GSP loop guard triggered.")

        if can_cover_interval_with_one_tour(a, b, instance.h, instance.L):
            length = tour_length(a, b, instance.h)
            tours.append(
                Tour(
                    left=a,
                    right=b,
                    length=length,
                    maximal=abs(length - instance.L) <= 1e-6,
                    tag="gsp-final-one-tour",
                )
            )
            steps.append({"action": "final_one", "a": a, "b": b, "length": length})
            break

        farthest = choose_farthest_endpoint(a, b, instance.h)
        if abs(farthest - b) <= EPS:
            left = solve_maximal_left_endpoint(right=b, h=instance.h, L=instance.L)
            if left <= 0 <= b:
                reached_projection = True
                break
            length = tour_length(left, b, instance.h)
            tours.append(Tour(left=left, right=b, length=length, maximal=True, tag="gsp-max-right"))
            steps.append({"action": "max_from_right", "left": left, "right": b, "length": length})
            b = left
        else:
            right = solve_maximal_right_endpoint(left=a, h=instance.h, L=instance.L)
            if a <= 0 <= right:
                reached_projection = True
                break
            length = tour_length(a, right, instance.h)
            tours.append(Tour(left=a, right=right, length=length, maximal=True, tag="gsp-max-left"))
            steps.append({"action": "max_from_left", "left": a, "right": right, "length": length})
            a = right

    if b >= a + EPS and reached_projection:
        if can_cover_interval_with_one_tour(a, b, instance.h, instance.L):
            length = tour_length(a, b, instance.h)
            tours.append(
                Tour(
                    left=a,
                    right=b,
                    length=length,
                    maximal=abs(length - instance.L) <= 1e-6,
                    tag="gsp-final-one-after-projection",
                )
            )
            steps.append({"action": "final_one_after_projection", "a": a, "b": b, "length": length})
        else:
            length_left = tour_length(a, 0.0, instance.h)
            length_right = tour_length(0.0, b, instance.h)
            if length_left > instance.L + EPS or length_right > instance.L + EPS:
                raise RuntimeError("GSP final projection split produced infeasible tours.")
            tours.append(
                Tour(
                    left=a,
                    right=0.0,
                    length=length_left,
                    maximal=abs(length_left - instance.L) <= 1e-6,
                    tag="gsp-final-two-left",
                )
            )
            tours.append(
                Tour(
                    left=0.0,
                    right=b,
                    length=length_right,
                    maximal=abs(length_right - instance.L) <= 1e-6,
                    tag="gsp-final-two-right",
                )
            )
            steps.append(
                {
                    "action": "final_two_with_projection",
                    "a": a,
                    "b": b,
                    "length_left": length_left,
                    "length_right": length_right,
                }
            )

    return Solution.from_tours(
        tours,
        metadata={"algorithm": "gsp_one_segment", "steps": steps},
    )
