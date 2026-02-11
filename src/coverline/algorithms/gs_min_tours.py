from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

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
    left_x: float
    right_x: float
    left_idx: int
    right_idx: int
    farthest: float
    chosen_tour: tuple[float, float]
    one_tour_finish: bool


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _solve_min_tours_gs_bisect(instance: Instance, mode: str) -> Solution:
    if instance.is_empty:
        return Solution.from_tours([], metadata={"algorithm": mode, "steps": []})

    segments = instance.segments
    a = [s.a for s in segments]
    b = [s.b for s in segments]

    left_idx = 0
    right_idx = len(segments) - 1
    left_x = a[left_idx]
    right_x = b[right_idx]

    tours: list[Tour] = []
    steps: list[GreedyStep] = []

    guard = 0
    while left_idx <= right_idx:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("GS loop guard triggered.")

        if can_cover_all_with_one_tour(left_x, right_x, instance.h, instance.L):
            length = tour_length(left_x, right_x, instance.h)
            t = Tour(
                left=left_x,
                right=right_x,
                length=length,
                maximal=abs(length - instance.L) <= EPS,
                tag="gs-finish",
            )
            steps.append(
                GreedyStep(
                    left_x=left_x,
                    right_x=right_x,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    farthest=choose_farthest_endpoint(left_x, right_x, instance.h),
                    chosen_tour=(left_x, right_x),
                    one_tour_finish=True,
                )
            )
            tours.append(t)
            break

        farthest = choose_farthest_endpoint(left_x, right_x, instance.h)
        if abs(farthest - right_x) <= EPS:
            right = right_x
            left = solve_maximal_left_endpoint(right=right, h=instance.h, L=instance.L)
        else:
            left = left_x
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
                left_x=left_x,
                right_x=right_x,
                left_idx=left_idx,
                right_idx=right_idx,
                chosen_tour=(t.left, t.right),
                farthest=farthest,
                one_tour_finish=False,
            )
        )
        tours.append(t)

        # Update remaining boundaries by removing the tour-covered suffix/prefix.
        if abs(farthest - right_x) <= EPS:
            # Remove [left, right_x]. Remaining is S ∩ (-∞, left] (closure).
            if left <= left_x + EPS:
                break
            idx = bisect_right(a, left, lo=left_idx, hi=right_idx + 1) - 1
            if idx < left_idx:
                break
            if left <= b[idx] + EPS:
                right_idx = idx
                right_x = _clamp(left, a[idx], b[idx])
            else:
                right_idx = idx
                right_x = b[idx]
        else:
            # Remove [left_x, right]. Remaining is S ∩ [right, ∞) (closure).
            if right >= right_x - EPS:
                break
            idx = bisect_right(a, right, lo=left_idx, hi=right_idx + 1) - 1
            if idx < left_idx:
                left_idx = left_idx
                left_x = a[left_idx]
            elif right <= b[idx] + EPS:
                left_idx = idx
                left_x = _clamp(right, a[idx], b[idx])
            else:
                left_idx = idx + 1
                if left_idx > right_idx:
                    break
                left_x = a[left_idx]

    return Solution.from_tours(
        tours,
        metadata={
            "algorithm": mode,
            "steps": [step.__dict__ for step in steps],
        },
    )


def solve_min_tours_gs(instance: Instance) -> Solution:
    """Greedy strategy (GS) for MinTours; binary-search variant (Theorem 2)."""
    return _solve_min_tours_gs_bisect(instance, mode="gs_min_tours_log")


def solve_min_tours_gs_linear(instance: Instance) -> Solution:
    """Greedy strategy (GS) for MinTours; linear-sweep variant (Theorem 2)."""
    if instance.is_empty:
        return Solution.from_tours([], metadata={"algorithm": "gs_min_tours_linear", "steps": []})

    segments = instance.segments
    a = [s.a for s in segments]
    b = [s.b for s in segments]

    left_idx = 0
    right_idx = len(segments) - 1
    left_x = a[left_idx]
    right_x = b[right_idx]

    tours: list[Tour] = []
    steps: list[GreedyStep] = []

    guard = 0
    while left_idx <= right_idx:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("GS linear loop guard triggered.")

        if can_cover_all_with_one_tour(left_x, right_x, instance.h, instance.L):
            length = tour_length(left_x, right_x, instance.h)
            t = Tour(
                left=left_x,
                right=right_x,
                length=length,
                maximal=abs(length - instance.L) <= EPS,
                tag="gs-finish",
            )
            steps.append(
                GreedyStep(
                    left_x=left_x,
                    right_x=right_x,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    farthest=choose_farthest_endpoint(left_x, right_x, instance.h),
                    chosen_tour=(left_x, right_x),
                    one_tour_finish=True,
                )
            )
            tours.append(t)
            break

        farthest = choose_farthest_endpoint(left_x, right_x, instance.h)
        if abs(farthest - right_x) <= EPS:
            right = right_x
            left = solve_maximal_left_endpoint(right=right, h=instance.h, L=instance.L)
            length = tour_length(left, right, instance.h)
            t = Tour(
                left=left,
                right=right,
                length=length,
                maximal=abs(length - instance.L) <= 1e-6,
                tag="gs-max",
            )
            steps.append(
                GreedyStep(
                    left_x=left_x,
                    right_x=right_x,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    farthest=farthest,
                    chosen_tour=(left, right),
                    one_tour_finish=False,
                )
            )
            tours.append(t)
            if left <= left_x + EPS:
                break
            while right_idx >= left_idx and left < a[right_idx] - EPS:
                right_idx -= 1
            if right_idx < left_idx:
                break
            if left <= b[right_idx] + EPS:
                right_x = _clamp(left, a[right_idx], b[right_idx])
            else:
                right_x = b[right_idx]
        else:
            left = left_x
            right = solve_maximal_right_endpoint(left=left, h=instance.h, L=instance.L)
            length = tour_length(left, right, instance.h)
            t = Tour(
                left=left,
                right=right,
                length=length,
                maximal=abs(length - instance.L) <= 1e-6,
                tag="gs-max",
            )
            steps.append(
                GreedyStep(
                    left_x=left_x,
                    right_x=right_x,
                    left_idx=left_idx,
                    right_idx=right_idx,
                    farthest=farthest,
                    chosen_tour=(left, right),
                    one_tour_finish=False,
                )
            )
            tours.append(t)
            if right >= right_x - EPS:
                break
            while left_idx <= right_idx and right > b[left_idx] + EPS:
                left_idx += 1
            if left_idx > right_idx:
                break
            if right >= a[left_idx] - EPS:
                left_x = _clamp(right, a[left_idx], b[left_idx])
            else:
                left_x = a[left_idx]

    return Solution.from_tours(
        tours,
        metadata={
            "algorithm": "gs_min_tours_linear",
            "steps": [step.__dict__ for step in steps],
        },
    )
