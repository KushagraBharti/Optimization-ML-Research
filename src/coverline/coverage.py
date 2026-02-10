from __future__ import annotations

from dataclasses import dataclass

from .geometry import feasible_tour
from .model import EPS, Instance, Segment, Solution, ValidationReport


def _trim_segment(a: float, b: float) -> Segment | None:
    if b < a - EPS:
        return None
    return Segment(a, b)


def subtract_interval_from_segment(segment: Segment, left: float, right: float) -> tuple[Segment, ...]:
    if right < segment.a + EPS or left > segment.b - EPS:
        return (segment,)
    pieces: list[Segment] = []
    if left > segment.a + EPS:
        s = _trim_segment(segment.a, min(left, segment.b))
        if s is not None:
            pieces.append(s)
    if right < segment.b - EPS:
        s = _trim_segment(max(right, segment.a), segment.b)
        if s is not None:
            pieces.append(s)
    return tuple(pieces)


def subtract_interval_from_segments(
    segments: tuple[Segment, ...],
    left: float,
    right: float,
) -> tuple[Segment, ...]:
    if left > right:
        left, right = right, left
    result: list[Segment] = []
    for s in segments:
        result.extend(subtract_interval_from_segment(s, left, right))
    return tuple(result)


def covered_intersections(
    segments: tuple[Segment, ...],
    left: float,
    right: float,
) -> tuple[Segment, ...]:
    if left > right:
        left, right = right, left
    out: list[Segment] = []
    for s in segments:
        a = max(s.a, left)
        b = min(s.b, right)
        if b >= a - EPS:
            out.append(Segment(a, b))
    return tuple(out)


def uncovered_after_solution(instance: Instance, solution: Solution) -> tuple[Segment, ...]:
    uncovered = instance.segments
    for t in solution.tours:
        uncovered = subtract_interval_from_segments(uncovered, t.left, t.right)
    return uncovered


def validate_solution(instance: Instance, solution: Solution) -> ValidationReport:
    infeasible: list[int] = []
    notes: list[str] = []
    for idx, t in enumerate(solution.tours):
        if not feasible_tour(t.left, t.right, instance.h, instance.L):
            infeasible.append(idx)
    uncovered = uncovered_after_solution(instance, solution)
    valid = len(infeasible) == 0 and all(s.length <= EPS for s in uncovered)
    if infeasible:
        notes.append(f"Found {len(infeasible)} infeasible tours.")
    if any(s.length > EPS for s in uncovered):
        notes.append("Not all segment points are covered.")
    return ValidationReport(
        valid=valid,
        uncovered_segments=tuple(s for s in uncovered if s.length > EPS),
        infeasible_tours=tuple(infeasible),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class SweepState:
    remaining: tuple[Segment, ...]
    tours: tuple[tuple[float, float], ...]
