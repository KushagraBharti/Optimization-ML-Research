from __future__ import annotations

from typing import Iterable, List, Tuple

from coverage_planning.visualization.events import (
    AddSegmentEvent,
    AnimateTourEvent,
    EndTourEvent,
    MarkCoveredEvent,
    SetSceneEvent,
    StartTourEvent,
    classify_segment,
    compute_scene_bounds,
)

Segment = Tuple[float, float]

SCENE_MARGIN = 0.1
ANIMATION_STEPS_PER_PHASE = 20


def build_scene_events(
    segments: Iterable[Segment],
    h: float,
    L: float,
    *,
    margin: float = SCENE_MARGIN,
) -> List[dict]:
    seg_list = list(segments)
    x_min, x_max = compute_scene_bounds(seg_list, margin=margin)
    events: List[dict] = [
        SetSceneEvent(type="set_scene", h=float(h), L=float(L), x_min=float(x_min), x_max=float(x_max))
    ]
    for idx, (x1, x2) in enumerate(seg_list):
        side = classify_segment(x1, x2)
        events.append(
            AddSegmentEvent(
                type="add_segment",
                id=f"seg{idx}",
                x1=float(x1),
                x2=float(x2),
                side=side,
            )
        )
    return events


def generate_tour_events(
    tour_id: int,
    p: float,
    q: float,
    *,
    steps_per_phase: int = ANIMATION_STEPS_PER_PHASE,
) -> List[dict]:
    events: List[dict] = [
        StartTourEvent(type="start_tour", tour_id=tour_id, p=float(p), q=float(q))
    ]
    for phase in ("up", "across", "down"):
        for step in range(steps_per_phase):
            progress = (step + 1) / steps_per_phase
            events.append(
                AnimateTourEvent(
                    type="animate_tour",
                    tour_id=tour_id,
                    phase=phase,
                    progress=float(progress),
                )
            )
    x1, x2 = sorted((float(p), float(q)))
    events.append(MarkCoveredEvent(type="mark_covered", x1=x1, x2=x2))
    events.append(EndTourEvent(type="end_tour", tour_id=tour_id))
    return events


__all__ = ["build_scene_events", "generate_tour_events", "SCENE_MARGIN", "ANIMATION_STEPS_PER_PHASE"]

