from __future__ import annotations

from typing import Iterable, List, Tuple

from coverage_planning.visualization.algs.greedy_viz import gs
from coverage_planning.visualization.events import AlgoInfoEvent, DoneEvent

from .common import build_scene_events, generate_tour_events

Segment = Tuple[float, float]


def build_gs_events(
    segments: Iterable[Segment],
    h: float,
    L: float,
) -> List[dict]:
    seg_list = list(segments)
    events: List[dict] = build_scene_events(seg_list, h, L)
    events.append(AlgoInfoEvent(type="algo_info", name="gs", case=None))

    _, tours, _trace = gs(seg_list, h, L, trace=True)

    for idx, (p, q) in enumerate(tours):
        events.extend(generate_tour_events(idx, p, q))

    events.append(DoneEvent(type="done"))
    return events


__all__ = ["build_gs_events"]

