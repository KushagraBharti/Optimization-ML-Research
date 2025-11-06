from __future__ import annotations

from typing import Iterable, List, Tuple

from coverage_planning.visualization.algs.gsp_viz import gsp
from coverage_planning.visualization.events import AlgoInfoEvent, DoneEvent, ErrorEvent

from .common import build_scene_events, generate_tour_events

Segment = Tuple[float, float]


def build_gsp_events(
    segments: Iterable[Segment],
    h: float,
    L: float,
) -> List[dict]:
    seg_list = list(segments)
    if not seg_list:
        return [
            *build_scene_events([], h, L),
            ErrorEvent(type="error", text="No segments provided for GSP"),
            DoneEvent(type="done"),
        ]

    events: List[dict] = build_scene_events(seg_list, h, L)

    seg = seg_list[0]
    _, tours, trace = gsp(seg, h, L, trace=True)
    case = trace.get("case") if isinstance(trace, dict) else None

    events.append(AlgoInfoEvent(type="algo_info", name="gsp", case=case))
    for idx, (p, q) in enumerate(tours):
        events.extend(generate_tour_events(idx, p, q))

    events.append(DoneEvent(type="done"))
    return events


__all__ = ["build_gsp_events"]

