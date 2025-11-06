from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple

from coverage_planning.visualization.algs.dp_full_line_viz import dp_full_with_plan
from coverage_planning.visualization.events import (
    AlgoInfoEvent,
    BridgeTryEvent,
    DPAddCandidateEvent,
    DPDoneEvent,
    DPStartEvent,
    DoneEvent,
    ReplayStartEvent,
)

from .common import build_scene_events, generate_tour_events

Segment = Tuple[float, float]


def _emit_one_side_dp(
    events: List[dict],
    trace: Optional[dict],
    *,
    side: Optional[str],
    candidate_transform: Callable[[float], float] = lambda x: x,
    stage: Optional[str] = None,
) -> None:
    if trace is None:
        return

    start_event: dict = DPStartEvent(type="dp_start", mode="one_side", side=side)
    if stage is not None:
        start_event["stage"] = stage
    events.append(start_event)

    per_candidate = trace.get("per_candidate", [])
    for idx, x in enumerate(trace.get("candidates", [])):
        x_transformed = candidate_transform(x)
        events.append(DPAddCandidateEvent(type="dp_add_candidate", idx=idx, x=float(x_transformed)))
        if idx < len(per_candidate):
            entry = per_candidate[idx]
            for transition in entry.get("tried_transitions", []):
                events.append(
                    {
                        "type": "dp_try_transition",
                        "from": transition.get("from_idx"),
                        "to": transition.get("to_idx"),
                        "case": transition.get("case", ""),
                        "length": float(transition.get("length", 0.0)),
                        "feasible": bool(transition.get("feasible", False)),
                    }
                )
            picked = entry.get("picked")
            if picked is not None:
                events.append(
                    {
                        "type": "dp_pick_transition",
                        "from": picked.get("prev_idx"),
                        "to": idx,
                        "cost": float(picked.get("cost", 0.0)),
                        "case": picked.get("case", ""),
                    }
                )

    events.append(DPDoneEvent(type="dp_done"))


def _emit_tail_dp(
    events: List[dict],
    trace: Optional[dict],
) -> None:
    if trace is None:
        return

    start_event: dict = DPStartEvent(type="dp_start", mode="one_side", side="right", stage="tail")
    events.append(start_event)

    per_candidate = trace.get("per_candidate", [])
    for idx, x in enumerate(trace.get("candidates", [])):
        events.append(DPAddCandidateEvent(type="dp_add_candidate", idx=idx, x=float(x)))
        if idx < len(per_candidate):
            entry = per_candidate[idx]
            for transition in entry.get("tried_transitions", []):
                events.append(
                    {
                        "type": "dp_try_transition",
                        "from": idx,
                        "to": transition.get("next_idx"),
                        "case": transition.get("case", ""),
                        "length": float(transition.get("length", 0.0)),
                        "feasible": bool(transition.get("feasible", False)),
                    }
                )
            picked = entry.get("picked")
            if picked is not None:
                events.append(
                    {
                        "type": "dp_pick_transition",
                        "from": idx,
                        "to": picked.get("next_idx"),
                        "cost": float(picked.get("cost", 0.0)),
                        "case": picked.get("case", ""),
                    }
                )

    events.append(DPDoneEvent(type="dp_done"))


def build_dp_full_events(
    segments: Iterable[Segment],
    h: float,
    L: float,
) -> List[dict]:
    seg_list = list(segments)
    events: List[dict] = build_scene_events(seg_list, h, L)

    cost, tours, meta, trace = dp_full_with_plan(seg_list, h, L, trace=True)
    best_mode = meta.get("best_mode") if isinstance(meta, dict) else None
    events.append(AlgoInfoEvent(type="algo_info", name="dp_full", case=best_mode))

    left_trace = trace.get("left_trace") if isinstance(trace, dict) else None
    right_trace = trace.get("right_trace") if isinstance(trace, dict) else None
    tail_trace = trace.get("tail_trace") if isinstance(trace, dict) else None

    _emit_one_side_dp(
        events,
        left_trace,
        side="left",
        candidate_transform=lambda x: -float(x),
    )
    _emit_one_side_dp(events, right_trace, side="right")
    _emit_tail_dp(events, tail_trace)

    for attempt in trace.get("bridge_attempts", []) if isinstance(trace, dict) else []:
        bridge_event: dict = BridgeTryEvent(
            type="bridge_try",
            P=float(attempt.get("P", 0.0)),
            q=float(attempt.get("q", 0.0)),
            accepted=bool(attempt.get("accepted", False)),
        )
        if "selected" in attempt:
            bridge_event["selected"] = bool(attempt["selected"])
        events.append(bridge_event)

    events.append(ReplayStartEvent(type="replay_start"))
    for idx, (p, q) in enumerate(tours):
        events.extend(generate_tour_events(idx, p, q))

    events.append(DoneEvent(type="done"))
    return events


__all__ = ["build_dp_full_events"]

