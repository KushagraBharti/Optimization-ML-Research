from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from coverage_planning.visualization.algs.dp_one_side_viz import (
    dpos_with_plan,
    reconstruct_one_side_plan,
)
from coverage_planning.visualization.events import (
    AlgoInfoEvent,
    DPAddCandidateEvent,
    DPDoneEvent,
    DPStartEvent,
    DoneEvent,
    ReplayStartEvent,
)

from .common import build_scene_events, generate_tour_events

Segment = Tuple[float, float]


def _detect_side(segments: List[Segment]) -> Optional[str]:
    if not segments:
        return None
    if all(b <= 0.0 for _, b in segments):
        return "left"
    if all(a >= 0.0 for a, _ in segments):
        return "right"
    return None


def build_dpos_events(
    segments: Iterable[Segment],
    h: float,
    L: float,
) -> List[dict]:
    seg_list = list(segments)
    events: List[dict] = build_scene_events(seg_list, h, L)
    events.append(AlgoInfoEvent(type="algo_info", name="dpos", case=None))

    Sigma, candidates, plan, trace = dpos_with_plan(seg_list, h, L, trace=True)

    side = _detect_side(seg_list)
    events.append(DPStartEvent(type="dp_start", mode="one_side", side=side))

    per_candidate = trace.get("per_candidate", [])
    for idx, x in enumerate(trace.get("candidates", [])):
        events.append(DPAddCandidateEvent(type="dp_add_candidate", idx=idx, x=float(x)))
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

    events.append(ReplayStartEvent(type="replay_start"))
    tours = reconstruct_one_side_plan(candidates, plan)
    for idx, (p, q) in enumerate(tours):
        events.extend(generate_tour_events(idx, p, q))

    events.append(DoneEvent(type="done"))
    return events


__all__ = ["build_dpos_events"]

