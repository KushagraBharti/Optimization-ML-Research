"""Shared event schema for coverage-planning visualizations."""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypedDict

Segment = Tuple[float, float]


class SetSceneEvent(TypedDict):
    type: Literal["set_scene"]
    h: float
    L: float
    x_min: float
    x_max: float


class AddSegmentEvent(TypedDict):
    type: Literal["add_segment"]
    id: str
    x1: float
    x2: float
    side: Literal["left", "right", "straddle"]


class DPStartEvent(TypedDict, total=False):
    type: Literal["dp_start"]
    mode: Literal["one_side", "full"]
    side: Optional[Literal["left", "right"]]
    stage: str


class DPAddCandidateEvent(TypedDict):
    type: Literal["dp_add_candidate"]
    idx: int
    x: float


DPTryTransitionEvent = TypedDict(
    "DPTryTransitionEvent",
    {
        "type": Literal["dp_try_transition"],
        "from": Optional[int],
        "to": Optional[int],
        "case": str,
        "length": float,
        "feasible": bool,
    },
    total=False,
)


DPPickTransitionEvent = TypedDict(
    "DPPickTransitionEvent",
    {
        "type": Literal["dp_pick_transition"],
        "from": Optional[int],
        "to": Optional[int],
        "cost": float,
        "case": str,
    },
    total=False,
)


class DPDoneEvent(TypedDict):
    type: Literal["dp_done"]


class BridgeTryEvent(TypedDict, total=False):
    type: Literal["bridge_try"]
    P: float
    q: float
    accepted: bool
    selected: bool


class ReplayStartEvent(TypedDict):
    type: Literal["replay_start"]


class StartTourEvent(TypedDict):
    type: Literal["start_tour"]
    tour_id: int
    p: float
    q: float


class AnimateTourEvent(TypedDict):
    type: Literal["animate_tour"]
    tour_id: int
    phase: Literal["up", "across", "down"]
    progress: float


class MarkCoveredEvent(TypedDict):
    type: Literal["mark_covered"]
    x1: float
    x2: float


class EndTourEvent(TypedDict):
    type: Literal["end_tour"]
    tour_id: int


class AlgoInfoEvent(TypedDict):
    type: Literal["algo_info"]
    name: str
    case: Optional[str]


class ErrorEvent(TypedDict):
    type: Literal["error"]
    text: str


class DoneEvent(TypedDict):
    type: Literal["done"]


EventDict = Dict[str, object]


def classify_segment(x1: float, x2: float) -> Literal["left", "right", "straddle"]:
    """Return the relative position of a segment wrt the origin."""
    if x1 >= 0.0 and x2 >= 0.0:
        return "right"
    if x1 <= 0.0 and x2 <= 0.0:
        return "left"
    return "straddle"


def compute_scene_bounds(
    segments: Iterable[Segment],
    margin: float = 0.1,
) -> Tuple[float, float]:
    """Compute x-axis bounds for the scene with a fractional margin."""
    seg_list = list(segments)
    if not seg_list:
        span = 1.0
        pad = max(0.5, span * margin)
        return -pad, pad

    xs = [coord for seg in seg_list for coord in seg]
    min_x = min(xs)
    max_x = max(xs)
    span = max(max_x - min_x, 1e-6)
    pad = max(span * margin, 0.5)
    return min_x - pad, max_x + pad


__all__ = [
    "EventDict",
    "SetSceneEvent",
    "AddSegmentEvent",
    "DPStartEvent",
    "DPAddCandidateEvent",
    "DPTryTransitionEvent",
    "DPPickTransitionEvent",
    "DPDoneEvent",
    "BridgeTryEvent",
    "ReplayStartEvent",
    "StartTourEvent",
    "AnimateTourEvent",
    "MarkCoveredEvent",
    "EndTourEvent",
    "AlgoInfoEvent",
    "ErrorEvent",
    "DoneEvent",
    "classify_segment",
    "compute_scene_bounds",
]
