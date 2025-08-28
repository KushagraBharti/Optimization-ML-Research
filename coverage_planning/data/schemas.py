from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Dict, Any

Objective = Literal["MinTours", "MinLength_OneSide", "MinLength_FullLine"]
Provenance = Literal["heuristic", "reference"]

@dataclass
class Instance:
    segments: List[Tuple[float, float]]
    h: float
    L: float
    seed: int | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LabelMinTours:
    tours: List[Tuple[float, float]]
    count: int
    total_cost: float  # = count for MinTours (or 0 if you want)

@dataclass
class LabelMinLengthOneSide:
    candidates: List[float]
    prefix_costs: List[float]
    suffix_costs: List[float]
    optimal_total: float

@dataclass
class LabelMinLengthFullLine:
    optimal_total: float

@dataclass
class LabeledExample:
    instance: Instance
    objective: Objective
    provenance: Provenance
    label: Any   # one of the Label* above
    notes: Dict[str, Any] = field(default_factory=dict)  # e.g., solver timings, flags
