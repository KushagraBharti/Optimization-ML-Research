from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Literal, Optional, Protocol

# Protocols define the signatures we rely on
class MinToursSolver(Protocol):
    def __call__(self, segments: List[Tuple[float, float]], h: float, L: float
                 ) -> Tuple[int, List[Tuple[float, float]]]: ...

class MinLengthOneSideSolver(Protocol):
    def __call__(self, segments: List[Tuple[float, float]], h: float, L: float
                 ) -> Tuple[List[float], List[float], List[float]]: ...

class MinLengthFullLineSolver(Protocol):
    def __call__(self, segments: List[Tuple[float, float]], h: float, L: float
                 ) -> float: ...

SolverFamily = Literal["heuristic", "reference"]

class SolverProvider:
    def __init__(self, family: SolverFamily = "heuristic"):
        self.family = family
        if family == "heuristic":
            from coverage_planning.algs.heuristics.gs_mintours import greedy_min_tours
            from coverage_planning.algs.heuristics.dp_one_side_heur import dp_one_side
            from coverage_planning.algs.heuristics.dp_full_line_heur import dp_full_line
            self.min_tours: MinToursSolver = greedy_min_tours
            self.dp_one_side: MinLengthOneSideSolver = dp_one_side
            self.dp_full_line: MinLengthFullLineSolver = dp_full_line
        elif family == "reference":
            # these exist after Milestone 1
            from coverage_planning.algs.reference.gs_mintours_ref import greedy_min_tours_ref
            from coverage_planning.algs.reference.dp_one_side_ref import dp_one_side_ref
            from coverage_planning.algs.reference.dp_full_line_ref import dp_full_line_ref
            self.min_tours = greedy_min_tours_ref
            self.dp_one_side = dp_one_side_ref
            self.dp_full_line = dp_full_line_ref
        else:
            raise ValueError(f"Unknown solver family: {family}")
