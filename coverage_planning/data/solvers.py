from __future__ import annotations

from typing import Callable, Dict, List, Literal, Optional, Protocol, Tuple

# Protocols define the signatures we rely on
class MinToursSolver(Protocol):
    def __call__(
        self, segments: List[Tuple[float, float]], h: float, L: float
    ) -> Tuple[int, List[Tuple[float, float]]]:
        ...


class MinLengthOneSideSolver(Protocol):
    def __call__(
        self, segments: List[Tuple[float, float]], h: float, L: float
    ) -> Tuple[List[float], List[float], List[float]]:
        ...


class MinLengthFullLineSolver(Protocol):
    def __call__(self, segments: List[Tuple[float, float]], h: float, L: float) -> float:
        ...


SolverFamily = Literal["heuristic", "reference"]


class SolverProvider:
    def __init__(self, family: SolverFamily = "heuristic"):
        self.family = family
        if family == "heuristic":
            from coverage_planning.algs.heuristics.dp_full_line_heur import dp_full_line
            from coverage_planning.algs.heuristics.dp_one_side_heur import dp_one_side
            from coverage_planning.algs.heuristics.gs_mintours import greedy_min_tours

            self.min_tours: MinToursSolver = greedy_min_tours
            self.dp_one_side: MinLengthOneSideSolver = dp_one_side
            self.dp_full_line: MinLengthFullLineSolver = dp_full_line
        elif family == "reference":
            from coverage_planning.algs.reference import dp_full, dpos, gs

            self.min_tours = gs
            self.dp_one_side = dpos
            self.dp_full_line = dp_full
        else:
            raise ValueError(f"Unknown solver family: {family}")
