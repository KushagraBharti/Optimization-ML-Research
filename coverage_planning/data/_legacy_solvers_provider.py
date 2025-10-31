"""Deprecated solver provider wiring used in early data pipelines.

Prefer importing solvers directly from ``coverage_planning.algs.reference``.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Literal, Optional, Protocol, Tuple

SolverFamily = Literal["heuristic", "reference"]


class MinToursSolver(Protocol):
    def __call__(
        self,
        segments: List[Tuple[float, float]],
        h: float,
        L: float,
    ) -> Tuple[int, List[Tuple[float, float]]]:
        ...


class MinLengthOneSideSolver(Protocol):
    def __call__(
        self,
        segments: List[Tuple[float, float]],
        h: float,
        L: float,
    ) -> Tuple[List[float], List[float], List[float]]:
        ...


class MinLengthFullLineSolver(Protocol):
    def __call__(
        self,
        segments: List[Tuple[float, float]],
        h: float,
        L: float,
    ) -> float:
        ...


class SolverProvider:
    """Legacy helper that binds solver families for downstream pipelines."""

    def __init__(self, family: SolverFamily = "heuristic"):
        warnings.warn(
            "SolverProvider is deprecated; import solvers from "
            "coverage_planning.algs.reference instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.family = family
        if family == "heuristic":
            from coverage_planning.algs.heuristics import (
                dp_full_line as legacy_dp_full,
                dp_one_side as legacy_dpos,
                greedy_min_tours as legacy_gs,
            )

            self.min_tours: MinToursSolver = legacy_gs
            self.dp_one_side: MinLengthOneSideSolver = legacy_dpos
            self.dp_full_line: MinLengthFullLineSolver = legacy_dp_full
        elif family == "reference":
            from coverage_planning.algs.reference import dp_full, dpos, gs

            self.min_tours = gs
            self.dp_one_side = dpos
            self.dp_full_line = dp_full
        else:
            raise ValueError(f"Unknown solver family: {family}")


__all__ = [
    "SolverProvider",
    "SolverFamily",
    "MinToursSolver",
    "MinLengthOneSideSolver",
    "MinLengthFullLineSolver",
]
