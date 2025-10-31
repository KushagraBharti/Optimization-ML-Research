"""Deprecated compatibility shim for legacy solver provider wiring."""

from __future__ import annotations

import warnings

from coverage_planning.data._legacy_solvers_provider import (
    MinLengthFullLineSolver,
    MinLengthOneSideSolver,
    MinToursSolver,
    SolverFamily,
    SolverProvider,
)

warnings.warn(
    "coverage_planning.data.solvers is deprecated; import solvers directly from "
    "coverage_planning.algs.reference or use coverage_planning.data._legacy_solvers_provider.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "SolverProvider",
    "SolverFamily",
    "MinToursSolver",
    "MinLengthOneSideSolver",
    "MinLengthFullLineSolver",
]
