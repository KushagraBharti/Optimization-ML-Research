"""Coverline package."""

from .coverage import validate_solution
from .model import EPS, Instance, Segment, Solution, Tour, ValidationReport

__all__ = [
    "EPS",
    "Instance",
    "Segment",
    "Solution",
    "Tour",
    "ValidationReport",
    "validate_solution",
]
