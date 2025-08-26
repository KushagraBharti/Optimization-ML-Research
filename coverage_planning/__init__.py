# Heuristic APIs – default exports
from .algs.heuristics import (
    greedy_min_tours,
    greedy_min_length_one_segment,
    dp_one_side,
    dp_full_line,
)

# Geometry & constants
from .algs.geometry import tour_length, find_maximal_p, VERBOSE, EPS

# Reference solvers – available under .reference.*
from .algs import reference as reference

__all__ = [
    # geometry
    "tour_length", "find_maximal_p", "VERBOSE", "EPS",
    # heuristics (default)
    "greedy_min_tours",
    "greedy_min_length_one_segment",
    "dp_one_side",
    "dp_full_line",
    # namespaced access to reference solvers
    "reference",
]
