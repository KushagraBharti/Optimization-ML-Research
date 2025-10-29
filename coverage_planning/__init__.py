# Heuristic APIs – default exports
from .algs.heuristics import (
    greedy_min_tours,
    greedy_min_length_one_segment,
    dp_one_side,
    dp_full_line,
)

# Geometry & constants
from .algs.geometry import tour_length, find_maximal_p, VERBOSE
from .common.constants import (
    DEFAULT_SEED,
    EPS_GEOM,
    RNG_SEEDS,
    TOL_NUM,
    seed_everywhere,
)

# Backwards compatibility: preserve EPS name at package root.
EPS = EPS_GEOM

# Reference solvers – available under .reference.*
from .algs import reference as reference

__all__ = [
    # geometry
    "tour_length",
    "find_maximal_p",
    "VERBOSE",
    "EPS",
    "EPS_GEOM",
    "TOL_NUM",
    "DEFAULT_SEED",
    "RNG_SEEDS",
    "seed_everywhere",
    # heuristics (default)
    "greedy_min_tours",
    "greedy_min_length_one_segment",
    "dp_one_side",
    "dp_full_line",
    # namespaced access to reference solvers
    "reference",
]
