from __future__ import annotations

import os
import random
import sys
from typing import Dict

try:  # Prefer numpy when available, but keep optional.
    import numpy as _np  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _np = None

# Geometry tolerance: retain source of truth from the geometry module.
from coverage_planning.algs.geometry import EPS as _GEOM_EPS

EPS_GEOM: float = _GEOM_EPS
TOL_NUM: float = 1e-6
DEFAULT_SEED: int = 1337

RNG_SEEDS: Dict[str, int] = {
    "tests": DEFAULT_SEED,
    "bench": 4242,
    "data": 5150,
}


def seed_everywhere(seed: int) -> None:
    """Seed all supported RNG backends deterministically."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _np is not None:
        _np.random.seed(seed)  # pragma: no branch

    try:  # Torch is optional; ignore if unavailable.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - exercised only with CUDA
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch optional
        pass


__all__ = [
    "EPS_GEOM",
    "TOL_NUM",
    "DEFAULT_SEED",
    "RNG_SEEDS",
    "seed_everywhere",
]

