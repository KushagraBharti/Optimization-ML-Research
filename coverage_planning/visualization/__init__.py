"""Visualization subsystem package."""

from . import adapters
from .events import compute_scene_bounds

__version__ = "0.1"

__all__ = ["__version__", "adapters", "compute_scene_bounds"]

