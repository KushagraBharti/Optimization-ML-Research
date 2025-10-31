"""Deprecated shim; prefer :mod:`coverage_planning.eval.metrics`."""

from __future__ import annotations

import warnings

from coverage_planning.eval.metrics import (
    bridge_benefit,
    candidate_size_summary,
    extract_candidate_sizes,
    summarize_candidates,
)

__all__ = [
    "bridge_benefit",
    "candidate_size_summary",
    "summarize_candidates",
    "extract_candidate_sizes",
]

warnings.warn(
    "The top-level 'eval.metrics' module is deprecated; import from "
    "'coverage_planning.eval.metrics' instead.",
    DeprecationWarning,
    stacklevel=2,
)
