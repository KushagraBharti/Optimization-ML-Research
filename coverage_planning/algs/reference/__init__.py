from __future__ import annotations

try:
    from .greedy_ref import greedy_min_tours_ref as gs  # type: ignore[no-redef]
except ImportError:  # pragma: no cover - fallback for direct execution
    from importlib import import_module

    gs = import_module("coverage_planning.algs.reference.greedy_ref").greedy_min_tours_ref  # type: ignore[attr-defined]

try:
    from .gsp_ref import greedy_min_length_one_segment_ref as gsp  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    from importlib import import_module

    gsp = import_module("coverage_planning.algs.reference.gsp_ref").greedy_min_length_one_segment_ref  # type: ignore[attr-defined]

try:
    from .dp_one_side_ref import dp_one_side_ref as dpos  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    from importlib import import_module

    dpos = import_module("coverage_planning.algs.reference.dp_one_side_ref").dp_one_side_ref  # type: ignore[attr-defined]

try:
    from .dp_full_line_ref import dp_full_line_ref as dp_full  # type: ignore[no-redef]
except ImportError:  # pragma: no cover
    from importlib import import_module

    dp_full = import_module("coverage_planning.algs.reference.dp_full_line_ref").dp_full_line_ref  # type: ignore[attr-defined]

__all__ = ["gs", "gsp", "dpos", "dp_full"]
