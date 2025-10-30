from __future__ import annotations

from typing import Sequence

import numpy as np


def bridge_benefit(cost_full: float, cost_no_bridge: float) -> float:
    if cost_full <= 0.0:
        return 0.0
    return (cost_no_bridge - cost_full) / max(cost_full, 1e-9)


def summarize_candidates(candidates: Sequence[int]) -> dict[str, float]:
    if not candidates:
        return {"count": 0}
    arr = np.array(candidates, dtype=float)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }
