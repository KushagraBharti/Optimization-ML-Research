from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np


def bridge_benefit(cost_full: float, cost_no_bridge: float) -> float:
    """Return the fractional benefit of allowing bridge tours.

    Parameters
    ----------
    cost_full:
        Optimal cost with bridge-aware planning.
    cost_no_bridge:
        Cost when restricting to independent left/right solutions.
    """

    return (cost_no_bridge - cost_full) / max(cost_full, 1e-9)


def candidate_size_summary(values: Sequence[int]) -> Mapping[str, float]:
    """Summarise candidate-set sizes with robust percentiles."""

    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def extract_candidate_sizes(samples: Iterable[Mapping[str, object]]) -> Sequence[int]:
    """Collect candidate counts from an iterable of serialised samples."""

    counts = []
    for sample in samples:
        gold = sample.get("gold")
        if not isinstance(gold, Mapping):
            continue
        meta = gold.get("meta")
        if not isinstance(meta, Mapping):
            continue
        dp_meta = meta.get("dp_meta")
        if not isinstance(dp_meta, Mapping):
            continue
        total = 0
        for key in ("C_left", "C_right", "C_tail"):
            value = dp_meta.get(key, 0)
            if isinstance(value, (int, float)):
                total += int(value)
        if total > 0:
            counts.append(total)
    return counts
