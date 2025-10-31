"""Evaluation metrics and summary utilities for coverage planning outputs."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "bridge_benefit",
    "candidate_size_summary",
    "summarize_candidates",
    "extract_candidate_sizes",
]


def bridge_benefit(cost_full: float, cost_no_bridge: float) -> float:
    if cost_full <= 0.0:
        return 0.0
    return (cost_no_bridge - cost_full) / max(cost_full, 1e-9)


def candidate_size_summary(candidates: Sequence[int]) -> dict[str, float]:
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


summarize_candidates = candidate_size_summary


def extract_candidate_sizes(samples: Iterable[Mapping[str, object]]) -> Sequence[int]:
    """Collect candidate counts from serialised samples."""

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
