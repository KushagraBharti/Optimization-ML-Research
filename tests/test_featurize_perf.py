from __future__ import annotations

import time

import pytest

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference.dp_one_side_ref import dp_one_side_with_plan, reconstruct_one_side_plan
from coverage_planning.learn.featurize import featurize_sample


def _make_large_sample(num_segments: int = 360) -> dict:
    segments = []
    cursor = 0.0
    for _ in range(num_segments):
        segments.append((cursor, cursor + 1.0))
        cursor += 1.5
    h = 5.0
    L = tour_length(0.0, segments[-1][1], h) * 1.2
    _, candidates, plan = dp_one_side_with_plan(list(segments), h, L)
    tours = reconstruct_one_side_plan(candidates, plan)
    return {
        "segments": segments,
        "h": h,
        "L": L,
        "tours": tours,
        "objective": "min_length",
    }


def test_featurize_performance_envelope() -> None:
    sample = _make_large_sample()
    start = time.monotonic()
    result = featurize_sample(sample, sparse_threshold=256)
    duration = time.monotonic() - start
    assert duration < 1.0
    assert any(step["mask"]["format"] == "sparse" for step in result["steps"])
