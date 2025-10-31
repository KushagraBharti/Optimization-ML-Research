from __future__ import annotations

from coverage_planning.learn.mask_predicates import (
    contiguous_cover,
    covers_contiguously,
    normalize_intervals,
    subtract_interval,
)


def test_mask_predicates_basic_operations() -> None:
    intervals = normalize_intervals([(1.5, 2.0), (0.0, 1.0), (1.0, 1.4)])
    assert intervals == [(0.0, 1.0), (1.0, 2.0)]

    remainder = subtract_interval(intervals, (0.25, 0.75))
    assert remainder == [(0.0, 0.25), (0.75, 1.0), (1.0, 2.0)]

    assert covers_contiguously(intervals, 0.0, 2.0)
    coverage = contiguous_cover(intervals, 0.0, 2.0)
    assert coverage is not None
    assert coverage.start_index == 0
    assert coverage.end_index == 1
