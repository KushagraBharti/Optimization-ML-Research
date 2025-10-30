from __future__ import annotations

import math

import pytest

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM
from coverage_planning.learn.transition_core import (
    classify_x,
    covers_no_gap,
    is_maximal_pair,
    normalize_intervals,
    trim_covered,
)


def _assert_intervals_equal(actual, expected, tol=EPS_GEOM):
    assert len(actual) == len(expected)
    for (a0, b0), (a1, b1) in zip(actual, expected):
        assert math.isclose(a0, a1, abs_tol=tol)
        assert math.isclose(b0, b1, abs_tol=tol)


def test_classify_x_basic() -> None:
    intervals = [(-5.0, -3.0), (0.0, 2.0), (5.0, 9.0)]

    assert classify_x(-10.0, intervals) == ("before", None)
    assert classify_x(1.5, intervals) == ("inseg", 1)
    assert classify_x(3.0, intervals) == ("gap", 1)
    assert classify_x(20.0, intervals) == ("beyond", None)


def test_covers_no_gap_true_false() -> None:
    single = [(0.0, 10.0)]
    assert covers_no_gap(2.0, 8.0, single)

    disjoint = [(0.0, 2.0), (5.0, 10.0)]
    assert not covers_no_gap(1.0, 6.0, disjoint)

    touching = [(0.0, 2.0), (2.0 + 5.0 * EPS_GEOM, 5.0)]
    assert not covers_no_gap(0.5, 4.0, touching)


def test_is_maximal_pair_matches_geometry() -> None:
    h = 4.0
    q = 9.0
    p_expected = 3.0
    L = tour_length(p_expected, q, h)
    # Sanity: geometry helper returns same reference.
    p_star = find_maximal_p(q, h, L)
    assert math.isclose(p_star, p_expected, rel_tol=0.0, abs_tol=1e-6)

    assert is_maximal_pair(p_expected, q, h, L)
    assert not is_maximal_pair(p_expected - 1e-4, q, h, L)


def test_trim_covered_various() -> None:
    base = [(-5.0, -1.0), (-0.5, 0.5), (1.0, 4.0), (6.0, 9.0)]

    removed_middle = trim_covered(base, -0.5, 0.5)
    _assert_intervals_equal(removed_middle, [(-5.0, -1.0), (1.0, 4.0), (6.0, 9.0)])

    trimmed_prefix = trim_covered(base, -5.0, -3.0)
    _assert_intervals_equal(
        trimmed_prefix,
        [(-3.0, -1.0), (-0.5, 0.5), (1.0, 4.0), (6.0, 9.0)],
    )

    trimmed_suffix = trim_covered(base, 2.0, 7.0)
    _assert_intervals_equal(
        trimmed_suffix,
        [(-5.0, -1.0), (-0.5, 0.5), (1.0, 2.0), (7.0, 9.0)],
    )

    cross_zero = [(-4.0, -1.0), (1.0, 4.0)]
    trimmed_cross = trim_covered(cross_zero, -2.0, 2.0)
    _assert_intervals_equal(trimmed_cross, [(-4.0, -2.0), (2.0, 4.0)])


def test_normalize_intervals_merges_and_sorts() -> None:
    intervals = [(3.0, 5.0), (0.0, 1.0), (0.9, 2.1)]
    normalized = normalize_intervals(intervals)
    _assert_intervals_equal(normalized, [(0.0, 2.1), (3.0, 5.0)])


def test_classify_x_knife_edges() -> None:
    single = [(0.0, 10.0)]
    for delta in (0.0, EPS_GEOM / 2.0, -EPS_GEOM / 2.0):
        assert classify_x(0.0 + delta, single)[0] == "inseg"
        assert classify_x(10.0 + delta, single)[0] == "inseg"

    dual = [(0.0, 5.0), (10.0, 15.0)]
    for delta in (0.0, EPS_GEOM / 2.0, -EPS_GEOM / 2.0):
        assert classify_x(0.0 + delta, dual)[0] == "inseg"
        assert classify_x(5.0 + delta, dual)[0] == "inseg"
        assert classify_x(10.0 + delta, dual)[0] == "inseg"
        assert classify_x(15.0 + delta, dual)[0] == "inseg"


def test_trim_covered_idempotent() -> None:
    start = [(0.0, 10.0), (12.0, 15.0)]
    after_first = trim_covered(start, 2.0, 8.0)
    after_second = trim_covered(after_first, 2.0, 8.0)
    _assert_intervals_equal(after_first, after_second)

    cross = [(-5.0, -1.0), (1.0, 5.0)]
    trimmed_once = trim_covered(cross, -2.0, 2.0)
    trimmed_twice = trim_covered(trimmed_once, -2.0, 2.0)
    _assert_intervals_equal(trimmed_once, trimmed_twice)
