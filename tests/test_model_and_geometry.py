from __future__ import annotations

import math

import pytest

from coverline.geometry import (
    feasible_tour,
    solve_maximal_left_endpoint,
    solve_maximal_right_endpoint,
    tour_length,
)
from coverline.model import EPS, Instance, Segment


def test_segment_invariant() -> None:
    with pytest.raises(ValueError):
        Segment(2.0, 1.0)


def test_instance_sort_and_disjoint_validation() -> None:
    inst = Instance.from_iterable(h=4.0, L=20.0, segments=[(5.0, 6.0), (1.0, 2.0)])
    assert inst.segments[0].a == 1.0
    with pytest.raises(ValueError):
        Instance.from_iterable(h=4.0, L=20.0, segments=[(1.0, 2.0), (2.0, 3.0)])


def test_tour_length_formula() -> None:
    left, right, h = -3.0, 5.0, 4.0
    length = tour_length(left, right, h)
    expected = math.hypot(left, h) + math.hypot(right, h) + (right - left)
    assert abs(length - expected) <= 1e-12


def test_maximal_endpoint_left_solver_hits_length_bound() -> None:
    h = 4.0
    L = 18.0
    right = 8.0
    left = solve_maximal_left_endpoint(right=right, h=h, L=L)
    assert tour_length(left, right, h) <= L + 1e-7
    assert abs(tour_length(left, right, h) - L) <= 1e-6
    assert not feasible_tour(left - 1e-3, right, h, L)


def test_maximal_endpoint_right_solver_hits_length_bound() -> None:
    h = 4.0
    L = 18.0
    left = -7.0
    right = solve_maximal_right_endpoint(left=left, h=h, L=L)
    assert tour_length(left, right, h) <= L + 1e-7
    assert abs(tour_length(left, right, h) - L) <= 1e-6
    assert not feasible_tour(left, right + 1e-3, h, L)
