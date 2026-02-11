from __future__ import annotations

import random

from coverline.algorithms import solve_min_length_two_side
from coverline.coverage import validate_solution
from coverline.model import Instance
from coverline.oracles.exact_small import exact_min_length_two_side


def _random_two_side(seed: int, n: int) -> Instance:
    rng = random.Random(seed)
    left = []
    right = []
    x = -1.2
    for _ in range(max(1, n // 2)):
        seg_len = rng.uniform(0.18, 0.35)
        gap = rng.uniform(0.05, 0.18)
        b = x
        a = b - seg_len
        left.append((a, b))
        x = a - gap
    left = list(reversed(left))
    x = 1.2
    for _ in range(n - len(left)):
        seg_len = rng.uniform(0.18, 0.35)
        gap = rng.uniform(0.05, 0.18)
        a = x
        b = a + seg_len
        right.append((a, b))
        x = b + gap
    return Instance.from_iterable(h=4.0, L=20.0, segments=left + right)


def _random_with_crossing_projection(seed: int) -> Instance:
    rng = random.Random(seed)
    left = []
    x = -6.0
    for _ in range(2):
        seg_len = rng.uniform(0.35, 0.8)
        gap = rng.uniform(0.25, 0.6)
        b = x
        a = b - seg_len
        left.append((a, b))
        x = a - gap
    left = list(reversed(left))

    center_left = -rng.uniform(0.8, 1.8)
    center_right = rng.uniform(0.8, 1.8)
    center = [(center_left, center_right)]

    right = []
    x = 5.5
    for _ in range(2):
        seg_len = rng.uniform(0.35, 0.8)
        gap = rng.uniform(0.25, 0.6)
        a = x
        b = a + seg_len
        right.append((a, b))
        x = b + gap
    return Instance.from_iterable(h=4.0, L=24.0, segments=left + center + right)


def test_two_side_matches_exact_on_small_random_batch() -> None:
    for n in range(2, 8):
        instance = _random_two_side(seed=700 + n, n=n)
        fast = solve_min_length_two_side(instance)
        exact = exact_min_length_two_side(instance)
        assert abs(fast.total_length - exact.total_length) <= 1e-6
        assert validate_solution(instance, fast).valid


def test_two_side_matches_exact_with_crossing_projection_batch() -> None:
    for seed in range(840, 846):
        instance = _random_with_crossing_projection(seed=seed)
        fast = solve_min_length_two_side(instance)
        exact = exact_min_length_two_side(instance)
        assert abs(fast.total_length - exact.total_length) <= 1e-6
        assert validate_solution(instance, fast).valid


def test_two_side_reduces_to_one_side() -> None:
    instance = Instance.from_iterable(h=4.0, L=16.0, segments=[(0.5, 1.3), (2.0, 2.8), (3.5, 4.2)])
    fast = solve_min_length_two_side(instance)
    exact = exact_min_length_two_side(instance)
    assert abs(fast.total_length - exact.total_length) <= 1e-6


def test_two_side_with_segment_crossing_projection_matches_exact() -> None:
    instance = Instance.from_iterable(
        h=4.0,
        L=24.0,
        segments=[(-8.0, -7.0), (-2.5, 2.0), (6.2, 7.1)],
    )
    fast = solve_min_length_two_side(instance)
    exact = exact_min_length_two_side(instance)
    assert abs(fast.total_length - exact.total_length) <= 1e-6
    assert validate_solution(instance, fast).valid


def test_two_side_interior_projection_case_matches_exact() -> None:
    instance = Instance.from_iterable(
        h=4.0,
        L=18.0,
        segments=[(-8.0, -1.0), (1.0, 7.0)],
    )
    fast = solve_min_length_two_side(instance)
    exact = exact_min_length_two_side(instance)
    assert abs(fast.total_length - exact.total_length) <= 1e-6
    assert validate_solution(instance, fast).valid
    assert fast.metadata.get("case") == "interior_projection"
    assert exact.metadata.get("case") == "interior"
