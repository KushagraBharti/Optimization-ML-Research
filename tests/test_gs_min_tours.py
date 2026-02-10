from __future__ import annotations

import random

from coverline.algorithms import solve_min_tours_gs, solve_min_tours_gs_linear
from coverline.coverage import validate_solution
from coverline.model import Instance


def _random_instance(seed: int, n: int) -> Instance:
    rng = random.Random(seed)
    segments = []
    x = -1.0 - (0.4 * n)
    for _ in range(n):
        gap = rng.uniform(0.05, 0.2)
        seg_len = rng.uniform(0.12, 0.3)
        a = x + gap
        b = a + seg_len
        segments.append((a, b))
        x = b
    return Instance.from_iterable(h=4.0, L=20.0, segments=segments)


def test_gs_variants_equivalent_on_fixed_instance() -> None:
    instance = Instance.from_iterable(
        h=4.0,
        L=22.0,
        segments=[(-7.8, -6.8), (-5.6, -4.9), (-1.5, -0.6), (1.6, 2.8), (4.5, 5.9)],
    )
    s1 = solve_min_tours_gs(instance)
    s2 = solve_min_tours_gs_linear(instance)
    assert s1.tour_count == s2.tour_count
    assert abs(s1.total_length - s2.total_length) <= 1e-6
    assert validate_solution(instance, s1).valid
    assert validate_solution(instance, s2).valid


def test_gs_variants_equivalent_on_random_batch() -> None:
    for n in range(2, 9):
        instance = _random_instance(seed=1200 + n, n=n)
        s1 = solve_min_tours_gs(instance)
        s2 = solve_min_tours_gs_linear(instance)
        assert s1.tour_count == s2.tour_count
        assert validate_solution(instance, s1).valid
        assert validate_solution(instance, s2).valid
