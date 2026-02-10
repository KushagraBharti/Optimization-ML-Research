from __future__ import annotations

from coverline.algorithms import solve_min_length_one_side_dpos, solve_min_length_one_side_dpos_with_artifacts
from coverline.candidates import build_candidate_sets_one_side
from coverline.coverage import validate_solution
from coverline.geometry import tour_length
from coverline.model import EPS, Instance
from coverline.oracles.exact_small import exact_min_length_one_side, exhaustive_grid_instances


def test_dpos_matches_exact_on_exhaustive_small_grid() -> None:
    instances = exhaustive_grid_instances(
        h=4.0,
        L=16.5,
        x_values=[0.6, 1.2, 1.8, 2.6, 3.2, 4.0],
        max_segments=3,
    )
    # Exhaustive set can be large; this deterministic cap still provides broad coverage.
    for instance in instances[:120]:
        fast = solve_min_length_one_side_dpos(instance)
        exact = exact_min_length_one_side(instance)
        assert abs(fast.total_length - exact.total_length) <= 1e-6
        assert validate_solution(instance, fast).valid


def test_dpos_candidate_structure_on_solution() -> None:
    instance = Instance.from_iterable(
        h=4.0,
        L=19.0,
        segments=[(0.8, 1.7), (2.5, 3.3), (4.2, 5.0), (6.1, 7.3)],
    )
    sol, _ = solve_min_length_one_side_dpos_with_artifacts(instance)
    candidates = set(build_candidate_sets_one_side(instance).union)
    seg_left_points = {s.a for s in instance.segments}

    for t in sol.tours:
        assert any(abs(t.right - c) <= 1e-6 for c in candidates)
        left_is_segment_left = any(abs(t.left - a) <= 1e-6 for a in seg_left_points)
        left_is_maximal = abs(tour_length(t.left, t.right, instance.h) - instance.L) <= 1e-6
        assert left_is_segment_left or left_is_maximal
