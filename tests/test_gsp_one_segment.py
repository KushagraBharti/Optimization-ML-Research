from __future__ import annotations

from coverline.algorithms import solve_min_length_one_segment_gsp, solve_min_tours_gs
from coverline.coverage import validate_solution
from coverline.demo.figure_examples import _find_fig2_counterexample
from coverline.model import Instance
from coverline.oracles.exact_small import exact_min_length_two_side


def test_gsp_improves_over_gs_on_figure2_counterexample() -> None:
    instance, gs, gsp = _find_fig2_counterexample()
    assert gsp.total_length < gs.total_length - 1e-6
    assert validate_solution(instance, gsp).valid


def test_gsp_matches_exact_on_one_segment_cases() -> None:
    cases = [
        Instance.from_iterable(h=4.0, L=20.0, segments=[(1.0, 9.0)]),
        Instance.from_iterable(h=4.0, L=21.5, segments=[(-9.0, -1.0)]),
        Instance.from_iterable(h=4.0, L=22.0, segments=[(-8.0, 7.5)]),
    ]
    for instance in cases:
        gsp = solve_min_length_one_segment_gsp(instance)
        exact = exact_min_length_two_side(instance)
        assert abs(gsp.total_length - exact.total_length) <= 1e-6
        assert validate_solution(instance, gsp).valid


def test_gsp_vs_gs_for_non_crossing_segment() -> None:
    instance = Instance.from_iterable(h=4.0, L=20.0, segments=[(1.0, 7.5)])
    gsp = solve_min_length_one_segment_gsp(instance)
    gs = solve_min_tours_gs(instance)
    assert gsp.total_length <= gs.total_length + 1e-6
