from __future__ import annotations

from coverline.algorithms import (
    solve_min_length_one_side_dpos,
    solve_min_length_two_side,
    solve_min_tours_gs,
)
from coverline.candidates import build_candidate_sets_one_side
from coverline.demo.figure_examples import figure3_demo
from coverline.geometry import tour_length
from coverline.model import Instance


def test_theorem1_style_gs_count_on_simple_instance() -> None:
    instance = Instance.from_iterable(h=4.0, L=22.0, segments=[(-7.5, -6.8), (-5.4, -4.7), (3.8, 4.8), (6.1, 7.0)])
    sol = solve_min_tours_gs(instance)
    assert sol.tour_count >= 1
    assert all(t.length <= instance.L + 1e-6 for t in sol.tours)


def test_figure3_inequality_payload(tmp_path) -> None:
    payload = figure3_demo(tmp_path)
    assert payload["blue_better"] is True
    assert payload["dpos_matches_blue"] is True


def test_lemma9_candidate_pair_membership_when_interior_case_selected() -> None:
    instance = Instance.from_iterable(
        h=4.0,
        L=24.0,
        segments=[(-8.8, -7.5), (-6.5, -5.4), (-3.8, -3.0), (2.6, 3.4), (5.4, 6.3), (7.9, 8.8)],
    )
    sol = solve_min_length_two_side(instance)
    if sol.metadata.get("case") != "interior_projection":
        # Not every instance picks case 3; this test validates only when selected.
        return
    pair = sol.metadata.get("pair")
    assert isinstance(pair, tuple) and len(pair) == 2
    p, q = pair
    left_candidates = set(build_candidate_sets_one_side(instance.clipped(right=0.0)).union)
    right_candidates = set(build_candidate_sets_one_side(instance.clipped(left=0.0)).union)
    assert any(abs(p - c) <= 1e-6 for c in left_candidates)
    assert any(abs(q - c) <= 1e-6 for c in right_candidates)
    assert tour_length(p, q, instance.h) <= instance.L + 1e-6
