from __future__ import annotations

from coverline.algorithms.min_length_two_side import _build_suffix_artifacts_positive
from coverline.candidates import build_candidate_sets_one_side
from coverline.model import Instance
from coverline.oracles.exact_small import exact_min_length_one_side


def test_topdown_suffix_dp_matches_exact_on_right_side_candidates() -> None:
    instance = Instance.from_iterable(
        h=4.0,
        L=19.0,
        segments=[(0.8, 1.7), (2.4, 3.2), (4.1, 5.0), (6.0, 7.1)],
    )
    query_points = list(build_candidate_sets_one_side(instance).union)
    artifacts = _build_suffix_artifacts_positive(instance, query_points=query_points)

    for q in query_points:
        got = artifacts.cost[artifacts.candidate_resolver.resolve(q)]
        exact = exact_min_length_one_side(instance.clipped(left=q)).total_length
        assert abs(got - exact) <= 1e-6


def test_topdown_suffix_dp_matches_exact_on_mirrored_left_queries() -> None:
    left_instance = Instance.from_iterable(
        h=4.0,
        L=20.0,
        segments=[(-8.8, -7.9), (-6.1, -5.0), (-3.7, -2.8), (-1.9, -1.1)],
    )
    c_left = [x for x in build_candidate_sets_one_side(left_instance).union if x < -1e-9]
    mirrored = left_instance.mirrored_x()
    query_points = [-p for p in c_left]
    artifacts = _build_suffix_artifacts_positive(mirrored, query_points=query_points)

    for p in c_left:
        got = artifacts.cost[artifacts.candidate_resolver.resolve(-p)]
        exact = exact_min_length_one_side(left_instance.clipped(right=p)).total_length
        assert abs(got - exact) <= 1e-6
