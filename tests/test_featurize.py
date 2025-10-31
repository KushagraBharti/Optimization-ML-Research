from __future__ import annotations

import math

import pytest

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import (
    dp_full_with_plan,
    dp_one_side_with_plan,
    reconstruct_one_side_plan,
)
from coverage_planning.learn.transition_core import trim_covered
from coverage_planning.learn.featurize import featurize_sample


def _make_one_side_sample() -> dict:
    segments = [(0.0, 3.0), (5.0, 7.0)]
    h = 2.0
    L = 20.0
    _, candidates, plan = dp_one_side_with_plan(list(segments), h, L)
    tours = reconstruct_one_side_plan(candidates, plan)
    return {
        "segments": segments,
        "h": h,
        "L": L,
        "tours": tours,
        "objective": "min_length",
    }


def _make_bridge_sample() -> dict:
    segments = [(-4.0, -1.0), (1.0, 4.0)]
    h = 2.0
    L = tour_length(-4.0, 4.0, h)
    return {
        "segments": segments,
        "h": h,
        "L": L,
        "tours": [(-4.0, 4.0)],
        "objective": "min_length",
    }


def _make_multi_step_sample() -> dict:
    segments = [(0.0, 3.0), (4.0, 7.0)]
    h = 2.0
    L = 15.0
    _, candidates, plan = dp_one_side_with_plan(list(segments), h, L)
    tours = reconstruct_one_side_plan(candidates, plan)
    return {
        "segments": segments,
        "h": h,
        "L": L,
        "tours": tours,
        "objective": "min_length",
    }


def test_featurize_one_side_small() -> None:
    sample = _make_one_side_sample()
    result = featurize_sample(sample)

    tours = sample["tours"]
    assert len(result["steps"]) == len(tours)
    assert len(result["graph"]["seg_nodes"]) == len(sample["segments"])
    assert len(result["graph"]["cand_nodes"]) >= len(tours)

    for step, (p, q) in zip(result["steps"], tours):
        mask = step["mask"]
        q_idx = step["y_right"]
        p_idx = step["y_left"]
        if mask["format"] == "dense":
            assert mask["mask_right"][q_idx] == 1
            assert p_idx in mask["mask_left_given_right"][q_idx]
        else:
            assert q_idx in mask["legal_right"]
            assert (q_idx, p_idx) in mask["legal_pairs"]


def test_featurize_full_line_small() -> None:
    segments = [(-4.0, -1.0), (1.0, 4.0)]
    h = 2.5
    _, tours, _ = dp_full_with_plan(list(segments), h, 18.0)
    assert tours, "solver returned empty tour set"
    sample = {
        "segments": segments,
        "h": h,
        "L": 18.0,
        "tours": tours,
        "objective": "min_length",
    }
    result = featurize_sample(sample)
    assert len(result["steps"]) == len(tours)
    bridge_seen = any(step["case"] == "bridge" for step in result["steps"])
    if not bridge_seen:
        assert any(
            (step["mask"]["format"] == "dense" and any(step["mask"]["mask_right"]))
            or step["mask"]["format"] == "sparse"
            for step in result["steps"]
        )


def test_featurize_sparse_masks() -> None:
    sample = _make_one_side_sample()
    result = featurize_sample(sample, sparse_threshold=2)
    assert any(step["mask"]["format"] == "sparse" for step in result["steps"])
    for step in result["steps"]:
        if step["mask"]["format"] == "sparse":
            assert step["mask"]["legal_right"]
            assert step["mask"]["legal_pairs"]


def test_featurize_bridge_trimming_final_state() -> None:
    sample = _make_bridge_sample()
    result = featurize_sample(sample)

    assert len(result["steps"]) == 1
    step = result["steps"][0]
    assert step["case"] == "bridge"
    if step["mask"]["format"] == "dense":
        assert any(step["mask"]["mask_right"])
    else:
        assert step["mask"]["legal_pairs"]

    left_state = [seg for seg in sample["segments"] if seg[1] <= 0.0]
    right_state = [seg for seg in sample["segments"] if seg[0] >= 0.0]
    p, q = sample["tours"][0]
    assert not trim_covered(left_state, p, 0.0)
    assert not trim_covered(right_state, 0.0, q)


def test_featurize_value_parity() -> None:
    sample = _make_one_side_sample()
    tours = sample["tours"]
    h = sample["h"]
    remaining = sum(tour_length(p, q, h) for p, q in tours)
    stored = []
    for p, q in tours:
        stored.append(remaining)
        remaining -= tour_length(p, q, h)
    sample["stored_values"] = stored

    result = featurize_sample(sample)
    extracted = [step["value"] for step in result["steps"]]
    assert extracted == pytest.approx(stored)


def test_featurize_value_mismatch_raises() -> None:
    sample = _make_one_side_sample()
    tours = sample["tours"]
    h = sample["h"]
    remaining = sum(tour_length(p, q, h) for p, q in tours)
    stored = []
    for p, q in tours:
        stored.append(remaining)
        remaining -= tour_length(p, q, h)
    stored[0] += 1.0
    sample["stored_values"] = stored

    with pytest.raises(AssertionError):
        featurize_sample(sample)


def test_featurize_all_zero_mask_guard() -> None:
    sample = _make_multi_step_sample()
    tours = list(sample["tours"])
    sample["tours"] = [tours[0]] + tours

    with pytest.raises(AssertionError):
        featurize_sample(sample)


def _assert_graph_finite(graph: dict) -> None:
    for node in graph["seg_nodes"]:
        assert all(math.isfinite(v) for v in node)
    for node in graph["cand_nodes"]:
        assert all(math.isfinite(v) for v in node)
    for edges in graph["edges"].values():
        for feat in edges["feat"]:
            if isinstance(feat, list) and feat:
                assert all(math.isfinite(v) for v in feat)


def test_featurize_scale_robustness() -> None:
    small_segments = [(0.0, 2e-6)]
    h_small = 1e-6
    L_small = tour_length(0.0, 2e-6, h_small) * 1.1
    _, candidates_small, plan_small = dp_one_side_with_plan(list(small_segments), h_small, L_small)
    tours_small = reconstruct_one_side_plan(candidates_small, plan_small)
    small_sample = {
        "segments": small_segments,
        "h": h_small,
        "L": L_small,
        "tours": tours_small,
        "objective": "min_length",
    }

    large_segments = [(0.0, 1e4), (2e4, 3e4)]
    h_large = 1e3
    L_large = tour_length(0.0, 3e4, h_large) * 1.1
    _, candidates_large, plan_large = dp_one_side_with_plan(list(large_segments), h_large, L_large)
    tours_large = reconstruct_one_side_plan(candidates_large, plan_large)
    large_sample = {
        "segments": large_segments,
        "h": h_large,
        "L": L_large,
        "tours": tours_large,
        "objective": "min_length",
    }

    small_result = featurize_sample(small_sample)
    large_result = featurize_sample(large_sample)

    _assert_graph_finite(small_result["graph"])
    _assert_graph_finite(large_result["graph"])
