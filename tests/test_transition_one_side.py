from __future__ import annotations

from coverage_planning.algs.geometry import tour_length
from coverage_planning.learn.transition_one_side import enumerate_one_side_transitions


def test_one_side_all_candidates_legal() -> None:
    original = [(0.0, 3.0), (5.0, 7.0)]
    state = list(original)
    candidates = [0.0, 3.0, 7.0]
    result = enumerate_one_side_transitions(original, state, candidates, h=2.0, L=40.0)

    assert result["legal_q_idx"] == [0, 1, 2]
    assert set(result["case_per_q"].values()) == {"case1"}
    for q_idx in result["legal_q_idx"]:
        assert result["legal_p_idx_per_q"][q_idx] == [0]


def test_one_side_gap_start_uses_case2() -> None:
    original = [(0.0, 3.0), (5.0, 7.0)]
    state = [(5.0, 7.0)]
    candidates = [0.0, 3.0, 5.0, 7.0]
    L = 14.7

    result = enumerate_one_side_transitions(original, state, candidates, h=2.0, L=L)

    assert 3 in result["legal_q_idx"]
    assert result["case_per_q"][3] == "case2"
    assert result["legal_p_idx_per_q"][3] == [2]


def test_one_side_case3_single_maximal_option() -> None:
    original = [(0.0, 3.0)]
    state = list(original)
    candidates = [0.0, 1.0, 2.0, 3.0]
    L = tour_length(1.0, 2.0, h=1.0)

    result = enumerate_one_side_transitions(original, state, candidates, h=1.0, L=L)

    assert result["case_per_q"][2] == "case3"
    assert result["legal_p_idx_per_q"][2] == [1]
    assert 3 not in result["legal_q_idx"]


def test_case2_respects_current_state() -> None:
    original = [(0.0, 3.0), (5.0, 7.0)]
    # First segment fully covered previously; remaining uncovered starts at 6.0
    state = [(6.0, 7.0)]
    candidates = [0.0, 3.0, 5.0, 6.0, 7.0]
    L = 14.7

    result = enumerate_one_side_transitions(original, state, candidates, h=2.0, L=L)
    q_idx = candidates.index(7.0)
    assert q_idx not in result["legal_q_idx"]


def test_candidate_snapping_stability() -> None:
    original = [(0.0, 3.0)]
    state = list(original)
    base_candidates = [0.0, 1.5, 3.0]
    res_base = enumerate_one_side_transitions(original, state, base_candidates, h=1.5, L=20.0)

    perturb = [c + (1e-10 if idx == 1 else 0.0) for idx, c in enumerate(base_candidates)]
    res_shift = enumerate_one_side_transitions(original, state, perturb, h=1.5, L=20.0)

    assert res_base["legal_q_idx"] == res_shift["legal_q_idx"]
    for q_idx in res_base["legal_q_idx"]:
        assert res_base["legal_p_idx_per_q"][q_idx] == res_shift["legal_p_idx_per_q"][q_idx]
