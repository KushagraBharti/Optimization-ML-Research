from __future__ import annotations

from coverage_planning.algs.geometry import tour_length
from coverage_planning.learn.transition_full import enumerate_full_line_transitions


def test_full_left_only_actions() -> None:
    left = [(0.0, 3.0)]
    state_left = list(left)
    result = enumerate_full_line_transitions(
        left_ref=left,
        right=[],
        C_left=[0.0, 3.0],
        C_right=[],
        C_tail_right=[],
        h=2.0,
        L=20.0,
        state_left=state_left,
        state_right=[],
    )
    assert result["allow_bridge"] is False
    assert result["legal_right_no_bridge"] == []
    assert result["legal_left_no_bridge"] == [0, 1]


def test_full_right_only_actions() -> None:
    right = [(0.0, 4.0)]
    state_right = list(right)
    result = enumerate_full_line_transitions(
        left_ref=[],
        right=right,
        C_left=[],
        C_right=[0.0, 4.0],
        C_tail_right=[4.0],
        h=1.5,
        L=25.0,
        state_left=[],
        state_right=state_right,
    )
    assert result["allow_bridge"] is False
    assert result["legal_left_no_bridge"] == []
    assert result["legal_right_no_bridge"] == [0, 1]


def test_full_bridge_exists() -> None:
    left = [(0.0, 3.0)]
    right = [(0.0, 3.0)]
    state_left = list(left)
    state_right = list(right)
    L = tour_length(-3.0, 3.0, h=2.0) + 1e-6
    result = enumerate_full_line_transitions(
        left_ref=left,
        right=right,
        C_left=[0.0, 3.0],
        C_right=[0.0, 3.0],
        C_tail_right=[3.0],
        h=2.0,
        L=L,
        state_left=state_left,
        state_right=state_right,
    )
    assert result["allow_bridge"] is True
    assert result["legal_bridges"] == [(1, 1)]


def test_full_bridge_disallowed_when_budget_small() -> None:
    left = [(0.0, 3.0)]
    right = [(0.0, 3.0)]
    state_left = list(left)
    state_right = list(right)
    L = 10.0
    result = enumerate_full_line_transitions(
        left_ref=left,
        right=right,
        C_left=[0.0, 3.0],
        C_right=[0.0, 3.0],
        C_tail_right=[3.0],
        h=2.0,
        L=L,
        state_left=state_left,
        state_right=state_right,
    )
    assert result["allow_bridge"] is False
    assert result["legal_bridges"] == []
