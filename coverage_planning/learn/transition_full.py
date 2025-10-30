from __future__ import annotations

from typing import Any, Dict, List, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.learn.transition_core import classify_x
from coverage_planning.learn.transition_one_side import (
    enumerate_one_side_transitions,
)

SNAP_TOL = max(EPS_GEOM * 4.0, 1e-8)

Interval = Tuple[float, float]

__all__ = ["enumerate_full_line_transitions"]


def _is_in_candidates(value: float, candidates: List[float]) -> bool:
    for candidate in candidates:
        if abs(candidate - value) <= SNAP_TOL:
            return True
    return False


def enumerate_full_line_transitions(
    left_ref: List[Interval],
    right: List[Interval],
    C_left: List[float],
    C_right: List[float],
    C_tail_right: List[float],
    h: float,
    L: float,
    state_left: List[Interval],
    state_right: List[Interval],
) -> Dict[str, Any]:
    """Legality oracle for Algorithm 4 (full-line DP). Enumerates actions only."""
    legal_right: List[int] = []
    legal_left: List[int] = []
    legal_bridges: List[Tuple[int, int]] = []

    if right and C_right and state_right:
        right_legality = enumerate_one_side_transitions(
            right, state_right, C_right, h, L
        )
        legal_right = sorted(right_legality["legal_q_idx"])

    if left_ref and C_left and state_left:
        left_legality = enumerate_one_side_transitions(
            left_ref, state_left, C_left, h, L
        )
        legal_left = sorted(left_legality["legal_q_idx"])

    if not (left_ref and right and C_left and C_right and state_left and state_right):
        return {
            "legal_right_no_bridge": legal_right,
            "legal_left_no_bridge": legal_left,
            "legal_bridges": legal_bridges,
            "allow_bridge": False,
        }

    tail_candidates = C_tail_right

    for p_idx, P in enumerate(C_left):
        try:
            kind_left, _ = classify_x(P, state_left, eps=EPS_GEOM)
        except ValueError:
            continue
        if kind_left != "inseg":
            continue
        p = -P
        for q_idx, q in enumerate(C_right):
            try:
                kind_right, _ = classify_x(q, state_right, eps=EPS_GEOM)
            except ValueError:
                continue
            if kind_right != "inseg":
                continue
            length = tour_length(p, q, h)
            if length > L + TOL_NUM:
                continue
            p_star = find_maximal_p(q, h, L)
            if abs(p - p_star) > TOL_NUM:
                continue
            if not tail_candidates or not _is_in_candidates(q, tail_candidates):
                continue
            legal_bridges.append((p_idx, q_idx))

    legal_bridges.sort()
    return {
        "legal_right_no_bridge": legal_right,
        "legal_left_no_bridge": legal_left,
        "legal_bridges": legal_bridges,
        "allow_bridge": bool(legal_bridges),
    }
