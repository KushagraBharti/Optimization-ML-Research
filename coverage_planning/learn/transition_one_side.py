from __future__ import annotations

from typing import Any, Dict, List, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.learn.transition_core import (
    classify_x,
    covers_no_gap,
    is_maximal_pair,
)

SNAP_TOL = max(EPS_GEOM * 4.0, 1e-8)

Interval = Tuple[float, float]

__all__ = ["enumerate_one_side_transitions"]


def _snap_candidate_index(value: float, candidates: List[float]) -> int | None:
    best_idx: int | None = None
    best_diff = float("inf")
    for idx, candidate in enumerate(candidates):
        diff = abs(candidate - value)
        if diff < best_diff - 1e-16:
            best_diff = diff
            best_idx = idx
    if best_idx is not None and best_diff <= SNAP_TOL:
        return best_idx
    return None


def enumerate_one_side_transitions(
    original_segments: List[Interval],
    current_state: List[Interval],
    candidates: List[float],
    h: float,
    L: float,
) -> Dict[str, Any]:
    """Legality oracle for Algorithm 3 (one-side DP). No costs, no DP recursion."""
    if not original_segments or not candidates:
        return {"legal_q_idx": [], "legal_p_idx_per_q": {}, "case_per_q": {}}
    if not current_state:
        return {"legal_q_idx": [], "legal_p_idx_per_q": {}, "case_per_q": {}}

    a1 = original_segments[0][0]
    a1_idx = _snap_candidate_index(a1, candidates)

    legal_q_idx: List[int] = []
    legal_p_idx_per_q: Dict[int, List[int]] = {}
    case_per_q: Dict[int, str] = {}

    for q_idx, q in enumerate(candidates):
        try:
            kind_state, _ = classify_x(q, current_state, eps=EPS_GEOM)
        except ValueError:
            continue
        if kind_state != "inseg":
            continue

        p_star = find_maximal_p(q, h, L)
        try:
            kind_origin, origin_idx = classify_x(p_star, original_segments, eps=EPS_GEOM)
        except ValueError:
            continue

        candidates_for_q: List[int] = []
        case_label: str | None = None

        if kind_origin == "before":
            if a1_idx is None:
                continue
            if not current_state:
                continue
            current_start = current_state[0][0]
            if abs(current_start - a1) > EPS_GEOM:
                continue
            length = tour_length(a1, q, h)
            if length <= L + TOL_NUM:
                candidates_for_q = [a1_idx]
                case_label = "case1"

        elif kind_origin == "gap" and origin_idx is not None:
            case_label = "case2"
            for seg_idx in range(origin_idx + 1, len(original_segments)):
                a_k = original_segments[seg_idx][0]
                start_idx = _snap_candidate_index(a_k, candidates)
                if start_idx is None:
                    continue
                if not current_state:
                    break
                try:
                    state_kind, state_idx = classify_x(a_k, current_state, eps=EPS_GEOM)
                except ValueError:
                    continue
                if state_kind != "inseg" or state_idx is None:
                    continue
                current_left = current_state[state_idx][0]
                if abs(current_left - a_k) > EPS_GEOM:
                    continue
                if not covers_no_gap(a_k, q, current_state, eps=EPS_GEOM):
                    continue
                length = tour_length(a_k, q, h)
                if length <= L + TOL_NUM:
                    candidates_for_q.append(start_idx)

        elif kind_origin == "inseg":
            if not is_maximal_pair(p_star, q, h, L, tol=TOL_NUM):
                continue
            start_idx = _snap_candidate_index(p_star, candidates)
            if start_idx is None:
                continue
            candidates_for_q = [start_idx]
            case_label = "case3"

        if not candidates_for_q or case_label is None:
            continue

        candidates_for_q = sorted(set(candidates_for_q))
        legal_q_idx.append(q_idx)
        legal_p_idx_per_q[q_idx] = candidates_for_q
        case_per_q[q_idx] = case_label

    legal_q_idx.sort()
    return {
        "legal_q_idx": legal_q_idx,
        "legal_p_idx_per_q": legal_p_idx_per_q,
        "case_per_q": case_per_q,
    }
