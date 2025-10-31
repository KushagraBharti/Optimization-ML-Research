"""Feature extraction utilities for reference DP plans.

The featurizer mirrors the decision surface of the reference dynamic programs.
It emits per-step legality masks (dense or sparse) plus lightweight graph
features over segments and candidate endpoints.  The implementation is careful
to fence off legacy heuristics: all legality checks are sourced from the
reference transition oracles.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import dp_full_with_plan, dp_one_side_with_plan
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.learn.transition_core import trim_covered
from coverage_planning.learn.transition_full import enumerate_full_line_transitions
from coverage_planning.learn.transition_one_side import enumerate_one_side_transitions

SNAP_TOL = max(EPS_GEOM * 4.0, 1e-8)

__all__ = ["featurize_sample"]

Mask = Dict[str, object]
Step = Dict[str, object]
Interval = Tuple[float, float]


def _find_candidate_index(candidates: Sequence[float], value: float) -> int:
    best_idx: int | None = None
    best_diff = float("inf")
    for idx, candidate in enumerate(candidates):
        diff = abs(candidate - value)
        if diff < best_diff - 1e-15:
            best_diff = diff
            best_idx = idx
    if best_idx is None:
        raise ValueError(f"value {value:.12f} not found among candidates")
    return best_idx


def _build_mask(
    legal_q_idx: Iterable[int],
    legal_p_idx_per_q: Mapping[int, Iterable[int]],
    candidate_count: int,
    sparse_threshold: int,
) -> Mask:
    q_set = sorted(set(int(idx) for idx in legal_q_idx))
    if not q_set:
        raise AssertionError("transition mask is empty")

    use_dense: bool
    if sparse_threshold >= 0 and candidate_count > sparse_threshold:
        use_dense = False
    else:
        use_dense = sparse_threshold >= 0 and len(q_set) > sparse_threshold

    if use_dense:
        mask_right = [0] * candidate_count
        mask_left_given_right: List[List[int]] = [[] for _ in range(candidate_count)]
        for q_idx in q_set:
            mask_right[q_idx] = 1
            p_values = sorted(set(int(p) for p in legal_p_idx_per_q.get(q_idx, [])))
            mask_left_given_right[q_idx] = p_values
        if not any(mask_right):
            raise AssertionError("dense mask does not mark any legal right endpoint")
        return {
            "format": "dense",
            "mask_right": mask_right,
            "mask_left_given_right": mask_left_given_right,
        }

    legal_pairs: List[Tuple[int, int]] = []
    for q_idx in q_set:
        for p_idx in sorted(set(int(p) for p in legal_p_idx_per_q.get(q_idx, []))):
            legal_pairs.append((q_idx, p_idx))
    if not legal_pairs:
        raise AssertionError("sparse mask enumerated zero legal (p, q) pairs")
    return {
        "format": "sparse",
        "legal_right": q_set,
        "legal_pairs": legal_pairs,
    }


def _ensure_action_present(mask: Mask, q_idx: int, p_idx: int) -> None:
    if mask["format"] == "dense":
        mask_right = mask["mask_right"]
        mask_left_given_right = mask["mask_left_given_right"]
        assert mask_right[q_idx] == 1, "chosen q_idx not legal under dense mask"
        assert p_idx in mask_left_given_right[q_idx], "chosen (p_idx, q_idx) not legal under dense mask"
    else:
        legal_right = mask["legal_right"]
        legal_pairs = mask["legal_pairs"]
        assert q_idx in legal_right, "chosen q_idx not legal under sparse mask"
        assert (q_idx, p_idx) in legal_pairs, "chosen (p_idx, q_idx) not legal under sparse mask"


def _remaining_values(tours: Sequence[Tuple[float, float]], h: float) -> List[float]:
    remaining = sum(tour_length(min(p, q), max(p, q), h) for p, q in tours)
    values: List[float] = []
    for p, q in tours:
        values.append(remaining)
        remaining -= tour_length(min(p, q), max(p, q), h)
    return values


def _graph_from_segments_candidates(
    segments: Sequence[Interval],
    candidate_positions: Sequence[float],
) -> Dict[str, object]:
    seg_nodes = [
        [float(a), float(b), float(b - a), float((a + b) * 0.5)]
        for a, b in segments
    ]
    if not candidate_positions:
        cand_nodes = [[0.0, 0.0, 0.0]]
    else:
        cand_nodes = [
            [float(c), float(abs(c)), float(1.0 if c >= 0.0 else -1.0)]
            for c in candidate_positions
        ]
    edge_idx: List[Tuple[int, int]] = []
    edge_feat: List[List[float]] = []
    for seg_idx, (a, b) in enumerate(segments):
        for cand_idx, c in enumerate(candidate_positions):
            if a - EPS_GEOM <= c <= b + EPS_GEOM:
                edge_idx.append((seg_idx, cand_idx))
                edge_feat.append([float(c - a), float(b - c)])
    graph = {
        "seg_nodes": seg_nodes,
        "cand_nodes": cand_nodes,
        "edges": {
            "seg_to_cand": {
                "idx": edge_idx,
                "feat": edge_feat,
            }
        },
    }
    return graph


def _reflect_segments(segments: Sequence[Interval]) -> List[Interval]:
    reflected = [(-b, -a) for a, b in segments]
    reflected.sort(key=lambda seg: seg[0])
    return reflected


def _reflect_tour_to_left_domain(tour: Tuple[float, float]) -> Tuple[float, float]:
    p, q = tour
    return (abs(q), abs(p))


def _restore_left_tour(tour_ref: Tuple[float, float]) -> Tuple[float, float]:
    p_ref, q_ref = tour_ref
    return (-q_ref, -p_ref)


def _split_segments(
    segments: Sequence[Interval],
) -> Tuple[List[Interval], List[Interval]]:
    left: List[Interval] = []
    right: List[Interval] = []
    for a, b in segments:
        if b <= 0.0 + EPS_GEOM:
            left.append((a, b))
        elif a >= 0.0 - EPS_GEOM:
            right.append((a, b))
        else:
            left.append((a, 0.0))
            right.append((0.0, b))
    left.sort(key=lambda seg: seg[0])
    right.sort(key=lambda seg: seg[0])
    return left, right


def _featurize_one_side(
    segments: Sequence[Interval],
    tours: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    *,
    sparse_threshold: int,
) -> Tuple[List[Step], List[float]]:
    segs = sorted((float(a), float(b)) for a, b in segments)
    if not segs:
        return [], []
    _, candidates, _plan = dp_one_side_with_plan(list(segs), h, L, tol=TOL_NUM)
    extra_points = {float(seg[0]) for seg in segs}
    candidate_list = sorted(set(float(c) for c in candidates) | extra_points)
    state: List[Interval] = [tuple(seg) for seg in segs]
    steps: List[Step] = []
    values = _remaining_values(tours, h)

    for order_idx, (p, q) in enumerate(tours):
        q_idx = _find_candidate_index(candidate_list, q)
        p_idx = _find_candidate_index(candidate_list, p)
        mask_data = enumerate_one_side_transitions(
            segs, state, candidate_list, h, L
        )
        mask = _build_mask(
            mask_data["legal_q_idx"],
            mask_data["legal_p_idx_per_q"],
            len(candidate_list),
            sparse_threshold,
        )
        _ensure_action_present(mask, q_idx, p_idx)
        case_map = mask_data.get("case_per_q", {})
        case_label = case_map.get(q_idx, "case-unknown")
        steps.append(
            {
                "index": order_idx,
                "case": case_label,
                "tour": (float(p), float(q)),
                "mask": mask,
                "y_left": p_idx,
                "y_right": q_idx,
                "value": values[order_idx],
            }
        )
        state = trim_covered(state, p, q)

    return steps, candidate_list


def _featurize_left_side(
    segments: Sequence[Interval],
    tours: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    *,
    sparse_threshold: int,
) -> Tuple[List[Step], List[float]]:
    if not segments:
        return [], []
    segments_ref = _reflect_segments(segments)
    _, candidates_ref, _plan = dp_one_side_with_plan(list(segments_ref), h, L, tol=TOL_NUM)
    extra_points = {float(seg[0]) for seg in segments_ref}
    candidate_list = sorted(set(float(c) for c in candidates_ref) | extra_points)
    state_ref: List[Interval] = [tuple(seg) for seg in segments_ref]
    steps: List[Step] = []
    values = _remaining_values(tours, h)

    for order_idx, (p, q) in enumerate(tours):
        p_ref, q_ref = _reflect_tour_to_left_domain((p, q))
        q_idx = _find_candidate_index(candidate_list, q_ref)
        p_idx = _find_candidate_index(candidate_list, p_ref)
        mask_data = enumerate_one_side_transitions(
            segments_ref, state_ref, candidate_list, h, L
        )
        mask = _build_mask(
            mask_data["legal_q_idx"],
            mask_data["legal_p_idx_per_q"],
            len(candidate_list),
            sparse_threshold,
        )
        _ensure_action_present(mask, q_idx, p_idx)
        case_map = mask_data.get("case_per_q", {})
        case_label = f"left_{case_map.get(q_idx, 'case-unknown')}"
        steps.append(
            {
                "index": order_idx,
                "case": case_label,
                "tour": (float(p), float(q)),
                "mask": mask,
                "y_left": p_idx,
                "y_right": q_idx,
                "value": values[order_idx],
            }
        )
        state_ref = trim_covered(state_ref, p_ref, q_ref)

    return steps, [ -float(c) for c in candidate_list ]


def _featurize_right_side(
    segments: Sequence[Interval],
    tours: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    *,
    sparse_threshold: int,
) -> Tuple[List[Step], List[float]]:
    if not segments:
        return [], []
    return _featurize_one_side(
        segments,
        tours,
        h,
        L,
        sparse_threshold=sparse_threshold,
    )


def _featurize_full_line(
    segments: Sequence[Interval],
    tours: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    *,
    sparse_threshold: int,
) -> Tuple[List[Step], List[float]]:
    if not segments:
        return [], []
    _cost, _solver_tours, meta = dp_full_with_plan(list(segments), h, L)
    context = meta.get("plan_context")
    if context is None:
        raise RuntimeError("plan context not available for full-line featurization")

    left_segments, right_segments = _split_segments(segments)
    left_ref = _reflect_segments(left_segments)
    state_left: List[Interval] = [tuple(seg) for seg in left_ref]
    state_right: List[Interval] = [tuple(seg) for seg in right_segments]

    candidate_left = list(context.C_left)
    candidate_right = list(context.C_right)
    candidate_tail = list(context.C_tail)
    candidate_left_aug = sorted(set(candidate_left) | {seg[0] for seg in left_ref})
    candidate_right_aug = sorted(set(candidate_right) | {seg[0] for seg in right_segments})

    steps: List[Step] = []
    values = _remaining_values(tours, h)
    order_idx = 0

    for step_idx, (p, q) in enumerate(tours):
        if p <= 0.0 + EPS_GEOM and q <= 0.0 + EPS_GEOM:
            # Left-only tour (actual coordinates <= 0).
            p_ref, q_ref = _reflect_tour_to_left_domain((p, q))
            if not candidate_left_aug:
                raise AssertionError("left candidates unavailable for left-side tour")
            q_idx = _find_candidate_index(candidate_left_aug, q_ref)
            p_idx = _find_candidate_index(candidate_left_aug, p_ref)
            mask_data = enumerate_one_side_transitions(
                left_ref, state_left, candidate_left_aug, h, L
            )
            mask = _build_mask(
                mask_data["legal_q_idx"],
                mask_data["legal_p_idx_per_q"],
                len(candidate_left_aug),
                sparse_threshold,
            )
            _ensure_action_present(mask, q_idx, p_idx)
            case_label = f"left_{mask_data.get('case_per_q', {}).get(q_idx, 'case-unknown')}"
            steps.append(
                {
                    "index": order_idx,
                    "case": case_label,
                    "tour": (float(p), float(q)),
                    "mask": mask,
                    "y_left": p_idx,
                    "y_right": q_idx,
                    "value": values[step_idx],
                }
            )
            state_left = trim_covered(state_left, p_ref, q_ref)
            order_idx += 1
        elif p >= 0.0 - EPS_GEOM and q >= 0.0 - EPS_GEOM:
            # Right-only tour.
            if not candidate_right_aug:
                raise AssertionError("right candidates unavailable for right-side tour")
            q_idx = _find_candidate_index(candidate_right_aug, q)
            p_idx = _find_candidate_index(candidate_right_aug, p)
            mask_data = enumerate_one_side_transitions(
                right_segments, state_right, candidate_right_aug, h, L
            )
            mask = _build_mask(
                mask_data["legal_q_idx"],
                mask_data["legal_p_idx_per_q"],
                len(candidate_right_aug),
                sparse_threshold,
            )
            _ensure_action_present(mask, q_idx, p_idx)
            case_label = f"right_{mask_data.get('case_per_q', {}).get(q_idx, 'case-unknown')}"
            steps.append(
                {
                    "index": order_idx,
                    "case": case_label,
                    "tour": (float(p), float(q)),
                    "mask": mask,
                    "y_left": p_idx,
                    "y_right": q_idx,
                    "value": values[step_idx],
                }
            )
            state_right = trim_covered(state_right, p, q)
            order_idx += 1
        else:
            # Bridge tour spanning both halves.
            if not (candidate_left and candidate_right and candidate_tail):
                raise AssertionError("bridge tour requires candidate sets on both sides")
            P = abs(p)
            q_idx = _find_candidate_index(candidate_right, q)
            p_idx = _find_candidate_index(candidate_left, P)
            mask_data_full = enumerate_full_line_transitions(
                left_ref,
                right_segments,
                candidate_left,
                candidate_right,
                candidate_tail,
                h,
                L,
                state_left,
                state_right,
            )
            legal_pairs = mask_data_full.get("legal_bridges", [])
            legal_pairs = [(int(q_raw), int(p_raw)) for p_raw, q_raw in mask_data_full.get("legal_bridges", [])]
            mask = {
                "format": "sparse",
                "legal_right": sorted({pair[0] for pair in legal_pairs}),
                "legal_pairs": legal_pairs,
            }
            pair = (q_idx, p_idx)
            if pair not in mask["legal_pairs"]:
                raise AssertionError("bridge pair not legal under transition oracle")
            steps.append(
                {
                    "index": order_idx,
                    "case": "bridge",
                    "tour": (float(p), float(q)),
                    "mask": mask,
                    "y_left": p_idx,
                    "y_right": q_idx,
                    "value": values[step_idx],
                }
            )
            # Bridge covers both halves out to the chosen endpoints.
            state_left = trim_covered(state_left, 0.0, P)
            state_right = trim_covered(state_right, 0.0, q)
            order_idx += 1

    candidate_positions: List[float] = []
    candidate_positions.extend([-float(c) for c in candidate_left_aug])
    candidate_positions.extend(float(c) for c in candidate_right_aug)
    candidate_positions.extend(float(c) for c in candidate_tail)
    candidate_positions.append(0.0)
    candidates_unique = sorted(set(candidate_positions))
    return steps, candidates_unique


def _categorise_segments(segments: Sequence[Interval]) -> str:
    has_negative = any(b <= -EPS_GEOM for a, b in segments)
    has_positive = any(a >= EPS_GEOM for a, b in segments)
    if has_negative and not has_positive:
        return "left"
    if has_positive and not has_negative:
        return "right"
    return "full"


def _validate_stored_values(
    expected: Sequence[float] | None,
    observed: Sequence[float],
) -> List[float]:
    if expected is None:
        return list(observed)
    if len(expected) != len(observed):
        raise AssertionError("stored value vector length mismatch")
    for idx, (exp, obs) in enumerate(zip(expected, observed)):
        if not math.isclose(exp, obs, rel_tol=1e-6, abs_tol=2 * TOL_NUM):
            raise AssertionError(f"stored step value #{idx} disagrees with recomputed budget")
    return [float(v) for v in expected]


def featurize_sample(sample: Mapping[str, object], sparse_threshold: int = 64) -> Dict[str, object]:
    """Materialise learning features for a labelled sample.

    Parameters
    ----------
    sample:
        Mapping with the keys ``segments``, ``h``, ``L`` and ``tours``.  Optional
        ``stored_values`` is validated against recomputed DP budgets.
    sparse_threshold:
        Switch to sparse masks when the number of legal right endpoints is at
        most this many.
    """

    segments = [tuple(map(float, seg)) for seg in sample["segments"]]  # type: ignore[index]
    h = float(sample["h"])  # type: ignore[index]
    L = float(sample["L"])  # type: ignore[index]
    tours = [tuple(map(float, tour)) for tour in sample["tours"]]  # type: ignore[index]
    stored_values = sample.get("stored_values")
    stored_vec = [float(v) for v in stored_values] if stored_values is not None else None

    kind = _categorise_segments(segments)
    if kind == "right":
        steps, candidate_positions = _featurize_right_side(
            segments,
            tours,
            h,
            L,
            sparse_threshold=sparse_threshold,
        )
    elif kind == "left":
        steps, candidate_positions = _featurize_left_side(
            segments,
            tours,
            h,
            L,
            sparse_threshold=sparse_threshold,
        )
    else:
        steps, candidate_positions = _featurize_full_line(
            segments,
            tours,
            h,
            L,
            sparse_threshold=sparse_threshold,
        )

    if not steps and tours:
        raise AssertionError("featurizer produced zero steps for non-empty tour set")

    observed_values = [step["value"] for step in steps]
    validated_values = _validate_stored_values(stored_vec, observed_values)
    for value, step in zip(validated_values, steps):
        step["value"] = value

    graph = _graph_from_segments_candidates(segments, candidate_positions)

    return {
        "graph": graph,
        "steps": steps,
        "objective": sample.get("objective", "min_length"),
        "dp_cost": sample.get("cost"),
        "metadata": {
            "candidate_positions": candidate_positions,
            "sparse_threshold": sparse_threshold,
        },
    }
