from __future__ import annotations

import math
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference.dp_full_line_ref import FullLinePlanContext, dp_full_line_with_plan
from coverage_planning.algs.reference.dp_one_side_ref import dp_one_side_with_plan
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.learn.transition_core import trim_covered
from coverage_planning.learn.transition_full import enumerate_full_line_transitions
from coverage_planning.learn.transition_one_side import enumerate_one_side_transitions

SNAP_TOL = max(EPS_GEOM * 4.0, 1e-8)
RATIO_LIMIT = 1e6

__all__ = ["featurize_sample", "featurize_samples"]


def _as_float_segments(segments: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    return [tuple(float(v) for v in seg) for seg in segments]


def _extract_objective(sample: Dict[str, object]) -> str:
    if "objective" in sample:
        return str(sample["objective"])
    meta = sample.get("meta") or sample.get("metadata") or {}
    if isinstance(meta, dict) and "objective" in meta:
        return str(meta["objective"])
    gold = sample.get("gold")
    if isinstance(gold, dict):
        gmeta = gold.get("meta")
        if isinstance(gmeta, dict) and "objective" in gmeta:
            return str(gmeta["objective"])
    raise ValueError("sample missing objective metadata")


def _extract_tours(sample: Dict[str, object]) -> List[Tuple[float, float]]:
    if "tours" in sample:
        tours_src = sample["tours"]
    else:
        gold = sample.get("gold")
        if isinstance(gold, dict) and "tours" in gold:
            tours_src = gold["tours"]
        else:
            raise ValueError("sample missing tours")
    tours: List[Tuple[float, float]] = []
    for pair in tours_src:
        p, q = pair
        tours.append((float(p), float(q)))
    return tours


def _extract_stored_values(sample: Dict[str, object]) -> Optional[List[float]]:
    candidate_keys = ("stored_values", "values", "cost_to_go")

    def _read(container: Optional[Dict[str, object]]) -> Optional[List[float]]:
        if not isinstance(container, dict):
            return None
        for key in candidate_keys:
            seq = container.get(key)
            if seq is None:
                continue
            return [float(v) for v in seq]
        return None

    for container in (
        sample,
        sample.get("gold") if isinstance(sample.get("gold"), dict) else None,
        sample.get("meta") if isinstance(sample.get("meta"), dict) else None,
        sample.get("metadata") if isinstance(sample.get("metadata"), dict) else None,
    ):
        values = _read(container)
        if values is not None:
            return values
    return None


def _safe_div(num: float, denom: float) -> float:
    if abs(denom) <= TOL_NUM:
        return 0.0
    value = num / denom
    if not math.isfinite(value):
        return 0.0
    if value > RATIO_LIMIT:
        return RATIO_LIMIT
    if value < -RATIO_LIMIT:
        return -RATIO_LIMIT
    return value


def _compute_scale(segments: Sequence[Tuple[float, float]], h: float, L: float) -> float:
    extrema = [1.0, abs(h), abs(L)]
    for a, b in segments:
        extrema.append(abs(a))
        extrema.append(abs(b))
    return max(extrema)


def _segment_features(segments: Sequence[Tuple[float, float]], scale: float) -> List[List[float]]:
    features: List[List[float]] = []
    for a, b in segments:
        length = b - a
        center = 0.5 * (a + b)
        dist_min = min(abs(a), abs(b))
        dist_max = max(abs(a), abs(b))
        if b <= EPS_GEOM:
            side_flag = -1.0
        elif a >= -EPS_GEOM:
            side_flag = 1.0
        else:
            side_flag = 0.0
        features.append(
            [
                a / scale,
                b / scale,
                length / scale,
                center / scale,
                dist_min / scale,
                dist_max / scale,
                side_flag,
            ]
        )
    return features


def _augment_candidates(base: Sequence[float], additions: Sequence[float]) -> List[float]:
    values = list(base)
    for value in additions:
        if all(abs(value - existing) > SNAP_TOL for existing in values):
            values.append(value)
    values.sort()
    return values


def _candidate_features(
    candidates: Sequence[float],
    segments: Sequence[Tuple[float, float]],
    scale: float,
) -> List[List[float]]:
    seg_lefts = [a for a, _ in segments]
    seg_rights = [b for _, b in segments]
    features: List[List[float]] = []
    for value in candidates:
        is_left = any(abs(value - a) <= EPS_GEOM for a in seg_lefts)
        is_right = any(abs(value - b) <= EPS_GEOM for b in seg_rights)
        seg_idx = -1
        for idx, (a, b) in enumerate(segments):
            if a - EPS_GEOM <= value <= b + EPS_GEOM:
                seg_idx = idx
                break
        features.append(
            [
                value / scale,
                1.0 if is_left else 0.0,
                1.0 if is_right else 0.0,
                0.0,
                1.0 if abs(value) <= EPS_GEOM else 0.0,
                float(seg_idx),
            ]
        )
    return features


def _ensure_finite_features(features: Sequence[Sequence[float]]) -> None:
    for row in features:
        for value in row:
            if not math.isfinite(value):
                raise ValueError("Encountered non-finite feature value during featurization")


def _edge_indices_adjacency(count: int) -> Tuple[List[List[int]], List[List[float]]]:
    if count < 2:
        return [[], []], []
    src, dst = [], []
    feats: List[List[float]] = []
    for idx in range(count - 1):
        src.append(idx)
        dst.append(idx + 1)
        feats.append([0.0])
    return [src, dst], feats


def _build_graph(
    segments: Sequence[Tuple[float, float]],
    candidates: Sequence[float],
    h: float,
    L: float,
) -> Dict[str, object]:
    scale = _compute_scale(segments, h, L)
    seg_features = _segment_features(segments, scale)
    cand_features = _candidate_features(candidates, segments, scale)

    seg_edges_idx, seg_edges_feat = _edge_indices_adjacency(len(segments))
    for idx in range(len(seg_edges_feat)):
        gap = segments[idx + 1][0] - segments[idx][1]
        seg_edges_feat[idx][0] = gap / scale

    cand_edges_idx, cand_edges_feat = _edge_indices_adjacency(len(candidates))
    for idx in range(len(cand_edges_feat)):
        delta = candidates[idx + 1] - candidates[idx]
        cand_edges_feat[idx][0] = delta / scale

    cand_seg_idx_src: List[int] = []
    cand_seg_idx_dst: List[int] = []
    cand_seg_feat: List[List[float]] = []
    for c_idx, value in enumerate(candidates):
        for s_idx, (a, b) in enumerate(segments):
            if a - EPS_GEOM <= value <= b + EPS_GEOM:
                cand_seg_idx_src.append(c_idx)
                cand_seg_idx_dst.append(s_idx)
                cand_seg_feat.append([(value - a) / scale, (b - value) / scale])
                break

    _ensure_finite_features(seg_features)
    _ensure_finite_features(cand_features)
    _ensure_finite_features(seg_edges_feat)
    _ensure_finite_features(cand_edges_feat)
    _ensure_finite_features(cand_seg_feat)

    context = {
        "h_scaled": h / scale,
        "L_scaled": L / scale,
        "ratios": {
            "L_over_h": _safe_div(L, h),
        },
        "scale": scale,
    }

    return {
        "seg_nodes": seg_features,
        "cand_nodes": cand_features,
        "edges": {
            "seg_seg": {"index": seg_edges_idx, "feat": seg_edges_feat},
            "cand_cand": {"index": cand_edges_idx, "feat": cand_edges_feat},
            "cand_seg": {
                "index": [cand_seg_idx_src, cand_seg_idx_dst],
                "feat": cand_seg_feat,
            },
        },
        "context": context,
    }


def _snap_index(value: float, candidates: Sequence[float]) -> int:
    best_idx: Optional[int] = None
    best_diff = float("inf")
    for idx, candidate in enumerate(candidates):
        diff = abs(candidate - value)
        if diff < best_diff - 1e-16:
            best_diff = diff
            best_idx = idx
    if best_idx is not None and best_diff <= SNAP_TOL:
        return best_idx
    raise ValueError(f"Value {value:.12f} not found in candidate set")


def _split_left_right(
    segments: Sequence[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []
    for a, b in segments:
        if b <= 0.0 + EPS_GEOM:
            left.append((a, b))
        elif a >= 0.0 - EPS_GEOM:
            right.append((a, b))
        else:
            raise ValueError("Straddling segments are unsupported for featurization")
    left.sort(key=lambda seg: seg[0])
    right.sort(key=lambda seg: seg[0])
    return left, right


def _reflect_intervals(intervals: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    reflected = [(-b, -a) for a, b in intervals]
    reflected.sort(key=lambda seg: seg[0])
    return reflected


def _one_side_candidates(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    if not segments:
        return []
    _, candidates, _ = dp_one_side_with_plan(list(segments), h, L)
    starts = [a for a, _ in segments]
    augmented = _augment_candidates(candidates, starts)
    return augmented


def _full_line_candidates(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> Tuple[List[float], List[float], List[float]]:
    cost, tours, meta = dp_full_line_with_plan(list(segments), h, L)
    context = meta.get("plan_context")
    if not isinstance(context, FullLinePlanContext):
        return [], [], []
    return list(context.C_left), list(context.C_right), list(context.C_tail)


def _compute_value_targets(
    tours: Sequence[Tuple[float, float]],
    h: float,
    objective: str,
) -> List[float]:
    if objective == "min_length":
        remaining_costs = [tour_length(p, q, h) for p, q in tours]
        suffix = sum(remaining_costs)
        values: List[float] = []
        for cost in remaining_costs:
            values.append(suffix)
            suffix -= cost
        return values
    if objective == "min_tours":
        total = len(tours)
        return [float(total - idx) for idx in range(total)]
    raise ValueError(f"Unsupported objective: {objective}")


def _build_mask_from_mapping(
    mapping: Dict[int, List[int]],
    cand_count: int,
    sparse_threshold: int,
) -> Dict[str, object]:
    q_sorted = sorted(mapping.keys())
    if cand_count > sparse_threshold:
        pairs: List[Tuple[int, int]] = []
        for q_idx in q_sorted:
            for p_idx in sorted(mapping[q_idx]):
                pairs.append((q_idx, p_idx))
        return {
            "format": "sparse",
            "legal_right": q_sorted,
            "legal_pairs": pairs,
        }
    mask_right = [0] * cand_count
    mask_left: Dict[int, List[int]] = {}
    for q_idx in q_sorted:
        mask_right[q_idx] = 1
        mask_left[q_idx] = sorted(mapping[q_idx])
    return {
        "format": "dense",
        "mask_right": mask_right,
        "mask_left_given_right": mask_left,
    }


def _mask_has_legal(mask: Dict[str, object]) -> bool:
    if mask["format"] == "dense":
        return any(mask.get("mask_right", []))
    return bool(mask.get("legal_pairs"))


def _assert_nonzero_mask(
    mask: Dict[str, object],
    mapping: Dict[int, List[int]],
    is_terminal: bool,
    step_idx: int,
) -> None:
    if is_terminal:
        return
    if _mask_has_legal(mask):
        return
    if any(mapping.values()):
        return
    raise AssertionError(f"Step {step_idx} produced no legal actions (all-zero mask)")


def _collect_one_side_masks(
    original_segments: Sequence[Tuple[float, float]],
    current_state: Sequence[Tuple[float, float]],
    candidates: Sequence[float],
    h: float,
    L: float,
    sparse_threshold: int,
    actual_p: float,
    actual_q: float,
) -> Tuple[Dict[str, object], Dict[int, str], Dict[int, List[int]]]:
    enum_result = enumerate_one_side_transitions(
        list(original_segments),
        list(current_state),
        list(candidates),
        h,
        L,
    )
    q_to_ps: Dict[int, List[int]] = {}
    case_lookup: Dict[int, str] = {}
    for q_idx in enum_result["legal_q_idx"]:
        p_indices = enum_result["legal_p_idx_per_q"].get(q_idx, [])
        if not p_indices:
            continue
        case_lookup[q_idx] = enum_result["case_per_q"][q_idx]
        q_to_ps[q_idx] = sorted(p_indices)
    mask = _build_mask_from_mapping(q_to_ps, len(candidates), sparse_threshold)
    return mask, case_lookup, q_to_ps


def _collect_full_masks(
    left_ref_segments: Sequence[Tuple[float, float]],
    right_segments: Sequence[Tuple[float, float]],
    state_left_ref: Sequence[Tuple[float, float]],
    state_right: Sequence[Tuple[float, float]],
    C_left: Sequence[float],
    C_right: Sequence[float],
    C_tail: Sequence[float],
    global_candidates: Sequence[float],
    h: float,
    L: float,
    sparse_threshold: int,
) -> Tuple[Dict[str, object], Dict[int, str], Dict[int, List[int]]]:
    mapping: Dict[int, List[int]] = {}
    case_lookup: Dict[int, str] = {}

    # Right side legality (original coordinates)
    right_enum = enumerate_one_side_transitions(
        list(right_segments),
        list(state_right),
        list(C_right),
        h,
        L,
    )
    for q_idx in right_enum["legal_q_idx"]:
        q_val = C_right[q_idx]
        q_global = _snap_index(q_val, global_candidates)
        local_ps = []
        for p_idx in right_enum["legal_p_idx_per_q"][q_idx]:
            p_val = C_right[p_idx]
            p_global = _snap_index(p_val, global_candidates)
            local_ps.append(p_global)
        if local_ps:
            mapping.setdefault(q_global, []).extend(local_ps)
            case_lookup.setdefault(q_global, right_enum["case_per_q"][q_idx])

    # Left side legality (reflected coordinates converted back)
    left_enum = enumerate_one_side_transitions(
        list(left_ref_segments),
        list(state_left_ref),
        list(C_left),
        h,
        L,
    )
    for q_idx in left_enum["legal_q_idx"]:
        q_ref = C_left[q_idx]
        case_lookup_value = left_enum["case_per_q"][q_idx]
        for p_idx in left_enum["legal_p_idx_per_q"][q_idx]:
            p_ref = C_left[p_idx]
            actual_q = -p_ref
            actual_p = -q_ref
            q_global = _snap_index(actual_q, global_candidates)
            p_global = _snap_index(actual_p, global_candidates)
            mapping.setdefault(q_global, []).append(p_global)
            case_lookup.setdefault(q_global, case_lookup_value)

    # Bridges
    full_enum = enumerate_full_line_transitions(
        left_ref=list(left_ref_segments),
        right=list(right_segments),
        C_left=list(C_left),
        C_right=list(C_right),
        C_tail_right=list(C_tail),
        h=h,
        L=L,
        state_left=list(state_left_ref),
        state_right=list(state_right),
    )
    for p_idx, q_idx in full_enum["legal_bridges"]:
        p_val = -C_left[p_idx]
        q_val = C_right[q_idx]
        q_global = _snap_index(q_val, global_candidates)
        p_global = _snap_index(p_val, global_candidates)
        mapping.setdefault(q_global, []).append(p_global)
        case_lookup[q_global] = "bridge"

    # Deduplicate and sort
    for q_idx, points in mapping.items():
        mapping[q_idx] = sorted(set(points))

    mask = _build_mask_from_mapping(mapping, len(global_candidates), sparse_threshold)
    return mask, case_lookup, mapping


def _prepare_one_side_sample(
    sample: Dict[str, object],
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    sparse_threshold: int,
) -> Dict[str, object]:
    candidates = _one_side_candidates(segments, h, L)
    graph = _build_graph(segments, candidates, h, L)
    tours = _extract_tours(sample)
    objective = _extract_objective(sample)
    values = _compute_value_targets(tours, h, objective)
    stored_values = _extract_stored_values(sample)
    state = list(segments)

    steps: List[Dict[str, object]] = []
    for tour_idx, (p, q) in enumerate(tours):
        mask, case_lookup, mapping = _collect_one_side_masks(
            original_segments=segments,
            current_state=state,
            candidates=candidates,
            h=h,
            L=L,
            sparse_threshold=sparse_threshold,
            actual_p=p,
            actual_q=q,
        )
        is_terminal = tour_idx == len(tours) - 1
        _assert_nonzero_mask(mask, mapping, is_terminal, tour_idx)
        q_idx = _snap_index(q, candidates)
        p_idx = _snap_index(p, candidates)
        allowed = mapping.get(q_idx, [])
        if p_idx not in allowed:
            raise AssertionError("Chosen (p,q) pair is not legal under one-side enumerator")
        case = case_lookup.get(q_idx, "case1")
        value = values[tour_idx]
        if stored_values is not None and tour_idx < len(stored_values):
            stored_value = float(stored_values[tour_idx])
            if not math.isclose(stored_value, value, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError("Stored value mismatch for one-side instance")
            value = stored_value
        steps.append(
            {
                "mask": mask,
                "y_right": q_idx,
                "y_left": p_idx,
                "case": case,
                "value": value,
            }
        )
        state = trim_covered(list(state), p, q)

    return {
        "graph": graph,
        "steps": steps,
        "objective": objective,
        "meta": {
            "N_seg": len(segments),
            "N_cand": len(candidates),
            "num_steps": len(steps),
        },
    }


def _prepare_full_line_sample(
    sample: Dict[str, object],
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    sparse_threshold: int,
) -> Dict[str, object]:
    left_segments, right_segments = _split_left_right(segments)
    C_left_ref, C_right, C_tail = _full_line_candidates(segments, h, L)
    left_ref_segments = _reflect_intervals(left_segments)
    C_left_ref = _augment_candidates(C_left_ref, [a for a, _ in left_ref_segments])
    C_right = _augment_candidates(C_right, [a for a, _ in right_segments])
    left_candidates_actual = [-value for value in C_left_ref]
    global_candidates = sorted(
        set(left_candidates_actual + list(C_right) + [a for a, _ in segments])
    )
    graph = _build_graph(segments, global_candidates, h, L)

    tours = _extract_tours(sample)
    objective = _extract_objective(sample)
    values = _compute_value_targets(tours, h, objective)
    stored_values = _extract_stored_values(sample)

    state_left = list(left_segments)
    state_left_ref = list(left_ref_segments)
    state_right = list(right_segments)

    steps: List[Dict[str, object]] = []
    for idx, (p, q) in enumerate(tours):
        mask, case_lookup, mapping = _collect_full_masks(
            left_ref_segments=left_ref_segments,
            right_segments=right_segments,
            state_left_ref=state_left_ref,
            state_right=state_right,
            C_left=C_left_ref,
            C_right=C_right,
            C_tail=C_tail,
            global_candidates=global_candidates,
            h=h,
            L=L,
            sparse_threshold=sparse_threshold,
        )
        is_terminal = idx == len(tours) - 1
        _assert_nonzero_mask(mask, mapping, is_terminal, idx)
        q_idx = _snap_index(q, global_candidates)
        p_idx = _snap_index(p, global_candidates)
        allowed = mapping.get(q_idx, [])
        if p_idx not in allowed:
            raise AssertionError("Chosen bridge/side pair not legal under enumerators")

        is_bridge = p < -EPS_GEOM and q > EPS_GEOM
        case = case_lookup.get(q_idx, "bridge" if is_bridge else "case1")

        value = values[idx]
        if stored_values is not None and idx < len(stored_values):
            stored_value = float(stored_values[idx])
            if not math.isclose(stored_value, value, rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError("Stored value mismatch for full-line instance")
            value = stored_value

        steps.append(
            {
                "mask": mask,
                "y_right": q_idx,
                "y_left": p_idx,
                "case": case,
                "value": value,
            }
        )

        if is_bridge:
            state_left = trim_covered(state_left, p, 0.0)
            state_left_ref = trim_covered(state_left_ref, 0.0, -p)
            state_right = trim_covered(state_right, 0.0, q)
        elif q <= EPS_GEOM:
            state_left = trim_covered(state_left, p, q)
            state_left_ref = trim_covered(state_left_ref, -q, -p)
        else:
            state_right = trim_covered(state_right, p, q)

    return {
        "graph": graph,
        "steps": steps,
        "objective": objective,
        "meta": {
            "N_seg": len(segments),
            "N_cand": len(global_candidates),
            "num_steps": len(steps),
        },
    }


def featurize_sample(
    sample: Dict[str, object],
    *,
    sparse_threshold: int = 256,
) -> Dict[str, object]:
    segments_src = sample.get("segments")
    if segments_src is None:
        instance = sample.get("instance")
        if isinstance(instance, dict):
            segments_src = instance.get("segments")
    if segments_src is None:
        raise ValueError("sample missing segment geometry")
    segments = _as_float_segments(segments_src)
    segments.sort(key=lambda seg: seg[0])

    h = float(sample.get("h") or sample.get("instance", {}).get("h"))
    L = float(sample.get("L") or sample.get("instance", {}).get("L"))

    left_segments, right_segments = _split_left_right(segments)
    if not left_segments:
        return _prepare_one_side_sample(sample, segments, h, L, sparse_threshold)
    if not right_segments:
        # Mirror-only case: treat like one-side after reflecting back to origin
        reflected = _reflect_intervals(segments)
        result = _prepare_one_side_sample(sample, reflected, h, L, sparse_threshold)
        return result
    return _prepare_full_line_sample(sample, segments, h, L, sparse_threshold)


def featurize_samples(
    samples: Iterable[Dict[str, object]],
    *,
    sparse_threshold: int = 256,
) -> Iterator[Dict[str, object]]:
    for sample in samples:
        yield featurize_sample(sample, sparse_threshold=sparse_threshold)
