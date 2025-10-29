from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Tuple

from coverage_planning.algs.geometry import tour_length
from coverage_planning.common.constants import EPS_GEOM

from .schemas import CandidateSetMeta, GoldLabel, Instance, Sample
from .solvers import SolverProvider


def _split_reflect(
    segments: List[Tuple[float, float]]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Mirror left-hand segments to the right half-plane, mirroring dp_full_line."""
    left_ref: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []
    for a, b in segments:
        if b <= 0.0:
            left_ref.append((-b, -a))
        elif a >= 0.0:
            right.append((a, b))
        else:
            left_ref.append((0.0, -a))
            right.append((0.0, b))
    left_ref = [(l, r) for (l, r) in left_ref if r - l > EPS_GEOM]
    right = [(l, r) for (l, r) in right if r - l > EPS_GEOM]
    left_ref.sort(key=lambda seg: seg[0])
    right.sort(key=lambda seg: seg[0])
    return left_ref, right


def _ensure_L_for_one_side(
    segs: List[Tuple[float, float]], h: float, L: float
) -> Tuple[float, Dict[str, float]]:
    if not segs:
        return max(L, 2.05 * h), {}
    need = max([2.05 * h] + [tour_length(a, b, h) for (a, b) in segs])
    L2 = max(L, need + 1e-6)
    notes: Dict[str, float] = {}
    if L2 > L:
        notes["L_adjusted_from"] = L
        notes["L_adjusted_to"] = L2
    return L2, notes


def _disjointify(
    segs: List[Tuple[float, float]], *, min_gap: float = 1e-4
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(segs, key=lambda seg: seg[0]):
        if out and a <= out[-1][1] + min_gap:
            shift = out[-1][1] + min_gap - a
            a += shift
            b += shift
        if b - a < EPS_GEOM:
            b = a + EPS_GEOM
        out.append((a, b))
    return out


def label_min_tours(
    inst: Instance, provider: SolverProvider, *, split_tag: str = "unspecified", seed: int = 0
) -> Sample:
    segs = _disjointify(list(inst.segments))
    need = max([2.05 * inst.h] + [tour_length(a, b, inst.h) for (a, b) in segs]) if segs else 2.05 * inst.h
    L2 = max(inst.L, need + 1e-6)

    t0 = perf_counter()
    count, tours = provider.min_tours(segs, inst.h, L2)
    runtime = perf_counter() - t0

    gold = GoldLabel(
        tours=tuple(tours),
        cost=float(count),
        meta={
            "tour_count": count,
            "runtime_s": runtime,
            **({"L_adjusted_from": inst.L, "L_adjusted_to": L2} if L2 > inst.L else {}),
        },
    )
    adjusted_inst = Instance(tuple(segs), inst.h, L2)
    return Sample(instance=adjusted_inst, gold=gold, split_tag=split_tag, seed=seed)


def label_min_length_one_side(
    inst: Instance, provider: SolverProvider, *, split_tag: str = "unspecified", seed: int = 0
) -> Sample:
    segs = [(a, b) for (a, b) in inst.segments if a >= -EPS_GEOM and b >= -EPS_GEOM]
    if not segs:
        gold = GoldLabel(tours=None, cost=0.0, meta={"runtime_s": 0.0, "empty_instance": True})
        return Sample(instance=inst, gold=gold, split_tag=split_tag, seed=seed)

    L2, adjust_notes = _ensure_L_for_one_side(segs, inst.h, inst.L)
    t0 = perf_counter()
    prefix, suffix, candidates = provider.dp_one_side(segs, inst.h, L2)
    runtime = perf_counter() - t0

    gold = GoldLabel(
        tours=None,
        cost=float(prefix[-1]),
        meta={
            "prefix_costs": prefix,
            "suffix_costs": suffix,
            "candidates": candidates,
            "runtime_s": runtime,
            **adjust_notes,
        },
    )
    adjusted_inst = Instance(tuple(segs), inst.h, L2)
    candidate_meta = CandidateSetMeta(count=len(candidates), tags=("one_side",))
    return Sample(
        instance=adjusted_inst,
        gold=gold,
        candidate_meta=candidate_meta,
        split_tag=split_tag,
        seed=seed,
    )


def label_min_length_full_line(
    inst: Instance, provider: SolverProvider, *, split_tag: str = "unspecified", seed: int = 0
) -> Sample:
    left_ref, right = _split_reflect(list(inst.segments))

    need_left = max([2.05 * inst.h] + [tour_length(a, b, inst.h) for (a, b) in left_ref]) if left_ref else 2.05 * inst.h
    need_right = max([2.05 * inst.h] + [tour_length(a, b, inst.h) for (a, b) in right]) if right else 2.05 * inst.h
    L2 = max(inst.L, need_left + 1e-6, need_right + 1e-6)

    adjust_notes: Dict[str, float] = {}
    C_l: List[float] = []
    C_r: List[float] = []

    if left_ref:
        pref_L, _suf_L, C_l = provider.dp_one_side(left_ref, inst.h, L2)
        adjust_notes["C_left"] = len(C_l)
    if right:
        pref_R, _suf_R, C_r = provider.dp_one_side(right, inst.h, L2)
        adjust_notes["C_right"] = len(C_r)

    if C_l and C_r:
        per_p_min = [min(tour_length(p, q, inst.h) for q in C_r) for p in C_l]
        L_bridge = max(per_p_min)
        if L2 + EPS_GEOM < L_bridge:
            adjust_notes.update({"L_bridge_needed": L_bridge, "L_adjusted_from": L2})
            L2 = L_bridge + 1e-6
            adjust_notes["L_adjusted_to"] = L2

    t0 = perf_counter()
    cost = provider.dp_full_line(inst.segments, inst.h, L2)
    runtime = perf_counter() - t0

    gold = GoldLabel(
        tours=None,
        cost=float(cost),
        meta={"runtime_s": runtime, **adjust_notes},
    )
    adjusted_inst = Instance(inst.segments, inst.h, L2)
    candidate_meta = None
    total_candidates = int(adjust_notes.get("C_left", 0)) + int(adjust_notes.get("C_right", 0))
    if total_candidates:
        candidate_meta = CandidateSetMeta(count=total_candidates, tags=("full_line",))

    return Sample(
        instance=adjusted_inst,
        gold=gold,
        candidate_meta=candidate_meta,
        split_tag=split_tag,
        seed=seed,
    )

