from __future__ import annotations
from typing import List, Tuple, Dict, Any
from time import perf_counter

from .schemas import *
from .solvers import SolverProvider
from coverage_planning.algs.geometry import tour_length, EPS

def _split_reflect(segments: List[Tuple[float, float]]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Mirror left-hand segments to the right half-plane, like dp_full_line does."""
    left_ref, right = [], []
    for a, b in segments:
        if b <= 0:
            left_ref.append((-b, -a))
        elif a >= 0:
            right.append((a, b))
        else:  # straddles 0 → split at 0
            left_ref.append((0.0, -a))
            right.append((0.0, b))
    # remove degenerate zeros
    left_ref = [(l, r) for (l, r) in left_ref if r - l > EPS]
    right    = [(l, r) for (l, r) in right    if r - l > EPS]
    left_ref.sort(key=lambda s: s[0])
    right.sort(key=lambda s: s[0])
    return left_ref, right

def _ensure_L_for_one_side(segs: List[Tuple[float, float]], h: float, L: float) -> tuple[float, dict]:
    """
    Heuristic dp_one_side raises if any single segment has tour_length(a,b,h) > L.
    Make L large enough to pass that guard (and >= 2h).
    """
    need = max([2.05 * h] + [tour_length(a, b, h) for (a, b) in segs]) if segs else 2.05 * h
    L2 = max(L, need + 1e-6)
    notes = {}
    if L2 > L + 1e-12:
        notes["L_adjusted_from"] = L
        notes["L_adjusted_to"] = L2
    return L2, notes

def _disjointify(segs: List[Tuple[float, float]], min_gap: float = 1e-4) -> List[Tuple[float, float]]:
    """Ensure pair-wise disjoint with a minimum positive gap (defensive)."""
    out: List[Tuple[float, float]] = []
    for a, b in sorted(segs, key=lambda s: s[0]):
        if out and a <= out[-1][1] + min_gap:
            shift = out[-1][1] + min_gap - a
            a += shift
            b += shift
        if b - a < EPS:
            b = a + EPS
        out.append((a, b))
    return out

def label_min_tours(inst: Instance, provider: SolverProvider) -> LabeledExample:
    # Defensive: enforce positive gaps for Greedy MinTours precondition
    segs = _disjointify(inst.segments, min_gap=1e-4)

    # Ensure L is not too tight for per-segment feasibility
    need = max([2.05 * inst.h] + [tour_length(a, b, inst.h) for (a, b) in segs]) if segs else 2.05 * inst.h
    L2 = max(inst.L, need + 1e-6)

    # Watchdog: if a single instance is slow, bump L a bit and retry once
    import time as _t
    t0 = perf_counter()
    count, tours = provider.min_tours(segs, inst.h, L2)
    if _t.perf_counter() - t0 > 0.5:  # >0.5s for one instance is suspicious for Greedy
        # bump L modestly and retry once
        L2b = L2 * 1.05
        t1 = perf_counter()
        count2, tours2 = provider.min_tours(segs, inst.h, L2b)
        dt2 = perf_counter() - t1
        if dt2 < ( _t.perf_counter() - t0 ):
            count, tours, L2 = count2, tours2, L2b

    dt = perf_counter() - t0
    notes = {"runtime_s": dt}
    if L2 > inst.L + 1e-12:
        notes.update({"L_adjusted_from": inst.L, "L_adjusted_to": L2})

    return LabeledExample(
        instance=Instance(segments=segs, h=inst.h, L=L2, seed=inst.seed, meta=inst.meta),
        objective="MinTours",
        provenance=provider.family,
        label=LabelMinTours(tours=tours, count=count, total_cost=float(count)),
        notes=notes,
    )

def label_min_length_one_side(inst: Instance, provider: SolverProvider) -> LabeledExample:
    # ensure we truly pass only right-half segments
    segs = [(a, b) for (a, b) in inst.segments if a >= -EPS and b >= -EPS]
    if not segs:
        # trivial empty-case: nothing to cover
        return LabeledExample(
            instance=inst,
            objective="MinLength_OneSide",
            provenance=provider.family,
            label=LabelMinLengthOneSide(candidates=[], prefix_costs=[0.0], suffix_costs=[0.0], optimal_total=0.0),
            notes={"runtime_s": 0.0, "C_size": 0, "empty_instance": True}
        )

    L2, adjust_notes = _ensure_L_for_one_side(segs, inst.h, inst.L)
    t0 = perf_counter()
    prefix, suffix, C = provider.dp_one_side(segs, inst.h, L2)
    dt = perf_counter() - t0

    notes = {"runtime_s": dt, "C_size": len(C)}
    notes.update(adjust_notes)
    return LabeledExample(
        instance=Instance(segments=segs, h=inst.h, L=L2, seed=inst.seed, meta=inst.meta),
        objective="MinLength_OneSide",
        provenance=provider.family,
        label=LabelMinLengthOneSide(
            candidates=C,
            prefix_costs=prefix,
            suffix_costs=suffix,
            optimal_total=prefix[-1],
        ),
        notes=notes
    )

def label_min_length_full_line(inst: Instance, provider: SolverProvider) -> LabeledExample:
    # Split into sides (mirror left to right) just like the solver does.
    left_ref, right = _split_reflect(inst.segments)

    # First, ensure L is big enough to pass the one-sided heuristic guard on each side.
    need_left  = max([2.05 * inst.h] + [tour_length(a, b, inst.h) for (a, b) in left_ref]) if left_ref else 2.05 * inst.h
    need_right = max([2.05 * inst.h] + [tour_length(a, b, inst.h) for (a, b) in right])    if right    else 2.05 * inst.h
    L2 = max(inst.L, need_left + 1e-6, need_right + 1e-6)

    # Precompute candidate sets at L2 (heuristic one-sided DP).
    C_l, C_r = [], []
    if left_ref:
        pref_L, _suf_L, C_l = provider.dp_one_side(left_ref, inst.h, L2)
    if right:
        pref_R, _suf_R, C_r = provider.dp_one_side(right, inst.h, L2)

    adjust_notes: Dict[str, float] = {}
    # Robustness: if both sides exist, ensure that for EVERY p in C_l
    # there is at least one q in C_r with len(p,q) <= L2.
    # Compute L_bridge = max_p min_q len(p,q,h).
    if C_l and C_r:
        # per-p minimal bridge length
        per_p_min = [min(tour_length(p, q, inst.h) for q in C_r) for p in C_l]
        L_bridge = max(per_p_min)  # the worst p’s best possible bridge
        if L2 + 1e-12 < L_bridge:
            # bump L2 so every p has at least one feasible q
            L2 = L_bridge + 1e-6
            adjust_notes.update({
                "L_adjusted_from": inst.L,
                "L_adjusted_to": L2,
                "need_left": need_left,
                "need_right": need_right,
                "L_bridge_needed": L_bridge
            })

    # Now safe to call the heuristic full-line DP without empty-sequence max().
    t0 = perf_counter()
    cost = provider.dp_full_line(inst.segments, inst.h, L2)
    dt = perf_counter() - t0

    notes = {"runtime_s": dt}
    notes.update(adjust_notes)
    return LabeledExample(
        instance=Instance(segments=inst.segments, h=inst.h, L=L2, seed=inst.seed, meta=inst.meta),
        objective="MinLength_FullLine",
        provenance=provider.family,
        label=LabelMinLengthFullLine(optimal_total=cost),
        notes=notes
    )


