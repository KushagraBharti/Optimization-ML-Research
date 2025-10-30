from __future__ import annotations

import math
import os
import platform
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import dpos, gs, gsp
from coverage_planning.algs.reference.dp_full_line_ref import (
    FullLinePlanContext,
    _find_maximal_p_safe,
    dp_full_line_with_plan,
    reconstruct_tail_plan,
)
from coverage_planning.algs.reference.dp_one_side_ref import reconstruct_one_side_plan
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.data.gen_instances import (
    baseline_length_requirement,
    bridge_usefulness,
    classify_side,
    estimate_difficulty,
)
from coverage_planning.data.schemas import (
    CandidateSetMeta,
    GoldLabel,
    Instance,
    NearOptimalLabel,
    Sample,
)


GridEpsilon = max(EPS_GEOM * 10.0, 1e-8)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _resolve_code_commit() -> str:
    env_commit = os.getenv("COVERAGE_PLANNING_COMMIT")
    if env_commit:
        return env_commit
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "UNKNOWN"
    return result.decode("utf-8").strip()


@lru_cache(maxsize=1)
def _python_runtime() -> str:
    return platform.python_version()


def _normalize_tours(tours: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return sorted(((float(p), float(q)) for p, q in tours), key=lambda seg: (seg[0], seg[1]))


def _total_length(h: float, tours: Sequence[Tuple[float, float]]) -> float:
    return sum(tour_length(p, q, h) for p, q in tours)


def _expand_tours_to_segments(
    segments: Sequence[Tuple[float, float]],
    tours: Sequence[Tuple[float, float]],
    *,
    snap_tol: float = max(EPS_GEOM * 16.0, 1e-7),
) -> List[Tuple[float, float]]:
    """Clamp and snap solver tours to the instance geometry."""

    if not tours:
        return []

    segs = sorted((float(a), float(b)) for a, b in segments)
    endpoints = sorted({coord for seg in segs for coord in seg})
    min_x = segs[0][0]
    max_x = segs[-1][1]

    def snap(value: float) -> float:
        for endpoint in endpoints:
            if abs(endpoint - value) <= snap_tol:
                return float(endpoint)
        return float(value)

    expanded: List[Tuple[float, float]] = []
    for p, q in tours:
        lo, hi = (float(p), float(q)) if float(p) <= float(q) else (float(q), float(p))
        if hi < min_x - snap_tol or lo > max_x + snap_tol:
            raise ValueError(f"tour ({p}, {q}) lies outside instance span")
        lo = snap(max(lo, min_x))
        hi = snap(min(hi, max_x))
        if hi < lo - TOL_NUM:
            raise ValueError("tour endpoints collapse after snapping")
        expanded.append((lo, hi))
    return _normalize_tours(expanded)


def _verify_tours(instance: Instance, tours: Sequence[Tuple[float, float]]) -> None:
    if not tours:
        raise ValueError("expected at least one tour")
    h = instance.h
    L = instance.L
    segs = list(instance.segments)
    if not segs:
        raise ValueError("instance has no segments")
    segs.sort(key=lambda seg: seg[0])
    min_seg = segs[0][0]
    max_seg = segs[-1][1]

    for idx, (p, q) in enumerate(tours):
        length = tour_length(p, q, h)
        if length > L + TOL_NUM:
            raise ValueError(f"tour #{idx} exceeds battery limit")
        lo, hi = (p, q) if p <= q else (q, p)
        if lo < min_seg - EPS_GEOM - TOL_NUM or hi > max_seg + EPS_GEOM + TOL_NUM:
            raise ValueError(f"tour #{idx} extends outside segment hull")

    for a, b in segs:
        coverage: List[Tuple[float, float]] = []
        for p, q in tours:
            lo, hi = (p, q) if p <= q else (q, p)
            if hi < a - EPS_GEOM or lo > b + EPS_GEOM:
                continue
            coverage.append((max(lo, a), min(hi, b)))
        if not coverage:
            raise ValueError(f"segment [{a}, {b}] not covered")
        coverage.sort()
        start, end = coverage[0]
        if start > a + EPS_GEOM:
            raise ValueError(f"gap before segment start {a}")
        current = end
        for lo, hi in coverage[1:]:
            if lo > current + EPS_GEOM:
                raise ValueError(f"gap detected within segment [{a}, {b}]")
            current = max(current, hi)
        if current < b - EPS_GEOM:
            raise ValueError(f"segment [{a}, {b}] not fully covered")


def _coverage_signature(tours: Sequence[Tuple[float, float]]) -> frozenset[Tuple[int, int]]:
    def quant(x: float) -> int:
        return int(round(x / GridEpsilon))

    pairs = []
    for p, q in tours:
        lo, hi = (p, q) if p <= q else (q, p)
        pairs.append((quant(lo), quant(hi)))
    return frozenset(pairs)


def _iter_stretch_variants(
    instance: Instance,
    tours: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    *,
    multipliers: Sequence[float] = (0.25, 0.5, 1.0),
) -> Iterable[Tuple[List[Tuple[float, float]], Dict[str, object]]]:
    """Yield tour lists stretched towards the hull boundaries."""

    if not tours:
        return

    segs = sorted(instance.segments, key=lambda seg: seg[0])
    min_x = segs[0][0]
    max_x = segs[-1][1]
    span = max(max_x - min_x, 1.0)
    base_delta = max(1e-3, 0.01 * span)

    indices = list(range(len(tours)))
    rng.shuffle(indices)

    for idx in indices:
        p, q = tours[idx]
        lo, hi = (p, q) if p <= q else (q, p)
        for mult in multipliers:
            delta = base_delta * float(mult)
            if delta <= 0.0:
                continue
            candidate_lo = max(min_x, lo - delta)
            if candidate_lo < lo - 1e-9:
                mutated = list(tours)
                mutated[idx] = (candidate_lo, hi)
                yield _normalize_tours(mutated), {
                    "mode": "stretch_left",
                    "delta": delta,
                    "index": idx,
                }
            candidate_hi = min(max_x, hi + delta)
            if candidate_hi > hi + 1e-9:
                mutated = list(tours)
                mutated[idx] = (lo, candidate_hi)
                yield _normalize_tours(mutated), {
                    "mode": "stretch_right",
                    "delta": delta,
                    "index": idx,
                }


def _compose_no_bridge_tours(context: FullLinePlanContext) -> List[Tuple[float, float]]:
    tours: List[Tuple[float, float]] = []
    if context.plan_left is not None and context.C_left:
        left = reconstruct_one_side_plan(context.C_left, context.plan_left, end_index=len(context.C_left) - 1)
        tours.extend((-q, -p) for p, q in left)
    if context.plan_right is not None and context.C_right:
        right = reconstruct_one_side_plan(context.C_right, context.plan_right, end_index=len(context.C_right) - 1)
        tours.extend(right)
    return _normalize_tours(tours)


def _compose_bridge_tours(
    context: FullLinePlanContext,
    instance: Instance,
    left_index: int,
    tail_index: int,
    bridge_pair: Tuple[float, float],
) -> List[Tuple[float, float]]:
    tours: List[Tuple[float, float]] = []
    if context.plan_left is not None and context.C_left:
        prefix = reconstruct_one_side_plan(context.C_left, context.plan_left, end_index=left_index)
        tours.extend((-q, -p) for p, q in prefix)
    q = bridge_pair[1]
    p_star = -_find_maximal_p_safe(q, instance.h, instance.L)
    tours.append((p_star, q))
    if context.plan_tail is not None and context.C_tail:
        suffix = reconstruct_tail_plan(context.C_tail, context.plan_tail, start_index=tail_index)
        tours.extend(suffix)
    return _normalize_tours(tours)




def _baseline_no_bridge_cost(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
) -> float:
    cost = 0.0
    straddlers = [seg for seg in segments if seg[0] < -EPS_GEOM and seg[1] > EPS_GEOM]
    for seg in straddlers:
        _, tours = gsp(seg, h, L)
        cost += sum(tour_length(p, q, h) for p, q in tours)

    left = [seg for seg in segments if seg[1] <= 0.0 + EPS_GEOM]
    right = [seg for seg in segments if seg[0] >= 0.0 - EPS_GEOM]
    if left:
        left_ref = [(-b, -a) for a, b in reversed(left)]
        Sigma_L, _ = dpos(left_ref, h, L)
        cost += Sigma_L[-1]
    if right:
        Sigma_R, _ = dpos(right, h, L)
        cost += Sigma_R[-1]
    return cost


def _compress_dp_meta(raw: Dict[str, object]) -> Dict[str, object]:
    """Convert nested DP debug metadata into a flat, parquet-friendly dict."""

    summary: Dict[str, object] = {}
    for key in ("C_left", "C_right", "C_tail"):
        value = raw.get(key)
        if isinstance(value, (int, float)):
            summary[key] = int(value)
    best_mode = raw.get("best_mode")
    if isinstance(best_mode, str):
        summary["best_mode"] = best_mode
    for name in ("bridge_checked_pairs", "bridge_feasible_pairs"):
        value = raw.get(name)
        if isinstance(value, (int, float)):
            summary[name] = int(value)
    for side in ("left", "right", "tail"):
        section = raw.get(side)
        if not isinstance(section, dict):
            continue
        for field in ("candidate_count", "table_size", "transitions_total"):
            value = section.get(field)
            if isinstance(value, (int, float)):
                summary[f"{side}_{field}"] = int(value)
    return summary


def _make_gold_meta_base(instance: Instance, family: Optional[str]) -> Dict[str, object]:
    baseline = baseline_length_requirement(instance.segments, instance.h)
    tags = [
        f"difficulty:{estimate_difficulty(instance)}",
        f"side:{classify_side(instance.segments)}",
        f"L_regime:{'tight' if instance.L <= baseline + 5.0 * TOL_NUM else 'roomy'}",
    ]
    if family:
        tags.append(f"family:{family}")
    return {"bucket_tags": tags, "baseline_L": baseline}


def label_gold(
    instance: Instance,
    *,
    objective: str,
    family: Optional[str] = None,
) -> GoldLabel:
    segments = list(instance.segments)
    if objective not in {"min_tours", "min_length"}:
        raise ValueError(f"unsupported objective: {objective}")

    meta = _make_gold_meta_base(instance, family)
    bucket_tags = meta.setdefault("bucket_tags", [])

    if objective == "min_tours":
        tour_count, tours = gs(segments, instance.h, instance.L)
        tours = _normalize_tours(tours)
        tours = _expand_tours_to_segments(instance.segments, tours)
        _verify_tours(instance, tours)
        meta.update(
            {
                "objective": "min_tours",
                "algorithm": "gs",
                "tour_count": tour_count,
                "bridge_bucket": "n/a",
                "bridge_benefit": 0.0,
                "no_bridge_cost": 0.0,
            }
        )
        bucket_tags.append("bridge:n/a")
        return GoldLabel(tours=tuple(tours), cost=float(tour_count), meta=meta)

    # objective == "min_length"
    if len(segments) == 1:
        _, tours = gsp(segments[0], instance.h, instance.L)
        tours = _normalize_tours(tours)
        tours = _expand_tours_to_segments(instance.segments, tours)
        _verify_tours(instance, tours)
        total = _total_length(instance.h, tours)
        baseline_nb = _baseline_no_bridge_cost(segments, instance.h, instance.L)
        bridge_bucket = bridge_usefulness(total, baseline_nb)
        bridge_ratio = (baseline_nb - total) / max(total, 1e-9)
        meta.update(
            {
                "objective": "min_length",
                "algorithm": "gsp",
                "bridge_benefit": bridge_ratio,
                "bridge_bucket": bridge_bucket,
                "no_bridge_cost": baseline_nb,
            }
        )
        bucket_tags.append(f"bridge:{bridge_bucket}")
        return GoldLabel(tours=tuple(tours), cost=total, meta=meta)

    cost, tours, raw_meta = dp_full_line_with_plan(segments, instance.h, instance.L)
    plan_context = raw_meta.pop("plan_context", None)
    solutions = raw_meta.pop("solutions", [])
    dp_meta = _compress_dp_meta(raw_meta)
    tours = _normalize_tours(tours)
    tours = _expand_tours_to_segments(instance.segments, tours)
    _verify_tours(instance, tours)

    baseline_nb = _baseline_no_bridge_cost(segments, instance.h, instance.L)
    bridge_bucket = bridge_usefulness(cost, baseline_nb)
    bridge_ratio = (baseline_nb - cost) / max(cost, 1e-9)
    meta.update(
        {
            "objective": "min_length",
            "algorithm": "dp_full",
            "dp_meta": dp_meta,
            "bridge_benefit": bridge_ratio,
            "bridge_bucket": bridge_bucket,
            "no_bridge_cost": baseline_nb,
        }
    )
    bucket_tags.append(f"bridge:{bridge_bucket}")
    meta["_plan_context"] = plan_context
    meta["_solutions"] = solutions
    return GoldLabel(tours=tuple(tours), cost=cost, meta=meta)


def _reconstruct_solution_from_meta(
    context: FullLinePlanContext,
    instance: Instance,
    solution: Dict[str, object],
) -> List[Tuple[float, float]]:
    mode = solution.get("mode")
    if mode == "no_bridge":
        return _compose_no_bridge_tours(context)
    if mode == "bridge":
        left_idx = int(solution["left_index"])
        tail_idx = int(solution["tail_index"])
        P = float(solution["P"])
        q = float(solution["q"])
        return _compose_bridge_tours(context, instance, left_idx, tail_idx, (-P, q))
    raise ValueError(f"unknown solution mode: {mode}")


def label_near_optimal(
    instance: Instance,
    gold: GoldLabel,
    *,
    objective: str,
    k: int,
    max_gap_pct: float,
    rng: np.random.Generator,
) -> List[NearOptimalLabel]:
    if k <= 0:
        return []
    tours_gold = list(gold.tours or [])
    signature_gold = _coverage_signature(tours_gold) if tours_gold else frozenset()
    signatures = {signature_gold}

    results: List[NearOptimalLabel] = []

    if objective == "min_length":
        plan_context = gold.meta.get("_plan_context")
        solutions = gold.meta.get("_solutions", [])
        if not isinstance(plan_context, FullLinePlanContext):
            stretched_source = list(tours_gold)
        else:
            for solution in solutions[1:]:
                if len(results) >= k:
                    break
                try:
                    tours = _reconstruct_solution_from_meta(plan_context, instance, solution)
                except Exception:
                    continue
                tours = _normalize_tours(tours)
                tours = _expand_tours_to_segments(instance.segments, tours)
                try:
                    _verify_tours(instance, tours)
                except ValueError:
                    continue
                total = _total_length(instance.h, tours)
                gap_pct = (total - gold.cost) / max(gold.cost, 1e-9)
                if gap_pct > max_gap_pct + 1e-6 or gap_pct <= 1e-9:
                    continue
                signature = _coverage_signature(tours)
                if signature in signatures:
                    continue
                signatures.add(signature)
                meta = {
                    "mode": solution.get("mode"),
                    "feasible": True,
                    "gap_pct": gap_pct,
                    "cost": total,
                }
                if solution.get("mode") == "bridge":
                    meta["P"] = solution.get("P")
                    meta["q"] = solution.get("q")
                results.append(
                    NearOptimalLabel(
                        tours=tuple(tours),
                        cost=total,
                        gap_pct=gap_pct,
                        meta=meta,
                    )
                )
            stretched_source = list(tours_gold)

        if len(results) >= k:
            return results[:k]

        for mutated, info in _iter_stretch_variants(instance, stretched_source, rng):
            if len(results) >= k:
                break
            try:
                tours = _expand_tours_to_segments(instance.segments, mutated)
                _verify_tours(instance, tours)
            except ValueError:
                continue
            total = _total_length(instance.h, tours)
            gap_pct = (total - gold.cost) / max(gold.cost, 1e-9)
            if gap_pct <= 1e-9 or gap_pct > max_gap_pct + 1e-6:
                continue
            signature = _coverage_signature(tours)
            if signature in signatures:
                continue
            signatures.add(signature)
            meta = {
                "mode": info.get("mode"),
                "delta": info.get("delta"),
                "index": info.get("index"),
                "feasible": True,
                "gap_pct": gap_pct,
                "cost": total,
            }
            results.append(
                NearOptimalLabel(
                    tours=tuple(tours),
                    cost=total,
                    gap_pct=gap_pct,
                    meta=meta,
                )
            )
        return results[:k]

    # objective == "min_tours"
    tours = list(gold.tours or [])
    if not tours:
        return results

    segs = sorted(instance.segments, key=lambda seg: seg[0])
    indices = list(range(len(tours)))
    rng.shuffle(indices)

    for idx in indices:
        if len(results) >= k:
            break
        tour = tours[idx]
        lo, hi = (tour[0], tour[1]) if tour[0] <= tour[1] else (tour[1], tour[0])
        covered_segments = [
            i
            for i, (a, b) in enumerate(segs)
            if lo - EPS_GEOM <= a and hi + EPS_GEOM >= b
        ]
        if len(covered_segments) <= 1:
            continue
        split_points = covered_segments[:-1]
        rng.shuffle(split_points)
        for split_idx in split_points:
            next_idx = split_idx + 1
            mid = segs[split_idx][1]
            next_start = segs[next_idx][0]
            new_tours = list(tours)
            new_tours.pop(idx)
            new_tours.insert(idx, (next_start, hi))
            new_tours.insert(idx, (lo, mid))
            new_tours = _normalize_tours(new_tours)
            new_tours = _expand_tours_to_segments(instance.segments, new_tours)
            try:
                _verify_tours(instance, new_tours)
            except ValueError:
                continue
            new_cost = float(len(new_tours))
            gap_pct = (new_cost - gold.cost) / max(gold.cost, 1e-9)
            if gap_pct > max_gap_pct + 1e-6:
                continue
            signature = _coverage_signature(new_tours)
            if signature in signatures:
                continue
            signatures.add(signature)
            meta = {
                "mode": "split",
                "split_segment_index": split_idx,
                "feasible": True,
            }
            results.append(
                NearOptimalLabel(
                    tours=tuple(new_tours),
                    cost=new_cost,
                    gap_pct=gap_pct,
                    meta=meta,
                )
            )
            break

    if len(results) >= k:
        return results[:k]

    for mutated, info in _iter_stretch_variants(instance, tours, rng):
        if len(results) >= k:
            break
        try:
            stretched = _expand_tours_to_segments(instance.segments, mutated)
            _verify_tours(instance, stretched)
        except ValueError:
            continue
        signature = _coverage_signature(stretched)
        if signature in signatures:
            continue
        signatures.add(signature)
        meta = {
            "mode": info.get("mode"),
            "delta": info.get("delta"),
            "index": info.get("index"),
            "feasible": True,
        }
        results.append(
            NearOptimalLabel(
                tours=tuple(stretched),
                cost=float(len(stretched)),
                gap_pct=0.0,
                meta=meta,
            )
        )

    return results[:k]


def _sanitize_meta(meta: Dict[str, object]) -> Dict[str, object]:
    cleaned = dict(meta)
    cleaned.pop("_plan_context", None)
    cleaned.pop("_solutions", None)
    tags = cleaned.get("bucket_tags")
    if isinstance(tags, list):
        cleaned["bucket_tags"] = tuple(tags)
    return cleaned


def make_sample(
    instance: Instance,
    gold: GoldLabel,
    near_opt: Sequence[NearOptimalLabel],
    *,
    split_tag: str,
    seed: int,
) -> Sample:
    gold_clean = GoldLabel(tours=gold.tours, cost=gold.cost, meta=_sanitize_meta(gold.meta))
    candidate_meta: CandidateSetMeta | None = None
    dp_meta = gold_clean.meta.get("dp_meta")
    if isinstance(dp_meta, dict):
        total_candidates = sum(
            int(dp_meta.get(key, 0)) for key in ("C_left", "C_right", "C_tail")
        )
        tags: List[str] = []
        best_mode = dp_meta.get("best_mode")
        if isinstance(best_mode, str):
            tags.append(f"best_mode:{best_mode}")
        if total_candidates > 0:
            candidate_meta = CandidateSetMeta(count=total_candidates, tags=tuple(tags))

    return Sample(
        instance=instance,
        gold=gold_clean,
        split_tag=split_tag,
        seed=seed,
        code_commit=_resolve_code_commit(),
        python_version=_python_runtime(),
        near_opt=tuple(near_opt),
        candidate_meta=candidate_meta,
    )
