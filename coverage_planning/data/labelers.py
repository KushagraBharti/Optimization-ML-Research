"""Label generation utilities wired to the reference solver stack.

The helpers below provide the narrow surface consumed by dataset generators.
They intentionally funnel every call through the paper-faithful reference
solvers so downstream training code cannot accidentally depend on heuristic
implementations.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from coverage_planning.algs.reference import (
    dp_full_with_plan,
    dp_one_side_with_plan,
    reconstruct_one_side_plan,
)
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.data.schemas import (
    CandidateSetMeta,
    GoldLabel,
    Instance,
    NearOptimalLabel,
    Sample,
)

OneDTour = Tuple[float, float]

__all__ = ["label_gold", "label_near_optimal", "make_sample"]


def _categorise_segments(segments: Sequence[Tuple[float, float]]) -> str:
    has_negative = any(b <= -EPS_GEOM for a, b in segments)
    has_positive = any(a >= EPS_GEOM for a, b in segments)
    if has_negative and not has_positive:
        return "left"
    if has_positive and not has_negative:
        return "right"
    return "full"


def _reflect_segments(segments: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    reflected = [(-b, -a) for a, b in segments]
    reflected.sort(key=lambda seg: seg[0])
    return reflected


def _reflect_tours_to_physical(tours: Sequence[OneDTour]) -> List[OneDTour]:
    return [(-q, -p) for p, q in tours]


def _compute_bucket_tags(mode: str, meta: Dict[str, object]) -> List[str]:
    tags = [mode]
    best_mode = meta.get("best_mode")
    if isinstance(best_mode, str):
        tags.append(f"full:{best_mode}")
    return tags


def _candidate_meta_from_counts(counts: Dict[str, int]) -> CandidateSetMeta | None:
    total = counts.get("total", 0)
    if total <= 0:
        return None
    tags = tuple(sorted(k for k, v in counts.items() if k != "total" and v > 0))
    return CandidateSetMeta(count=total, tags=tags)


def label_gold(
    instance: Instance,
    *,
    objective: str = "min_length",
    family: str | None = None,
) -> GoldLabel:
    """Generate a gold label for ``instance`` using the reference solvers."""

    if objective != "min_length":
        raise ValueError(f"Unsupported objective: {objective!r}")

    segments = list(instance.segments)
    h = float(instance.h)
    L = float(instance.L)
    mode = _categorise_segments(segments)

    if mode == "right":
        Sigma, candidates, plan = dp_one_side_with_plan(segments, h, L, tol=TOL_NUM)
        tours = reconstruct_one_side_plan(candidates, plan)
        cost = float(Sigma[-1])
        dp_meta: Dict[str, object] = {
            "mode": "one_side_right",
            "candidate_count": len(candidates),
            "C_right": len(candidates),
            "C_left": 0,
            "C_tail": 0,
        }
        bucket_tags = ["one_side_right"]
    elif mode == "left":
        segments_ref = _reflect_segments(segments)
        Sigma, candidates, plan = dp_one_side_with_plan(segments_ref, h, L, tol=TOL_NUM)
        tours_ref = reconstruct_one_side_plan(candidates, plan)
        tours = _reflect_tours_to_physical(tours_ref)
        cost = float(Sigma[-1])
        dp_meta = {
            "mode": "one_side_left",
            "candidate_count": len(candidates),
            "C_right": 0,
            "C_left": len(candidates),
            "C_tail": 0,
        }
        bucket_tags = ["one_side_left"]
    else:
        cost, tours, meta = dp_full_with_plan(segments, h, L)
        context = meta.get("plan_context")
        c_left = len(getattr(context, "C_left", ())) if context is not None else 0
        c_right = len(getattr(context, "C_right", ())) if context is not None else 0
        c_tail = len(getattr(context, "C_tail", ())) if context is not None else 0
        dp_meta = {
            "mode": "full_line",
            "candidate_count": c_left + c_right + c_tail,
            "C_left": c_left,
            "C_right": c_right,
            "C_tail": c_tail,
            "best_mode": meta.get("best_mode"),
            "bridge_checked_pairs": meta.get("bridge_checked_pairs"),
            "bridge_feasible_pairs": meta.get("bridge_feasible_pairs"),
        }
        bucket_tags = _compute_bucket_tags("full_line", dp_meta)

    gold_meta: Dict[str, object] = {
        "dp_meta": dp_meta,
        "bucket_tags": bucket_tags,
    }
    if family is not None:
        gold_meta["family"] = family

    tours_tuple = tuple((float(p), float(q)) for p, q in tours)
    return GoldLabel(tours=tours_tuple, cost=float(cost), meta=gold_meta)


def label_near_optimal(
    instance: Instance,
    gold: GoldLabel,
    *,
    objective: str = "min_length",
    k: int = 0,
    max_gap_pct: float = 0.0,
    rng=None,
) -> Tuple[NearOptimalLabel, ...]:
    """Return a (possibly empty) tuple of near-optimal labels.

    For milestone 3.5 we keep this conservative and return an empty tuple.  The
    hook remains so downstream code has a uniform entry point once alternate
    policy sampling is implemented.
    """

    return tuple()


def make_sample(
    instance: Instance,
    gold: GoldLabel,
    near_opt: Sequence[NearOptimalLabel] = (),
    *,
    split_tag: str,
    seed: int,
) -> Sample:
    """Assemble a :class:`~coverage_planning.data.schemas.Sample` record."""

    dp_meta = gold.meta.get("dp_meta", {}) if isinstance(gold.meta, dict) else {}
    counts = {
        "total": int(dp_meta.get("candidate_count", 0)),
        "left": int(dp_meta.get("C_left", 0)),
        "right": int(dp_meta.get("C_right", 0)),
        "tail": int(dp_meta.get("C_tail", 0)),
    }
    candidate_meta = _candidate_meta_from_counts(counts)

    return Sample(
        instance=instance,
        gold=gold,
        near_opt=tuple(near_opt),
        split_tag=split_tag,
        seed=seed,
        candidate_meta=candidate_meta,
    )
