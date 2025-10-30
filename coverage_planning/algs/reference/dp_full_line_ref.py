from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from ..geometry import EPS, find_maximal_p, sort_segments, tour_length
    from .dp_one_side_ref import (
        OneSidePlan,
        dp_one_side_ref,
        dp_one_side_with_plan,
        reconstruct_one_side_plan,
    )
except ImportError:  # pragma: no cover
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from coverage_planning.algs.geometry import EPS, find_maximal_p, sort_segments, tour_length
    from coverage_planning.algs.reference.dp_one_side_ref import (
        OneSidePlan,
        dp_one_side_ref,
        dp_one_side_with_plan,
        reconstruct_one_side_plan,
    )

from coverage_planning.common.constants import TOL_NUM

__all__ = [
    "dp_full_line_ref",
    "dp_full_line_with_plan",
    "dp_one_side_tail_with_plan",
    "reconstruct_tail_plan",
    "TailPlan",
    "FullLinePlanContext",
]


# ---------------------------------------------------------------------------
#  Plan data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TailPlan:
    """Back-pointer data for the right-hand tail DP."""

    next_index: Tuple[int, ...]
    edges: Tuple[Tuple[float, float], ...]
    kinds: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not (len(self.next_index) == len(self.edges) == len(self.kinds)):
            raise ValueError("Tail plan components must share identical length")


@dataclass(frozen=True)
class FullLinePlanContext:
    """Materialised plans for reconstructing alternative solutions."""

    C_left: Tuple[float, ...]
    plan_left: Optional[OneSidePlan]
    C_right: Tuple[float, ...]
    plan_right: Optional[OneSidePlan]
    C_tail: Tuple[float, ...]
    plan_tail: Optional[TailPlan]


def _normalize_tours(tours: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return sorted(((float(p), float(q)) for p, q in tours), key=lambda seg: (seg[0], seg[1]))


def _reflect_left_tours(tours: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    reflected = [(-q, -p) for p, q in tours]
    return _normalize_tours(reflected)


def _compose_no_bridge_tours(context: FullLinePlanContext) -> List[Tuple[float, float]]:
    tours: List[Tuple[float, float]] = []
    if context.plan_left is not None and context.C_left:
        left_tours = reconstruct_one_side_plan(
            context.C_left, context.plan_left, end_index=len(context.C_left) - 1
        )
        tours.extend(_reflect_left_tours(left_tours))
    if context.plan_right is not None and context.C_right:
        right_tours = reconstruct_one_side_plan(
            context.C_right, context.plan_right, end_index=len(context.C_right) - 1
        )
        tours.extend(right_tours)
    return _normalize_tours(tours)


def _compose_bridge_tours(
    context: FullLinePlanContext,
    h: float,
    L: float,
    left_index: int,
    tail_index: int,
    bridge_pair: Tuple[float, float],
) -> List[Tuple[float, float]]:
    tours: List[Tuple[float, float]] = []
    if context.plan_left is not None and context.C_left:
        left_prefix = reconstruct_one_side_plan(context.C_left, context.plan_left, end_index=left_index)
        tours.extend(_reflect_left_tours(left_prefix))
    q = bridge_pair[1]
    p_star = -_find_maximal_p_safe(q, h, L)
    tours.append((p_star, q))
    if context.plan_tail is not None and context.C_tail:
        tail_tours = reconstruct_tail_plan(context.C_tail, context.plan_tail, start_index=tail_index)
        tours.extend(tail_tours)
    return _normalize_tours(tours)

# ---------------------------------------------------------------------------
#  Maximal tour solvers with robust bracketing
# ---------------------------------------------------------------------------
def _find_maximal_p_safe(q: float, h: float, L: float, *, tol: float = 1e-9) -> float:
    reach = 2.0 * math.hypot(q, h)
    if reach > L + EPS:
        raise ValueError("No feasible p: degenerate tour exceeds L")

    candidate = find_maximal_p(q, h, L)
    if candidate <= q + EPS:
        length = tour_length(candidate, q, h)
        if abs(length - L) <= 1e-7:
            return candidate

    hi = q
    f_hi = tour_length(hi, q, h)
    if f_hi > L + 1e-9:
        raise ValueError("No feasible p: numerical inconsistency at hi")

    span = max(1.0, 2.0 * L)
    lo = hi - span
    attempts = 0
    while tour_length(lo, q, h) <= L:
        span *= 2.0
        lo = hi - span
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Failed to bracket maximal p")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = tour_length(mid, q, h)
        if f_mid > L:
            lo = mid
        else:
            hi = mid
        if abs(f_mid - L) <= tol or hi - lo <= tol:
            hi = mid
            break

    f_hi = tour_length(hi, q, h)
    if not (L - 1e-6 <= f_hi <= L + 1e-6):
        raise RuntimeError("Maximal p bisection failed to reach tolerance")
    if hi > q + 1e-9:
        raise RuntimeError("Computed p exceeds q")
    return hi


def _find_maximal_q_safe(p: float, h: float, L: float, *, tol: float = 1e-9) -> float:
    reach = 2.0 * math.hypot(p, h)
    if reach > L + EPS:
        raise ValueError("No feasible q: degenerate tour exceeds L")

    base = math.hypot(p, h)
    K_prime = L - base + p
    if K_prime > 0.0:
        q_candidate = (K_prime * K_prime - h * h) / (2.0 * K_prime)
        if q_candidate >= p - EPS:
            length = tour_length(p, q_candidate, h)
            if abs(length - L) <= 1e-7:
                return q_candidate

    lo = p
    f_lo = tour_length(p, lo, h)
    if f_lo > L + 1e-9:
        raise ValueError("No feasible q: numerical inconsistency at lo")

    hi = max(p + 1.0, 2.0 * L)
    attempts = 0
    while tour_length(p, hi, h) < L - 1e-9:
        hi = p + (hi - p) * 2.0
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Failed to bracket maximal q")

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = tour_length(p, mid, h)
        if f_mid < L:
            lo = mid
        else:
            hi = mid
        if abs(f_mid - L) <= tol or hi - lo <= tol:
            hi = mid
            break

    f_hi = tour_length(p, hi, h)
    if not (L - 1e-6 <= f_hi <= L + 1e-6):
        raise RuntimeError("Maximal q bisection failed to reach tolerance")
    if hi < p - 1e-9:
        raise RuntimeError("Computed q precedes p")
    return hi


# ---------------------------------------------------------------------------
#  Common helpers
# ---------------------------------------------------------------------------
def _validate_full_segments(
    segments: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    segs = sort_segments(segments)
    for idx, (a, b) in enumerate(segs):
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Segment endpoints must be finite")
        if b <= a + EPS:
            raise ValueError("Each segment must have positive length")
        if idx > 0 and segs[idx - 1][1] >= a - EPS:
            raise ValueError("Segments must be pair-wise disjoint with positive gaps")
    return segs


def _split_reflect(
    segments: List[Tuple[float, float]]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    left_ref: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []

    for a, b in segments:
        if b <= 0.0 + EPS:
            l, r = -b, -a
            if r - l > EPS:
                left_ref.append((l, r))
        elif a >= 0.0 - EPS:
            if b - a > EPS:
                right.append((a, b))
        else:
            left_len = -a
            right_len = b
            if left_len > EPS:
                left_ref.append((0.0, left_len))
            if right_len > EPS:
                right.append((0.0, right_len))

    left_ref.sort(key=lambda s: s[0])
    right.sort(key=lambda s: s[0])
    return left_ref, right


def _bisect_segment_index(lefts: List[float], x: float) -> int:
    return bisect.bisect_right(lefts, x + EPS) - 1


def _segment_index_containing(
    lefts: List[float], rights: List[float], x: float
) -> int | None:
    idx = _bisect_segment_index(lefts, x)
    if 0 <= idx < len(lefts) and lefts[idx] - EPS <= x <= rights[idx] + EPS:
        return idx
    return None


def _classify_position_right(
    lefts: List[float], rights: List[float], x: float
) -> Tuple[str, int | None]:
    if x >= rights[-1] - EPS:
        return "beyond", len(rights) - 1
    idx = _bisect_segment_index(lefts, x)
    if 0 <= idx and lefts[idx] - EPS <= x <= rights[idx] + EPS:
        return "inseg", idx
    next_idx = idx + 1
    if 0 <= idx < len(lefts) - 1 and rights[idx] + EPS < x < lefts[next_idx] - EPS:
        return "gap", idx
    if next_idx >= 0 and next_idx < len(lefts) and x < lefts[next_idx] - EPS:
        return "gap", idx
    if idx < 0:
        return "before", None
    return "gap", idx


def _first_segment_right_of(
    lefts: List[float], x: float
) -> int:
    return bisect.bisect_right(lefts, x + EPS)


def _find_candidate_index(
    candidates: List[float],
    value: float,
    tol: float = 1e-8,
) -> int:
    pos = bisect.bisect_left(candidates, value)
    for idx in (pos - 1, pos, pos + 1):
        if 0 <= idx < len(candidates) and abs(candidates[idx] - value) <= tol:
            return idx
    raise KeyError(f"candidate {value:.12f} not found within tolerance")


# ---------------------------------------------------------------------------
#  Tail candidate generation and DP (mirror of Algorithm 3)
# ---------------------------------------------------------------------------
def _generate_tail_candidates(
    segs: List[Tuple[float, float]], h: float, L: float
) -> List[float]:
    lefts = [a for a, _ in segs]
    rights = [b for _, b in segs]

    candidates: set[float] = set(lefts + rights)
    queue = sorted(candidates)
    processed: set[float] = set()

    while queue:
        q = queue.pop()
        if q in processed:
            continue
        processed.add(q)
        try:
            r = _find_maximal_q_safe(q, h, L)
        except ValueError:
            continue

        if r <= lefts[0] + EPS:
            continue

        kind, idx = _classify_position_right(lefts, rights, r)
        if kind == "inseg" and idx is not None:
            if r not in candidates:
                candidates.add(r)
                queue.append(r)
            next_idx = idx + 1
        elif kind == "gap" and idx is not None:
            next_idx = idx + 1
        elif kind == "beyond":
            if rights[-1] not in candidates:
                candidates.add(rights[-1])
                queue.append(rights[-1])
            continue
        else:
            continue

        if next_idx < len(segs):
            next_start = lefts[next_idx]
            if next_start not in candidates:
                candidates.add(next_start)
                queue.append(next_start)

    return sorted(candidates)


def _prefer_tail_transition(
    new_cost: float,
    new_edge: Tuple[float, float],
    new_next: int,
    best_cost: float,
    best_edge: Tuple[float, float] | None,
    best_next: int | None,
    *,
    tol: float,
) -> bool:
    if new_cost < best_cost - tol:
        return True
    if abs(new_cost - best_cost) > tol:
        return False
    if best_edge is None or new_edge < best_edge:
        return True
    if new_edge == best_edge and (best_next is None or new_next < best_next):
        return True
    return False


def _dp_one_side_tail_core(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    debug: Optional[Dict[str, Any]] = None,
    compute_plan: bool,
    tol: float,
) -> Tuple[List[float], List[float], Optional[TailPlan]]:
    if not segments:
        base_tail = [0.0]
        plan = None
        if compute_plan:
            plan = TailPlan(next_index=( -1,), edges=((0.0, 0.0),), kinds=("terminal",))
        return base_tail, [0.0], plan

    segs = _validate_full_segments(segments)
    lefts = [a for a, _ in segs]
    rights = [b for _, b in segs]

    candidates = _generate_tail_candidates(segs, h, L)
    if not candidates:
        raise RuntimeError("Tail candidate generation produced an empty set")

    Tail = [math.inf] * len(candidates)
    Tail_map: Dict[float, float] = {}
    transitions_total = 0

    plan_next: List[int] = []
    plan_edge: List[Tuple[float, float]] = []
    plan_kind: List[str] = []

    for idx in range(len(candidates) - 1, -1, -1):
        q = candidates[idx]
        seg_idx = _segment_index_containing(lefts, rights, q)
        if seg_idx is None:
            seg_idx = _first_segment_right_of(lefts, q)
        try:
            r = _find_maximal_q_safe(q, h, L)
        except ValueError as exc:
            raise ValueError(
                f"Battery insufficient to extend from {q:.6f} on the right side"
            ) from exc

        kind, pos_idx = _classify_position_right(lefts, rights, r)

        best_cost = math.inf
        best_edge: Tuple[float, float] | None = None
        best_next: int | None = None
        best_kind: Optional[str] = None

        if kind == "beyond":
            best_cost = tour_length(q, rights[-1], h)
            best_edge = (q, rights[-1])
            best_next = -1
            best_kind = "tail-end"
        elif kind == "gap" and pos_idx is not None:
            t = pos_idx
            start_j = seg_idx
            if start_j is None:
                start_j = t + 1
            if start_j > t:
                raise ValueError("Gap classification inconsistent with segment ordering")
            continuation_start = lefts[t + 1]
            continuation_idx = _find_candidate_index(candidates, continuation_start)
            continuation_cost = Tail_map[continuation_start]
            for j in range(start_j, t + 1):
                length = tour_length(q, rights[j], h)
                if length > L + EPS:
                    continue
                candidate_cost = continuation_cost + length
                transitions_total += 1
                edge = (q, rights[j])
                if _prefer_tail_transition(
                    candidate_cost,
                    edge,
                    continuation_idx,
                    best_cost,
                    best_edge,
                    best_next,
                    tol=tol,
                ):
                    best_cost = candidate_cost
                    best_edge = edge
                    best_next = continuation_idx
                    best_kind = "tail-gap"
        elif kind == "inseg" and pos_idx is not None:
            t = pos_idx
            tail_idx_r = _find_candidate_index(candidates, r)
            continuation = math.inf
            if candidates[tail_idx_r] > q + EPS:
                continuation = Tail_map[candidates[tail_idx_r]]
            if continuation < math.inf:
                transitions_total += 1
                edge = (q, r)
                cand_cost = L + continuation
                if _prefer_tail_transition(
                    cand_cost,
                    edge,
                    tail_idx_r,
                    best_cost,
                    best_edge,
                    best_next,
                    tol=tol,
                ):
                    best_cost = cand_cost
                    best_edge = edge
                    best_next = tail_idx_r
                    best_kind = "tail-case3"

            if seg_idx is not None and seg_idx <= t - 1:
                for j in range(seg_idx, t):
                    length = tour_length(q, rights[j], h)
                    if length > L + EPS:
                        continue
                    next_start = lefts[j + 1]
                    continuation_idx = _find_candidate_index(candidates, next_start)
                    cont_cost = Tail_map[next_start]
                    cand_cost = cont_cost + length
                    transitions_total += 1
                    edge = (q, rights[j])
                    if _prefer_tail_transition(
                        cand_cost,
                        edge,
                        continuation_idx,
                        best_cost,
                        best_edge,
                        best_next,
                        tol=tol,
                    ):
                        best_cost = cand_cost
                        best_edge = edge
                        best_next = continuation_idx
                        best_kind = "tail-case3"
        else:
            raise ValueError("Unexpected classification for tail maximal tour")

        if math.isinf(best_cost) or best_edge is None:
            raise ValueError(f"No feasible tail transition from {q:.6f}")

        Tail[idx] = best_cost
        Tail_map[q] = best_cost
        if compute_plan:
            plan_next.append(best_next if best_next is not None else -1)
            plan_edge.append(best_edge)
            plan_kind.append(best_kind or "tail")

    if debug is not None:
        debug.clear()
        debug.update(
            {
                "candidate_count": len(candidates),
                "table_size": len(Tail),
                "transitions_total": transitions_total,
                "candidates": list(candidates),
            }
        )

    plan = None
    if compute_plan:
        plan_next.reverse()
        plan_edge.reverse()
        plan_kind.reverse()
        plan = TailPlan(
            next_index=tuple(plan_next),
            edges=tuple((float(p), float(q)) for p, q in plan_edge),
            kinds=tuple(plan_kind),
        )
    return Tail, candidates, plan


def dp_one_side_tail(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float]]:
    Tail, candidates, _ = _dp_one_side_tail_core(
        segments,
        h,
        L,
        debug=debug,
        compute_plan=False,
        tol=TOL_NUM,
    )
    return Tail, candidates


def dp_one_side_tail_with_plan(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    tol: float = TOL_NUM,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], TailPlan]:
    Tail, candidates, plan = _dp_one_side_tail_core(
        segments,
        h,
        L,
        debug=debug,
        compute_plan=True,
        tol=tol,
    )
    if plan is None:
        raise RuntimeError("Tail plan extraction failed")
    return Tail, candidates, plan


def reconstruct_tail_plan(
    candidates: Sequence[float],
    plan: TailPlan,
    *,
    start_index: int,
) -> List[Tuple[float, float]]:
    """Reconstruct tours generated by the tail DP starting at ``start_index``."""
    if not (0 <= start_index < len(candidates)):
        raise IndexError("start_index out of range")
    if len(plan.next_index) != len(candidates):
        raise ValueError("Tail plan mismatch with candidates")

    tours: List[Tuple[float, float]] = []
    idx = start_index
    visited = set()
    while idx != -1:
        if idx in visited:
            raise RuntimeError("Cycle detected in tail plan reconstruction")
        visited.add(idx)
        tours.append(plan.edges[idx])
        idx = plan.next_index[idx]
    return tours


# ---------------------------------------------------------------------------
#  Full-line Algorithm 4
# ---------------------------------------------------------------------------
def _dp_full_line_core(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    debug: Optional[Dict[str, Any]] = None,
    compute_plan: bool,
    tol: float,
) -> Tuple[
    float,
    List[Tuple[float, float]],
    Dict[str, Any],
    Optional[FullLinePlanContext],
    List[Dict[str, Any]],
]:
    segs = _validate_full_segments(segments)
    debug_payload: Dict[str, Any]
    plan_context: Optional[FullLinePlanContext] = None
    solutions: List[Dict[str, Any]] = []

    if not segs:
        debug_payload = {
            "left": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
            "right": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
            "tail": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
            "bridge_checked_pairs": 0,
            "bridge_feasible_pairs": 0,
            "best_mode": "no_bridge",
        }
        if debug is not None:
            debug.clear()
            debug.update(debug_payload)
        return 0.0, [], debug_payload, None, solutions

    left_ref, right = _split_reflect(segs)

    left_debug: Optional[Dict[str, Any]] = {} if debug is not None else None
    right_debug: Optional[Dict[str, Any]] = {} if debug is not None else None
    tail_debug: Optional[Dict[str, Any]] = {} if debug is not None else None

    if not left_ref:
        if compute_plan:
            Sigma_R, C_R, plan_R = dp_one_side_with_plan(right, h, L, tol=tol, debug=right_debug)
            plan_context = FullLinePlanContext(
                C_left=(),
                plan_left=None,
                C_right=tuple(C_R),
                plan_right=plan_R,
                C_tail=(0.0,),
                plan_tail=None,
            )
            tours = _normalize_tours(
                reconstruct_one_side_plan(plan_context.C_right, plan_R, end_index=len(C_R) - 1)
            )
        else:
            Sigma_R, _ = dp_one_side_ref(right, h, L, debug=right_debug)
            tours = []
        debug_payload = {
            "left": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
            "right": right_debug or {},
            "tail": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
            "bridge_checked_pairs": 0,
            "bridge_feasible_pairs": 0,
            "best_mode": "no_bridge",
        }
        if debug is not None:
            debug.clear()
            debug.update(debug_payload)
        return Sigma_R[-1], tours, debug_payload, plan_context, solutions

    if not right:
        if compute_plan:
            Sigma_L, C_L, plan_L = dp_one_side_with_plan(left_ref, h, L, tol=tol, debug=left_debug)
            plan_context = FullLinePlanContext(
                C_left=tuple(C_L),
                plan_left=plan_L,
                C_right=(),
                plan_right=None,
                C_tail=(0.0,),
                plan_tail=None,
            )
            tours = _normalize_tours(
                _reflect_left_tours(
                    reconstruct_one_side_plan(plan_context.C_left, plan_L, end_index=len(C_L) - 1)
                )
            )
        else:
            Sigma_L, _ = dp_one_side_ref(left_ref, h, L, debug=left_debug)
            tours = []
        debug_payload = {
            "left": left_debug or {},
            "right": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
            "tail": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
            "bridge_checked_pairs": 0,
            "bridge_feasible_pairs": 0,
            "best_mode": "no_bridge",
        }
        if debug is not None:
            debug.clear()
            debug.update(debug_payload)
        return Sigma_L[-1], tours, debug_payload, plan_context, solutions

    if compute_plan:
        Sigma_L, C_L, plan_L = dp_one_side_with_plan(left_ref, h, L, tol=tol, debug=left_debug)
        Sigma_R, C_R, plan_R = dp_one_side_with_plan(right, h, L, tol=tol, debug=right_debug)
        Tail_R, C_tail_R, plan_tail = dp_one_side_tail_with_plan(right, h, L, tol=tol, debug=tail_debug)
        plan_context = FullLinePlanContext(
            C_left=tuple(C_L),
            plan_left=plan_L,
            C_right=tuple(C_R),
            plan_right=plan_R,
            C_tail=tuple(C_tail_R),
            plan_tail=plan_tail,
        )
    else:
        Sigma_L, C_L = dp_one_side_ref(left_ref, h, L, debug=left_debug)
        Sigma_R, C_R = dp_one_side_ref(right, h, L, debug=right_debug)
        Tail_R, C_tail_R = dp_one_side_tail(right, h, L, debug=tail_debug)

    Sigma_left_map = {c: Sigma_L[idx] for idx, c in enumerate(C_L)}
    Tail_right_map = {c: Tail_R[idx] for idx, c in enumerate(C_tail_R)}

    best_cost = Sigma_L[-1] + Sigma_R[-1]
    best_mode = "no_bridge"
    best_left_index = len(C_L) - 1 if C_L else -1
    best_tail_index = -1
    best_bridge_pair: Optional[Tuple[float, float]] = None

    bridge_checked_pairs = 0
    bridge_feasible_pairs = 0

    if compute_plan and plan_context is not None:
        tours_no_bridge = _compose_no_bridge_tours(plan_context)
    else:
        tours_no_bridge: List[Tuple[float, float]] = []

    if compute_plan:
        solutions.append(
            {
                "mode": "no_bridge",
                "cost": Sigma_L[-1] + Sigma_R[-1],
                "left_index": best_left_index,
                "tail_index": None,
            }
        )

    left_index_map = {c: idx for idx, c in enumerate(C_L)}
    tail_index_map = {c: idx for idx, c in enumerate(C_tail_R)}

    for P in C_L:
        p = -P
        left_cost = Sigma_left_map[P]
        idx_left = left_index_map[P]
        for q in C_tail_R:
            bridge_checked_pairs += 1
            tour_len = tour_length(p, q, h)
            if tour_len > L + EPS:
                break
            try:
                p_star = _find_maximal_p_safe(q, h, L)
            except ValueError:
                continue
            if abs(p - p_star) > tol:
                continue
            if abs(tour_len - L) > 1e-6:
                continue
            right_cost = Tail_right_map[q]
            total = left_cost + tour_len + right_cost
            bridge_feasible_pairs += 1
            idx_tail = tail_index_map[q]
            if total < best_cost - tol or (
                abs(total - best_cost) <= tol
                and (best_bridge_pair is None or (P, q) < (-(best_bridge_pair[0]), best_bridge_pair[1]))
            ):
                best_cost = total
                best_mode = "bridge"
                best_left_index = idx_left
                best_tail_index = idx_tail
                best_bridge_pair = (-P, q)
            if compute_plan:
                solutions.append(
                    {
                        "mode": "bridge",
                        "cost": total,
                        "left_index": idx_left,
                        "tail_index": idx_tail,
                        "P": P,
                        "q": q,
                    }
                )

    if compute_plan and best_bridge_pair is not None and plan_context is not None:
        best_tours = _compose_bridge_tours(
            plan_context, h, L, best_left_index, best_tail_index, best_bridge_pair
        )
    elif best_mode == "no_bridge":
        best_tours = tours_no_bridge if compute_plan else []
    else:
        best_tours = []

    debug_payload = {
        "left": left_debug or {},
        "right": right_debug or {},
        "tail": tail_debug or {},
        "bridge_checked_pairs": bridge_checked_pairs,
        "bridge_feasible_pairs": bridge_feasible_pairs,
        "best_mode": best_mode,
    }

    if debug is not None:
        debug.clear()
        debug.update(debug_payload)

    return best_cost, best_tours, debug_payload, plan_context, solutions


def dp_full_line_ref(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[Tuple[float, float]]]:
    cost, _tours, debug_payload, _context, _solutions = _dp_full_line_core(
        segments,
        h,
        L,
        debug=debug,
        compute_plan=False,
        tol=TOL_NUM,
    )
    return cost, []


def dp_full_line_with_plan(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    tol: float = TOL_NUM,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[Tuple[float, float]], Dict[str, Any]]:
    cost, tours, debug_payload, context, solutions = _dp_full_line_core(
        segments,
        h,
        L,
        debug=debug,
        compute_plan=True,
        tol=tol,
    )

    meta: Dict[str, Any] = dict(debug_payload)
    if context is not None:
        meta.update(
            {
                "C_left": len(context.C_left),
                "C_right": len(context.C_right),
                "C_tail": len(context.C_tail),
            }
        )
    else:
        meta.update({"C_left": 0, "C_right": 0, "C_tail": 0})

    # Preserve plan materials for downstream consumers (e.g., near-opt search).
    meta["plan_context"] = context
    meta["solutions"] = sorted(solutions, key=lambda rec: (rec["cost"], rec.get("mode", ""), rec.get("left_index", -1), rec.get("tail_index", -1)))[:16]

    return cost, tours, meta


if __name__ == "__main__":
    segs = [(-5.0, -3.0), (-2.0, -1.0), (1.0, 2.0), (3.0, 4.0)]
    cost, tours = dp_full_line_ref(segs, h=2.0, L=50.0)
    assert cost > 0.0

    segs = [(-6.0, -2.5), (2.5, 6.0)]
    cost, _ = dp_full_line_ref(segs, h=1.5, L=25.0)
    assert cost > 0.0
