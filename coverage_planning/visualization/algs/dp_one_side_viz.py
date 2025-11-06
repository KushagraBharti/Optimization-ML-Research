"""Instrumented copy of the one-sided DP (DPOS) solver."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM

EPS = EPS_GEOM

__all__ = [
    "OneSidePlan",
    "generate_candidates_one_side",
    "generate_candidates_one_side_ref",
    "dp_one_side",
    "dp_one_side_ref",
    "dpos_with_plan",
    "reconstruct_one_side_plan",
]


# ---------------------------------------------------------------------------
#  Plan data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OneSidePlan:
    """Back-pointer data allowing deterministic reconstruction of tours."""

    prev_index: Tuple[int, ...]
    kinds: Tuple[str, ...]
    edges: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not (len(self.prev_index) == len(self.kinds) == len(self.edges)):
            raise ValueError("Plan components must share identical length")


# ---------------------------------------------------------------------------
#  Validation and helpers
# ---------------------------------------------------------------------------
def _validate_segments(
    segments: List[Tuple[float, float]], h: float, L: float
) -> List[Tuple[float, float]]:
    if not segments:
        raise ValueError("segments must be non-empty")

    segs = sorted(segments, key=lambda s: s[0])
    for idx, (a, b) in enumerate(segs):
        if a < -EPS:
            raise ValueError("dp_one_side expects x >= 0 (within EPS)")
        if not (a < b - EPS):
            raise ValueError("each segment must satisfy a < b with positive measure")
        if idx > 0 and segs[idx - 1][1] >= a - EPS:
            raise ValueError("segments must be pair-wise disjoint with positive gaps")
        if tour_length(a, b, h) > L + EPS:
            raise ValueError("Unreachable segment under L")
    return segs


def _find_candidate_index(candidates: List[float], value: float, tol: float = 1e-8) -> int:
    pos = bisect.bisect_left(candidates, value)
    for idx in (pos - 1, pos, pos + 1):
        if 0 <= idx < len(candidates) and abs(candidates[idx] - value) <= tol:
            return idx
    raise KeyError(f"candidate {value:.12f} not found within tolerance")


def _segment_index_containing(
    lefts: List[float], rights: List[float], x: float
) -> int | None:
    if x < lefts[0] - EPS or x > rights[-1] + EPS:
        return None
    idx = bisect.bisect_right(lefts, x + EPS) - 1
    if idx >= 0 and idx < len(lefts):
        if lefts[idx] - EPS <= x <= rights[idx] + EPS:
            return idx
    return None


def _classify_x(
    lefts: List[float], rights: List[float], x: float
) -> Tuple[str, int | None]:
    if x <= lefts[0] + EPS:
        return "before", None
    idx = bisect.bisect_right(lefts, x) - 1
    if idx >= 0:
        if lefts[idx] - EPS <= x <= rights[idx] + EPS:
            return "inseg", idx
        next_idx = idx + 1
        if next_idx < len(lefts) and rights[idx] + EPS < x < lefts[next_idx] - EPS:
            return "gap", idx
    return "beyond", None


# ---------------------------------------------------------------------------
#  Maximal tour solvers (robust wrappers)
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
        length = tour_length(mid, q, h)
        if length > L:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break

    length_hi = tour_length(hi, q, h)
    if not (L - 1e-6 <= length_hi <= L + 1e-6):
        raise RuntimeError("Maximal p bisection failed to reach tolerance")
    if hi > q + 1e-9:
        raise RuntimeError("Computed p exceeds q")
    return hi


# ---------------------------------------------------------------------------
#  Transition preference
# ---------------------------------------------------------------------------
def _prefer_transition(
    candidate_cost: float,
    candidate_edge: Tuple[float, float],
    candidate_prev: Optional[int],
    best_cost: float,
    best_edge: Optional[Tuple[float, float]],
    best_prev: Optional[int],
    *,
    tol: float,
) -> bool:
    """Return True if the candidate transition should replace the current best."""
    if candidate_cost + tol < best_cost:
        return True
    if abs(candidate_cost - best_cost) <= tol:
        if candidate_prev is None and best_prev is not None:
            return True
        if candidate_prev is not None and best_prev is not None and candidate_prev < best_prev:
            return True
        if best_edge is None:
            return True
        if candidate_edge < best_edge:
            return True
    return False


# ---------------------------------------------------------------------------
#  Core DP with optional trace capture
# ---------------------------------------------------------------------------
def _dp_one_side_core(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    debug: Optional[Dict[str, Any]] = None,
    compute_plan: bool,
    capture_trace: bool,
    tol: float,
) -> Tuple[List[float], List[float], Optional[OneSidePlan], Optional[Dict[str, Any]]]:
    segs = _validate_segments(segments, h, L)
    candidates = _generate_candidates_validated(segs, h, L)
    transitions_case2 = 0
    transitions_case3 = 0
    transitions_total = 0

    lefts = [a for a, _ in segs]
    rights = [b for _, b in segs]
    a1 = lefts[0]

    if not candidates:
        raise RuntimeError("Candidate set generation produced an empty set")

    right_candidate_idx: List[int] = []
    for b in rights:
        idx = _find_candidate_index(candidates, b)
        right_candidate_idx.append(idx)

    Sigma: List[float] = []
    plan_prev: List[int] = []
    plan_edge: List[Tuple[float, float]] = []
    plan_kind: List[str] = []

    trace_entries: List[Dict[str, Any]] = [] if capture_trace else []

    for k, c in enumerate(candidates):
        try:
            p = _find_maximal_p_safe(c, h, L)
        except ValueError as exc:
            raise ValueError(f"Battery insufficient to touch candidate at {c:.6f}") from exc

        if p > c + EPS:
            raise RuntimeError("Maximal tour start exceeds its end")

        seg_idx = _segment_index_containing(lefts, rights, c)
        if seg_idx is None:
            raise ValueError(f"Candidate {c:.6f} does not lie within any segment")

        kind, idx = _classify_x(lefts, rights, p)
        if kind == "beyond":
            raise ValueError(f"Maximal start {p:.6f} lies beyond the covered domain")

        candidate_trace: Optional[Dict[str, Any]] = None
        if capture_trace:
            candidate_trace = {
                "index": k,
                "candidate": float(c),
                "p_star": float(p),
                "classification": kind,
                "tried_transitions": [],
            }

        if kind == "before":
            value = tour_length(a1, c, h)
            Sigma.append(value)
            if compute_plan:
                plan_prev.append(-1)
                plan_edge.append((a1, c))
                plan_kind.append("case1_before")
            if capture_trace and candidate_trace is not None:
                candidate_trace["picked"] = {"prev_idx": None, "cost": float(value), "case": "case1_before"}
                candidate_trace["backpointer"] = {
                    "prev_idx": None,
                    "edge": (float(a1), float(c)),
                    "kind": "case1_before",
                }
                trace_entries.append(candidate_trace)
            continue

        best_cost = math.inf
        best_prev: Optional[int] = None
        best_edge: Optional[Tuple[float, float]] = None
        best_kind: Optional[str] = None

        if kind == "gap" and idx is not None:
            start = idx + 1
            if start > seg_idx:
                raise ValueError("Gap classification inconsistent with segment ordering")
            for j in range(start, seg_idx + 1):
                length = tour_length(segs[j][0], c, h)
                feasible = length <= L + EPS
                prev_idx = -1 if j == 0 else right_candidate_idx[j - 1]
                prev_cost = 0.0 if j == 0 else Sigma[prev_idx]
                cand_cost = prev_cost + length
                transitions_case2 += 1
                transitions_total += 1
                edge = (segs[j][0], c)
                if capture_trace and candidate_trace is not None:
                    candidate_trace["tried_transitions"].append(
                        {
                            "from_idx": None if prev_idx < 0 else prev_idx,
                            "to_idx": k,
                            "case": "case2_gap",
                            "segment_idx": j,
                            "length": float(length),
                            "feasible": feasible,
                            "cost": float(cand_cost),
                        }
                    )
                if not feasible:
                    continue
                if _prefer_transition(
                    cand_cost,
                    edge,
                    prev_idx if prev_idx >= 0 else None,
                    best_cost,
                    best_edge,
                    best_prev,
                    tol=tol,
                ):
                    best_cost = cand_cost
                    best_prev = prev_idx if prev_idx >= 0 else None
                    best_edge = edge
                    best_kind = "case2_gap"
            if math.isinf(best_cost):
                raise ValueError(f"No feasible Case 2 transition for candidate {c:.6f}")
        elif kind == "inseg" and idx is not None:
            try:
                p_idx = _find_candidate_index(candidates, p)
            except KeyError:
                closest = min(range(len(candidates)), key=lambda i: abs(candidates[i] - p))
                p_idx = closest
            if p_idx < k:
                prev_cost = Sigma[p_idx]
                alt_cost = L + prev_cost
                transitions_case3 += 1
                transitions_total += 1
                edge = (candidates[p_idx], c)
                if capture_trace and candidate_trace is not None:
                    candidate_trace["tried_transitions"].append(
                        {
                            "from_idx": p_idx,
                            "to_idx": k,
                            "case": "case3_inseg",
                            "segment_idx": None,
                            "length": float(L),
                            "feasible": True,
                            "cost": float(alt_cost),
                            "mode": "extend_from_candidate",
                        }
                    )
                if _prefer_transition(
                    alt_cost,
                    edge,
                    p_idx,
                    best_cost,
                    best_edge,
                    best_prev,
                    tol=tol,
                ):
                    best_cost = alt_cost
                    best_prev = p_idx
                    best_edge = edge
                    best_kind = "case3_inseg"
            for j in range(idx + 1, seg_idx + 1):
                length = tour_length(segs[j][0], c, h)
                feasible = length <= L + EPS
                prev_idx = -1 if j == 0 else right_candidate_idx[j - 1]
                prev_cost = 0.0 if j == 0 else Sigma[prev_idx]
                cand_cost = prev_cost + length
                transitions_case3 += 1
                transitions_total += 1
                edge = (segs[j][0], c)
                if capture_trace and candidate_trace is not None:
                    candidate_trace["tried_transitions"].append(
                        {
                            "from_idx": None if prev_idx < 0 else prev_idx,
                            "to_idx": k,
                            "case": "case3_inseg",
                            "segment_idx": j,
                            "length": float(length),
                            "feasible": feasible,
                            "cost": float(cand_cost),
                        }
                    )
                if not feasible:
                    continue
                if _prefer_transition(
                    cand_cost,
                    edge,
                    prev_idx if prev_idx >= 0 else None,
                    best_cost,
                    best_edge,
                    best_prev,
                    tol=tol,
                ):
                    best_cost = cand_cost
                    best_prev = prev_idx if prev_idx >= 0 else None
                    best_edge = edge
                    best_kind = "case3_inseg"
            if math.isinf(best_cost):
                raise ValueError(f"No feasible Case 3 transition for candidate {c:.6f}")
        else:
            raise ValueError("Unexpected classification for maximal start position")

        Sigma.append(best_cost)
        if compute_plan:
            if best_edge is None or best_kind is None:
                raise RuntimeError("Plan transition recording failed")
            plan_prev.append(-1 if best_prev is None else best_prev)
            plan_edge.append(best_edge)
            plan_kind.append(best_kind)

        if capture_trace and candidate_trace is not None:
            candidate_trace["picked"] = {
                "prev_idx": None if best_prev is None else best_prev,
                "cost": float(best_cost),
                "case": best_kind,
            }
            candidate_trace["backpointer"] = {
                "prev_idx": None if best_prev is None else best_prev,
                "edge": (float(best_edge[0]), float(best_edge[1])) if best_edge is not None else None,
                "kind": best_kind,
            }
            trace_entries.append(candidate_trace)

    if debug is not None:
        debug.clear()
        debug.update(
            {
                "candidate_count": len(candidates),
                "transitions_case2": transitions_case2,
                "transitions_case3": transitions_case3,
                "transitions_total": transitions_total,
                "table_size": len(Sigma),
                "candidates": list(candidates),
            }
        )

    plan: Optional[OneSidePlan] = None
    if compute_plan:
        plan = OneSidePlan(
            prev_index=tuple(plan_prev),
            kinds=tuple(plan_kind),
            edges=tuple((float(p0), float(q0)) for p0, q0 in plan_edge),
        )

    trace_payload: Optional[Dict[str, Any]] = None
    if capture_trace:
        trace_payload = {
            "candidates": [float(x) for x in candidates],
            "per_candidate": trace_entries,
        }
        if compute_plan:
            trace_payload["plan_prev"] = list(plan_prev)
            trace_payload["plan_edges"] = [(float(a), float(b)) for a, b in plan_edge]
            trace_payload["plan_kind"] = list(plan_kind)

    return Sigma, candidates, plan, trace_payload


# ---------------------------------------------------------------------------
#  Candidate set generation
# ---------------------------------------------------------------------------
def _generate_candidates_validated(
    segs: List[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    candidates: List[float] = []
    for a, b in segs:
        candidates.append(b)
        try:
            p = _find_maximal_p_safe(b, h, L)
        except ValueError:
            continue
        if p >= a - EPS:
            candidates.append(max(a, p))
    candidates_sorted = sorted(set(float(x) for x in candidates))
    if not candidates_sorted:
        raise RuntimeError("Candidate generation produced an empty set")
    return candidates_sorted


def generate_candidates_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    """Generate the candidate set C for Algorithm 3 (exact DPOS)."""
    segs = _validate_segments(segments, h, L)
    return _generate_candidates_validated(segs, h, L)


def generate_candidates_one_side_ref(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    """Compatibility alias matching the reference API."""
    return generate_candidates_one_side(segments, h, L)


# ---------------------------------------------------------------------------
#  Public DP entry points
# ---------------------------------------------------------------------------
def dp_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float]]:
    Sigma, candidates, _plan, _trace = _dp_one_side_core(
        segments,
        h,
        L,
        debug=debug,
        compute_plan=False,
        capture_trace=False,
        tol=TOL_NUM,
    )
    return Sigma, candidates


def dp_one_side_ref(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float]]:
    return dp_one_side(segments, h, L, debug=debug)


def dpos_with_plan(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    trace: bool = False,
    tol: float = TOL_NUM,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], OneSidePlan] | Tuple[List[float], List[float], OneSidePlan, Dict[str, Any]]:
    """Instrumented variant returning the optimal plan plus optional trace."""
    Sigma, candidates, plan, trace_payload = _dp_one_side_core(
        segments,
        h,
        L,
        debug=debug,
        compute_plan=True,
        capture_trace=trace,
        tol=tol,
    )
    if plan is None:
        raise RuntimeError("Plan extraction failed for one-side DP")
    if trace:
        if trace_payload is None:
            raise RuntimeError("Trace capture failed to produce payload")
        return Sigma, candidates, plan, trace_payload
    return Sigma, candidates, plan


def dp_one_side_with_plan(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    *,
    tol: float = TOL_NUM,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float], OneSidePlan]:
    """Compatibility shim for legacy callers."""
    Sigma, candidates, plan, _trace = _dp_one_side_core(
        segments,
        h,
        L,
        debug=debug,
        compute_plan=True,
        capture_trace=False,
        tol=tol,
    )
    if plan is None:
        raise RuntimeError("Plan extraction failed for one-side DP")
    return Sigma, candidates, plan


# ---------------------------------------------------------------------------
#  Plan reconstruction helpers
# ---------------------------------------------------------------------------
def reconstruct_one_side_plan(
    candidates: Sequence[float],
    plan: OneSidePlan,
    *,
    end_index: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """Reconstruct the tours finishing at ``candidates[end_index]``."""
    if len(candidates) != len(plan.prev_index):
        raise ValueError("Plan length mismatch with candidates")
    if end_index is None:
        end_index = len(candidates) - 1
    if not (0 <= end_index < len(candidates)):
        raise IndexError("end_index out of range")

    tours: List[Tuple[float, float]] = []
    idx = end_index
    visited = set()
    while idx >= 0:
        if idx in visited:
            raise RuntimeError("Cycle detected while reconstructing plan")
        visited.add(idx)
        tours.append(plan.edges[idx])
        prev_idx = plan.prev_index[idx]
        if prev_idx == -1:
            break
        idx = prev_idx
    else:  # pragma: no cover - defensive
        raise RuntimeError("Failed to reach start of plan while reconstructing")

    if plan.prev_index[idx] != -1:
        raise RuntimeError("Plan reconstruction did not terminate at sentinel")

    tours.reverse()
    return [(float(p), float(q)) for p, q in tours]


if __name__ == "__main__":
    segs = [(0.0, 5.0)]
    Sigma, C, plan, trace = dpos_with_plan(segs, h=3.0, L=40.0, trace=True)
    assert len(C) == 1 and abs(Sigma[0] - tour_length(0.0, 5.0, 3.0)) <= 1e-9
    assert trace["per_candidate"][0]["classification"] == "before"

