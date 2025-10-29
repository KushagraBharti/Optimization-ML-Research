from __future__ import annotations

import bisect
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..geometry import EPS, find_maximal_p, sort_segments, tour_length
    from .dp_one_side_ref import dp_one_side_ref
except ImportError:  # pragma: no cover
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from coverage_planning.algs.geometry import EPS, find_maximal_p, sort_segments, tour_length
    from coverage_planning.algs.reference.dp_one_side_ref import dp_one_side_ref

__all__ = ["dp_full_line_ref"]


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


def dp_one_side_tail(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], List[float]]:
    if not segments:
        return [0.0], [0.0]

    segs = _validate_full_segments(segments)
    lefts = [a for a, _ in segs]
    rights = [b for _, b in segs]

    candidates = _generate_tail_candidates(segs, h, L)
    if not candidates:
        raise RuntimeError("Tail candidate generation produced an empty set")

    Tail = [math.inf] * len(candidates)
    Tail_map: Dict[float, float] = {}
    transitions_total = 0

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

        best = math.inf
        if kind == "beyond":
            best = tour_length(q, rights[-1], h)
        elif kind == "gap" and pos_idx is not None:
            t = pos_idx
            start_j = seg_idx
            if start_j is None:
                start_j = t + 1
            if start_j > t:
                raise ValueError("Gap classification inconsistent with segment ordering")
            continuation = Tail_map[lefts[t + 1]]
            for j in range(start_j, t + 1):
                length = tour_length(q, rights[j], h)
                if length > L + EPS:
                    continue
                candidate_cost = continuation + length
                transitions_total += 1
                if candidate_cost < best:
                    best = candidate_cost
        elif kind == "inseg" and pos_idx is not None:
            t = pos_idx
            tail_idx_r = _find_candidate_index(candidates, r)
            if candidates[tail_idx_r] <= q + EPS:
                continuation = math.inf
            else:
                continuation = Tail_map[candidates[tail_idx_r]]
            if continuation < math.inf:
                best = min(best, L + continuation)
                transitions_total += 1

            if seg_idx <= t - 1:
                for j in range(seg_idx, t):
                    length = tour_length(q, rights[j], h)
                    if length > L + EPS:
                        continue
                    next_start = lefts[j + 1]
                    cont_cost = Tail_map[next_start]
                    candidate_cost = cont_cost + length
                    transitions_total += 1
                    if candidate_cost < best:
                        best = candidate_cost
        else:
            raise ValueError("Unexpected classification for tail maximal tour")

        if math.isinf(best):
            raise ValueError(f"No feasible tail transition from {q:.6f}")

        Tail[idx] = best
        Tail_map[q] = best

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

    return Tail, candidates


# ---------------------------------------------------------------------------
#  Full-line Algorithm 4
# ---------------------------------------------------------------------------
def dp_full_line_ref(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
    debug: Optional[Dict[str, Any]] = None,
) -> Tuple[float, List[Tuple[float, float]]]:
    segs = _validate_full_segments(segments)
    if not segs:
        if debug is not None:
            debug.clear()
            debug.update(
                {
                    "left": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
                    "right": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
                    "tail": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
                    "bridge_checked_pairs": 0,
                    "bridge_feasible_pairs": 0,
                }
            )
        return 0.0, []

    left_ref, right = _split_reflect(segs)

    left_debug: Optional[Dict[str, Any]] = {} if debug is not None else None
    right_debug: Optional[Dict[str, Any]] = {} if debug is not None else None
    tail_debug: Optional[Dict[str, Any]] = {} if debug is not None else None

    if not left_ref:
        Sigma_R, _ = dp_one_side_ref(right, h, L, debug=right_debug)
        if debug is not None:
            debug.clear()
            debug.update(
                {
                    "left": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
                    "right": right_debug or {},
                    "tail": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
                    "bridge_checked_pairs": 0,
                    "bridge_feasible_pairs": 0,
                }
            )
        return Sigma_R[-1], []
    if not right:
        Sigma_L, _ = dp_one_side_ref(left_ref, h, L, debug=left_debug)
        if debug is not None:
            debug.clear()
            debug.update(
                {
                    "left": left_debug or {},
                    "right": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
                    "tail": {"candidate_count": 0, "table_size": 0, "transitions_total": 0},
                    "bridge_checked_pairs": 0,
                    "bridge_feasible_pairs": 0,
                }
            )
        return Sigma_L[-1], []

    Sigma_L, C_L = dp_one_side_ref(left_ref, h, L, debug=left_debug)
    Sigma_R, _ = dp_one_side_ref(right, h, L, debug=right_debug)
    Tail_R, C_tail_R = dp_one_side_tail(right, h, L, debug=tail_debug)

    Sigma_left_map = {c: Sigma_L[idx] for idx, c in enumerate(C_L)}
    Tail_right_map = {c: Tail_R[idx] for idx, c in enumerate(C_tail_R)}

    best_cost = Sigma_L[-1] + Sigma_R[-1]
    best_arg: Tuple[str, float | None, float | None] = ("no_bridge", None, None)

    bridge_checked_pairs = 0
    bridge_feasible_pairs = 0

    for P in C_L:
        p = -P
        left_cost = Sigma_left_map[P]
        for q in C_tail_R:
            bridge_checked_pairs += 1
            tour_len = tour_length(p, q, h)
            if tour_len > L + EPS:
                break
            try:
                p_star = _find_maximal_p_safe(q, h, L)
            except ValueError:
                continue
            if abs(p - p_star) > 1e-7:
                continue
            if abs(tour_len - L) > 1e-6:
                continue
            right_cost = Tail_right_map[q]
            total = left_cost + tour_len + right_cost
            bridge_feasible_pairs += 1
            if total < best_cost:
                best_cost = total
                best_arg = ("bridge", P, q)

    if debug is not None:
        debug.clear()
        debug.update(
            {
                "left": left_debug or {},
                "right": right_debug or {},
                "tail": tail_debug or {},
                "bridge_checked_pairs": bridge_checked_pairs,
                "bridge_feasible_pairs": bridge_feasible_pairs,
                "best_mode": best_arg[0],
            }
        )

    return best_cost, []


if __name__ == "__main__":
    segs = [(-5.0, -3.0), (-2.0, -1.0), (1.0, 2.0), (3.0, 4.0)]
    cost, tours = dp_full_line_ref(segs, h=2.0, L=50.0)
    assert cost > 0.0

    segs = [(-6.0, -2.5), (2.5, 6.0)]
    cost, _ = dp_full_line_ref(segs, h=1.5, L=25.0)
    assert cost > 0.0
