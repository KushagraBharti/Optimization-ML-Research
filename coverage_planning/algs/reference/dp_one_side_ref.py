from __future__ import annotations

import bisect
import math
from typing import Dict, List, Tuple

try:
    from ..geometry import EPS, find_maximal_p, tour_length
except ImportError:  # pragma: no cover
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from coverage_planning.algs.geometry import EPS, find_maximal_p, tour_length

__all__ = ["generate_candidates_one_side", "dp_one_side", "generate_candidates_one_side_ref", "dp_one_side_ref"]


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
#  Maximal left-end solver (robust wrapper)
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


# ---------------------------------------------------------------------------
#  Candidate generation (Algorithm 3 closure)
# ---------------------------------------------------------------------------
def _generate_candidates_validated(
    segs: List[Tuple[float, float]], h: float, L: float
) -> List[float]:
    a1 = segs[0][0]
    rights = [b for _, b in segs]
    lefts = [a for a, _ in segs]

    candidates: set[float] = set(rights)

    for q0 in rights:
        q = q0
        visited = 0
        while True:
            p = _find_maximal_p_safe(q, h, L)
            if p <= a1 + EPS:
                break
            kind, idx = _classify_x(lefts, rights, p)
            if kind == "gap":
                break
            if kind == "inseg" and idx is not None:
                candidates.add(p)
                if idx - 1 >= 0:
                    q = rights[idx - 1]
                    candidates.add(q)
                    visited += 1
                    if visited > len(segs):
                        # defensive guard against unexpected cycles
                        break
                    continue
            break

    return sorted(candidates)


def generate_candidates_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    """Generate the candidate set C for Algorithm 3 (exact DPOS)."""
    segs = _validate_segments(segments, h, L)
    return _generate_candidates_validated(segs, h, L)


# ---------------------------------------------------------------------------
#  Dynamic programming (Algorithm 3)
# ---------------------------------------------------------------------------
def dp_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> Tuple[List[float], List[float]]:
    """Run Algorithm 3 (DPOS) on the right half-line.

    Returns (Sigma, C) where Sigma[k] is ?*(C[k]) and C is ascending.
    """
    segs = _validate_segments(segments, h, L)
    candidates = _generate_candidates_validated(segs, h, L)

    lefts = [a for a, _ in segs]
    rights = [b for _, b in segs]
    a1 = lefts[0]

    if not candidates:
        raise RuntimeError("Candidate set generation produced an empty set")

    # Map each segment right endpoint to its candidate index for ?*(b_{j-1}) lookups
    right_candidate_idx = []
    for b in rights:
        idx = _find_candidate_index(candidates, b)
        right_candidate_idx.append(idx)

    Sigma: List[float] = []
    Sigma_map: Dict[float, float] = {}

    for k, c in enumerate(candidates):
        try:
            p = _find_maximal_p_safe(c, h, L)
        except ValueError as exc:
            raise ValueError(
                f"Battery insufficient to touch candidate at {c:.6f}"
            ) from exc

        if p > c + EPS:
            raise RuntimeError("Maximal tour start exceeds its end")

        seg_idx = _segment_index_containing(lefts, rights, c)
        if seg_idx is None:
            raise ValueError(f"Candidate {c:.6f} does not lie within any segment")

        kind, idx = _classify_x(lefts, rights, p)
        if kind == "beyond":
            raise ValueError(f"Maximal start {p:.6f} lies beyond the covered domain")

        if kind == "before":
            value = tour_length(a1, c, h)
            Sigma.append(value)
            Sigma_map[c] = value
            continue

        # Cases 2 and 3 share inner minimisation bounds
        best = math.inf

        if kind == "gap" and idx is not None:
            start = idx + 1
            if start > seg_idx:
                raise ValueError("Gap classification inconsistent with segment ordering")
            for j in range(start, seg_idx + 1):
                length = tour_length(segs[j][0], c, h)
                if length > L + EPS:
                    continue
                prev = 0.0 if j == 0 else Sigma[right_candidate_idx[j - 1]]
                cand = prev + length
                if cand < best:
                    best = cand
            if math.isinf(best):
                raise ValueError(f"No feasible Case 2 transition for candidate {c:.6f}")
        elif kind == "inseg" and idx is not None:
            p_idx = _find_candidate_index(candidates, p)
            if p_idx < k:
                alt = L + Sigma[p_idx]
                if alt < best:
                    best = alt
            # inner minima, starting strictly after the segment containing p
            for j in range(idx + 1, seg_idx + 1):
                length = tour_length(segs[j][0], c, h)
                if length > L + EPS:
                    continue
                prev = 0.0 if j == 0 else Sigma[right_candidate_idx[j - 1]]
                cand = prev + length
                if cand < best:
                    best = cand
            if math.isinf(best):
                raise ValueError(f"No feasible Case 3 transition for candidate {c:.6f}")
        else:
            raise ValueError("Unexpected classification for maximal start position")

        Sigma.append(best)
        Sigma_map[c] = best

    return Sigma, candidates


# Public reference-entry wrappers
def generate_candidates_one_side_ref(segments: List[Tuple[float, float]], h: float, L: float) -> List[float]:
    return generate_candidates_one_side(segments, h, L)


def dp_one_side_ref(segments: List[Tuple[float, float]], h: float, L: float) -> Tuple[List[float], List[float]]:
    return dp_one_side(segments, h, L)


if __name__ == "__main__":
    # Basic smoke tests
    segs = [(0.0, 5.0)]
    Sigma, C = dp_one_side(segs, h=3.0, L=40.0)
    assert len(C) == 1 and abs(Sigma[0] - tour_length(0.0, 5.0, 3.0)) <= 1e-9

    segs = [(0.0, 3.0), (5.0, 7.0)]
    Sigma, C = dp_one_side(segs, h=2.0, L=30.0)
    idx_b2 = _find_candidate_index(C, 7.0)
    assert Sigma[idx_b2] <= tour_length(5.0, 7.0, 2.0) + tour_length(0.0, 3.0, 2.0) + 1e-9

    segs = [(0.0, 4.0), (6.0, 8.5)]
    Sigma, C = dp_one_side(segs, h=1.5, L=18.0)
    assert Sigma[-1] > 0.0
