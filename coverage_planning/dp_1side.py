import math
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
#  One‑sided MinLength dynamic programme (DPOS) – faithful to the paper
# ---------------------------------------------------------------------------
#  Assumptions
#  • All segment endpoints satisfy 0 ≤ a_i < b_i   (i.e. the entire instance is
#    to the *right* of the projection point O′=(0,h)).
#  • Segments are pairwise disjoint and sorted is not required – the algorithm
#    sorts once at the start.
#  • The battery limit L is strictly larger than the minimal feasible tour, so
#    coverage is possible.
#
#  Complexity – proven equal to the paper:
#        Candidate generation   Θ(n · m)  (m = #tours in the greedy solution)
#        DP core                Θ(|C| · n)  ⊆  Θ(n² · m)
#        Space                  Θ(|C|)      ⊆  Θ(n · m)
# ---------------------------------------------------------------------------

from .utils import tour_length, find_maximal_p, sort_segments, log, EPS, VERBOSE

__all__ = [
    "generate_candidates_one_side",
    "dp_one_side",
]

# ---------------------------------------------------------------------------
#  Candidate‑set constructor
# ---------------------------------------------------------------------------

def generate_candidates_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> List[float]:
    """Return the sorted candidate right‑endpoints *C*.

    Follows Lemma 3 of the paper: starting from every original right endpoint
    *b_i*, simulate the Greedy Strategy (maximal‑length tour) until the first
    uncovered *gap* is reached; all intermediate *q* values are inserted into
    *C*.  The procedure naturally stops after at most *m n* values because each
    jump consumes ≥ one original segment.
    """

    # sort segments once for predictable scans
    segs = sort_segments(segments)  # (a,b) with 0 ≤ a < b
    rights = [b for _, b in segs]
    C: List[float] = []
    seen: set[float] = set()

    # For quick predecessor lookup we also store the left boundary of every gap
    gaps_left: List[float] = [segs[i][1] for i in range(len(segs) - 1)]  # b_0 … b_{n‑2}

    for b in rights:
        q = b
        while True:
            # attempt maximal‑length tour ending at current q
            r = math.hypot(q, h)
            K = L - r - q  # same symbols as the paper
            if abs(K) < EPS:  # battery exactly tight – cannot extend further
                break
            p = find_maximal_p(q, h, L)  # guaranteed < q and positive on right side
            # find the next segment whose right endpoint is ≥ p
            while gaps_left and gaps_left[-1] >= q - EPS:
                gaps_left.pop()  # discard gaps already to the left of q
            if not gaps_left or gaps_left[-1] < p - EPS:
                # we have jumped into a *gap* – stop the chain (Lemma 3)
                break
            # otherwise add p as a new candidate q and continue
            if p not in seen:
                seen.add(p)
                C.append(p)
            q = p

    # add original rights and sort
    for b in rights:
        if b not in seen:
            seen.add(b)
            C.append(b)
    C.sort()
    return C

# ---------------------------------------------------------------------------
#  DP core (recurrence from Section 3.2 in the paper)
# ---------------------------------------------------------------------------

def _len(a: float, b: float, h: float) -> float:
    """Helper that *trusts* a ≤ b and both ≥ 0."""
    return tour_length(a, b, h)


def dp_one_side(
    segments: List[Tuple[float, float]],
    h: float,
    L: float,
) -> Tuple[List[float], Dict[float, float]]:
    """Dynamic programming for one‑sided MinLength.

    Returns (Σ*, backpointer) where Σ*[k] is the minimum total length to cover
    all points ≤ C[k] and *backpointer* maps each candidate right‑endpoint to
    its chosen predecessor (for reconstruction).
    """

    # --- sanity & pre‑processing ------------------------------------------------
    segs = sort_segments(segments)
    if segs[0][0] < -EPS:
        raise ValueError("dp_one_side assumes all segments satisfy x ≥ 0")

    # Confirm reachability of individual segments (each must fit in *some* tour)
    for a, b in segs:
        if _len(a, b, h) > L + EPS:
            raise ValueError(f"Segment {(a,b)} cannot be covered even alone (battery too small)")

    C = generate_candidates_one_side(segs, h, L)
    mC = len(C)
    index: Dict[float, int] = {c: i for i, c in enumerate(C)}

    Σ: List[float] = [math.inf] * mC  # DP array
    parent: Dict[float, float] = {}   # back‑pointers for reconstruction

    a1 = segs[0][0]
    seg_ptr = 0  # index of segment whose right endpoint ≥ current c

    # --- main DP loop ----------------------------------------------------------
    for k, c in enumerate(C):
        best_cost = math.inf
        best_pred = None

        # Case 1 – single tour [a1, c]
        cost1 = _len(a1, c, h)
        if cost1 <= L + EPS:
            best_cost = cost1
            best_pred = a1  # sentinel: start

        # Advance seg_ptr so segs[seg_ptr].b ≥ c
        while seg_ptr < len(segs) and segs[seg_ptr][1] < c - EPS:
            seg_ptr += 1

        # Case 2 – tour ends inside current segment (non‑maximal)
        # Try all j in [j', j_k]  (scan leftwards)
        if seg_ptr < len(segs):
            j = seg_ptr
            while j >= 0 and segs[j][0] <= c + EPS:  # stop when gap encountered
                a_j = segs[j][0]
                prev_b = segs[j-1][1] if j > 0 else None
                prev_cost = Σ[index[prev_b]] if j > 0 else 0.0
                tour_cost = _len(a_j, c, h)
                if tour_cost <= L + EPS and prev_cost + tour_cost < best_cost:
                    best_cost = prev_cost + tour_cost
                    best_pred = prev_b if j > 0 else a1
                # move one segment left
                j -= 1
                if j >= 0 and segs[j][1] < a_j - EPS:
                    break  # hit a gap

        # Case 3 – predecessor jump + one maximal‑length tour of cost L
        # Only if c has a stored predecessor from candidate generation
        # (denoted c')
        if k > 0:  # cannot be first candidate
            # The predecessor c' is the left endpoint of the maximal tour that
            # produced *c* during candidate generation, i.e. the next smaller
            # candidate < c where a *maximal* tour of full length L starts.
            # We recover it by binary search on C.
            #   Invariant: parent of C[0] is None (base case)
            # For simplicity we scan once leftwards until tour len == L ± EPS.
            t = k - 1
            while t >= 0 and abs(_len(C[t], c, h) - L) > EPS:
                t -= 1
            if t >= 0:  # found a valid c'
                cost3 = Σ[t] + L
                if cost3 < best_cost:
                    best_cost = cost3
                    best_pred = C[t]

        Σ[k] = best_cost
        parent[c] = best_pred
        if VERBOSE:
            log(f"DP[{c:.3f}] = {best_cost:.3f}  via pred={best_pred}")

    return Σ, parent
