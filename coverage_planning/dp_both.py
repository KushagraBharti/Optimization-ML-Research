import math
from typing import List, Tuple, Dict

from .utils import tour_length, log, EPS
from .dp_1side import dp_one_side, generate_candidates_one_side

__all__ = [
    "dp_full_line",
]

# ---------------------------------------------------------------------------
#  Min‑Length on both sides of the projection point O′ (general case).
#  This is Algorithm 4 from the paper – rebuilt *faithfully*.
# ---------------------------------------------------------------------------
#  High‑level outline
#  •   Split the instance at x = 0.  A segment that straddles 0 is split into
#      (a,0) on the left and (0,b) on the right; a zero‑length fragment is
#      discarded.
#  •   Reflect the left side to x ≥ 0 so we can reuse the one‑sided DP.
#  •   Run the one‑sided DP twice on each half: once forward to obtain the
#      prefix‑cost table Σ_left / Σ_right, and once in a mirrored coordinate
#      system to obtain the suffix‑cost table 𝚺̃_left / 𝚺̃_right.
#      (The paper calls these the “top–down” variants.)
#  •   Cost without a bridge is Σ_left(∞) + Σ_right(∞).
#  •   Otherwise, exactly one *maximal* bridge‑tour crosses O′.  For every
#      candidate pair (p,q) that can serve as that bridge we evaluate
#         Σ_left(p) + len(p,q) + 𝚺̃_right(q)
#      and keep the minimum.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _split_and_reflect(segments: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Return (left_reflected, right) after splitting straddling segments.

    • *left_reflected* lives entirely in x ≥ 0 and is obtained by the mapping
      (a,b)  ↦  (−b, −a)  for every original piece that lies in x ≤ 0.
    • *right* keeps its original coordinates (a,b) with a ≥ 0.
    """
    left_reflected: List[Tuple[float, float]] = []
    right:          List[Tuple[float, float]] = []

    for a, b in segments:
        if b <= 0.0:                      # fully left
            left_reflected.append((-b, -a))
        elif a >= 0.0:                    # fully right
            right.append((a, b))
        else:                             # straddles 0 → split
            if a < 0.0:
                left_reflected.append((0.0 - b, 0.0 - a))  # (−b, −0) ≡ (−b,0)
            if b > 0.0:
                right.append((0.0, b))

    # normalise: discard zero‑length fragments
    left_reflected = [(l, r) for (l, r) in left_reflected if r - l > EPS]
    right          = [(l, r) for (l, r) in right          if r - l > EPS]

    # sort by left endpoint (needed by dp_one_side)
    left_reflected.sort(key=lambda seg: seg[0])
    right.sort(key=lambda seg: seg[0])
    return left_reflected, right


def _suffix_dp_right(segments_right: List[Tuple[float, float]], h: float, L: float) -> Dict[float, float]:
    """Compute Σ̃_right(q): min cost to cover *from* q to the far right.

    We mirror the right‑side instance around its rightmost endpoint *B* so that
    it becomes a left‑aligned instance and reuse the one‑sided DP.  The mapping
        x  ↦  B − x
    sends the interval [q,B] to [0, B−q].
    """
    if not segments_right:
        return {}

    B = segments_right[-1][1]  # far right endpoint

    segs_mirror = [(B - b, B - a) for (a, b) in segments_right]
    segs_mirror.sort(key=lambda s: s[0])

    C_mirror, dp_mirror = dp_one_side(segs_mirror, h, L, side_label="RIGHT‑SUFFIX")

    # Build q ↦ suffix‑cost dictionary
    suffix: Dict[float, float] = {}
    for c_m, cost in zip(C_mirror, dp_mirror):
        q = B - c_m
        suffix[q] = cost
    return suffix


# ---------------------------------------------------------------------------
#  Main procedure
# ---------------------------------------------------------------------------

def dp_full_line(segments: List[Tuple[float, float]], h: float, L: float) -> float:
    """Return the *exact* minimum total flight length for covering *segments*.

    Implements Algorithm 4 (both‑side dynamic programme) from the paper, with
    the same variable names where possible.
    """

    # ----------------  Phase 0 – Pre‑processing  ---------------------------
    left_ref, right = _split_and_reflect(segments)

    log("Left (reflected):", left_ref)
    log("Right:", right, "\n")

    # Quick exits when the instance is one‑sided
    if not left_ref:
        _, dp_right = dp_one_side(right, h, L, side_label="RIGHT‑ONLY")
        return dp_right[-1]
    if not right:
        _, dp_left = dp_one_side(left_ref, h, L, side_label="LEFT‑ONLY")
        return dp_left[-1]

    # ----------------  Phase 1 – Independent one‑sided DPs  ---------------
    C_l, dp_l_prefix = dp_one_side(left_ref, h, L, side_label="LEFT")
    C_r, dp_r_prefix = dp_one_side(right,    h, L, side_label="RIGHT")

    Σ_left  = {c: v for c, v in zip(C_l, dp_l_prefix)}
    Σ_right = {c: v for c, v in zip(C_r, dp_r_prefix)}   # prefix – not yet used

    # Cost if no tour crosses the origin
    cost_no_bridge = dp_l_prefix[-1] + dp_r_prefix[-1]

    # ----------------  Phase 2 – Suffix table for the right side ----------
    Σ̃_right = _suffix_dp_right(right, h, L)

    # ----------------  Phase 3 – Enumerate *maximal* bridge candidates ----
    best = cost_no_bridge

    # Pre‑compute dictionary for quick index lookup in C_r
    idx_r = {c: i for i, c in enumerate(C_r)}

    for i_p, p in enumerate(C_l):
        # Farthest q we can reach with battery L (binary search)
        lo, hi = 0, len(C_r) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if tour_length(p, C_r[mid], h) <= L + EPS:
                lo = mid + 1
            else:
                hi = mid - 1
        if hi < 0:
            continue  # cannot pair this p with anything
        q = C_r[hi]

        # ---- maximality checks ----
        # 1. Right‑maximal: cannot extend q to the next candidate
        if hi + 1 < len(C_r) and tour_length(p, C_r[hi + 1], h) <= L + EPS:
            continue
        # 2. Left‑maximal: cannot extend p to the previous candidate
        if i_p > 0 and tour_length(C_l[i_p - 1], q, h) <= L + EPS:
            continue

        # ---- Total cost with this bridge ----
        cost_left  = dp_l_prefix[i_p]
        cost_bridge = tour_length(p, q, h)
        cost_right = Σ̃_right.get(q, math.inf)

        total = cost_left + cost_bridge + cost_right
        if total < best:
            best = total

    return best
