# src/solvers/dp.py

import math
from typing import List, Tuple

def tour_length(p: float, q: float, h: float = 1.0) -> float:
    """
    Same tour‐length formula as in greedy.py
    """
    return math.hypot(p, h) + math.hypot(q, h) + (q - p)


def cover_min_length(
    segments: List[Tuple[float, float]],
    L: float,
    h: float = 1.0
) -> List[Tuple[float, float]]:
    """
    Dynamic‐programming to cover all segments with minimal total tour length.
    We only consider tours whose endpoints coincide with segment endpoints.
    
    segments: sorted list of (a_i, b_i)
    Returns list of (p, q) in order.
    """
    n = len(segments)
    # Precompute feasible[i][j] = length if covering segments[i..j] is <= L, else inf
    feasible = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            p = segments[i][0]
            q = segments[j][1]
            l = tour_length(p, q, h)
            if l <= L:
                feasible[i][j] = l

    # DP: dp[k] = min total length to cover first k segments (0..k-1)
    dp = [math.inf] * (n + 1)
    back = [-1] * (n + 1)
    dp[0] = 0.0

    for k in range(1, n + 1):
        # try all i such that tour covers segments[i..k-1]
        for i in range(1, k + 1):
            cost = feasible[i - 1][k - 1]
            if cost < math.inf and dp[i - 1] + cost < dp[k]:
                dp[k] = dp[i - 1] + cost
                back[k] = i - 1

    # if dp[n] is inf, no solution
    if dp[n] == math.inf:
        return []

    # reconstruct tours
    tours: List[Tuple[float, float]] = []
    k = n
    while k > 0:
        i = back[k]
        p = segments[i][0]
        q = segments[k - 1][1]
        tours.append((p, q))
        k = i
    tours.reverse()
    return tours
