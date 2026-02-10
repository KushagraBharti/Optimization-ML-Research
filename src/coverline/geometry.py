from __future__ import annotations

import math

from .model import EPS


def dist_origin(x: float, h: float) -> float:
    return math.hypot(x, h)


def tour_length(left: float, right: float, h: float) -> float:
    if left > right + EPS:
        raise ValueError("Expected left <= right.")
    return dist_origin(left, h) + dist_origin(right, h) + abs(right - left)


def feasible_tour(left: float, right: float, h: float, L: float) -> bool:
    return tour_length(left, right, h) <= L + EPS


def can_cover_interval_with_one_tour(a: float, b: float, h: float, L: float) -> bool:
    return feasible_tour(a, b, h, L)


def can_cover_all_with_one_tour(min_x: float, max_x: float, h: float, L: float) -> bool:
    return feasible_tour(min_x, max_x, h, L)


def _bisection_monotone_decreasing_root(
    f,
    lo: float,
    hi: float,
    eps: float = EPS,
    max_iter: int = 120,
) -> float:
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo < -eps or f_hi > eps:
        raise ValueError("Invalid bisection bracket for decreasing function.")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) <= eps or abs(hi - lo) <= eps:
            return mid
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_maximal_left_endpoint(right: float, h: float, L: float) -> float:
    """Return the leftmost feasible endpoint p for a maximal tour (p, right)."""
    if 2.0 * dist_origin(right, h) > L + EPS:
        raise ValueError(f"Point x={right} is unreachable for L={L}, h={h}.")
    f = lambda x: tour_length(x, right, h) - L
    hi = right
    lo = right
    step = max(1.0, L)
    while f(lo) <= 0:
        lo -= step
        step *= 2.0
        if step > 1e16:
            raise RuntimeError("Failed to bracket root while solving maximal left endpoint.")
    return _bisection_monotone_decreasing_root(f=f, lo=lo, hi=hi)


def solve_maximal_right_endpoint(left: float, h: float, L: float) -> float:
    """Return the rightmost feasible endpoint q for a maximal tour (left, q)."""
    if 2.0 * dist_origin(left, h) > L + EPS:
        raise ValueError(f"Point x={left} is unreachable for L={L}, h={h}.")
    f = lambda x: tour_length(left, x, h) - L
    lo = left
    hi = left
    step = max(1.0, L)
    while f(hi) <= 0:
        hi += step
        step *= 2.0
        if step > 1e16:
            raise RuntimeError("Failed to bracket root while solving maximal right endpoint.")
    # f is increasing in x; convert by negating.
    g = lambda x: -f(x)
    return _bisection_monotone_decreasing_root(f=g, lo=lo, hi=hi)


def choose_farthest_endpoint(a: float, b: float, h: float) -> float:
    da = dist_origin(a, h)
    db = dist_origin(b, h)
    if da > db + EPS:
        return a
    if db > da + EPS:
        return b
    return min(a, b)
