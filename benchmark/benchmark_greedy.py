# benchmark/benchmark_greedy.py
"""
Speed test for Greedy MinTours on large *pair-wise disjoint* instances.

Three distributions:
    • Uniform      – random in [0,span], sorted & gap-corrected
    • Clustered    – Gaussian clusters but gaps enforced
    • Worst-case   – many tiny segments, minimum positive gap
"""

from __future__ import annotations
import random
import statistics
import time
from typing import List, Tuple

from coverage_planning.algs.heuristics.gs_mintours import greedy_min_tours
from coverage_planning.algs.geometry import tour_length, EPS


# ---------------------------------------------------------------------------
#  Utility: enforce positive gaps
# ---------------------------------------------------------------------------
def _disjointify(segs: List[Tuple[float, float]], gap: float = 1e-3) -> List[Tuple[float, float]]:
    """
    Given *unsorted* segments, sort them and push each one rightwards
    just enough so that a_{i} ≥ b_{i-1} + gap.
    """
    out: List[Tuple[float, float]] = []
    for a, b in sorted(segs, key=lambda s: s[0]):
        if out and a <= out[-1][1] + gap:
            shift = out[-1][1] + gap - a
            a += shift
            b += shift
        if b - a < EPS:                   # avoid zero-length
            b = a + EPS
        out.append((a, b))
    return out


# ---------------------------------------------------------------------------
#  Instance generators
# ---------------------------------------------------------------------------
def _random_uniform(n: int, span: float = 100.0) -> List[Tuple[float, float]]:
    xs = sorted(random.uniform(0.0, span) for _ in range(2 * n))
    segs = [(xs[2 * i], xs[2 * i + 1]) for i in range(n)]
    return _disjointify(segs)


def _random_clustered(
    n: int, span: float = 100.0, clusters: int = 5
) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    points_per = n // clusters
    for _ in range(clusters):
        centre = random.uniform(0.1 * span, 0.9 * span)
        for _ in range(points_per):
            length = random.uniform(0.2, 5.0)
            left = centre + random.gauss(0.0, 2.0)
            segs.append((left, left + length))
    return _disjointify(segs)


def _worst_case(n: int) -> List[Tuple[float, float]]:
    # tiny segments 0.009 long with gap 0.001
    return [(i * 0.01, i * 0.01 + 0.009) for i in range(n)]


# ---------------------------------------------------------------------------
#  Benchmark driver
# ---------------------------------------------------------------------------
def _bench(
    gen_fn,
    label: str,
    sizes: tuple[int, ...] = (1000, 5000, 10000),
    runs: int = 3,
) -> None:
    print(f"\n=== Greedy MinTours ({label}) ===")
    for n in sizes:
        times = []
        for _ in range(runs):
            segs = gen_fn(n)
            h = 0.0
            L = tour_length(segs[0][0], segs[-1][1], h) + 1e-6
            t0 = time.perf_counter()
            greedy_min_tours(segs, h, L)
            times.append(time.perf_counter() - t0)
        print(
            f"n={n:6d} | t_avg={statistics.mean(times):.4f}s "
            f"| t_std={statistics.stdev(times):.4f}s"
        )


if __name__ == "__main__":
    random.seed(123)
    _bench(_random_uniform, "Uniform")
    _bench(_random_clustered, "Clustered")
    _bench(_worst_case, "Worst-Case Tiny")
