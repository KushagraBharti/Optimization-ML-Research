# benchmark/benchmark_dp.py
"""
Timing harness for the exact DP algorithms.

• bench_one_side  – one-sided DP (DPOS) on random right-side instances
• bench_full_line – full-line DP on alternating left/right segments
"""

from __future__ import annotations
import random
import statistics
import time
from typing import List, Tuple

from coverage_planning.dp_1side import dp_one_side
from coverage_planning.dp_both import dp_full_line
from coverage_planning.utils import tour_length


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _random_segments_right(n: int, span: float = 100.0) -> List[Tuple[float, float]]:
    xs = sorted(random.uniform(0.0, span) for _ in range(2 * n))
    return [(xs[2 * i], xs[2 * i + 1]) for i in range(n)]


def _alternating_left_right(n: int, span: float = 100.0) -> List[Tuple[float, float]]:
    """
    Build n disjoint segments, alternating side of the origin.
    """
    xs = sorted(random.uniform(0.0, span) for _ in range(2 * n))
    segs = [(xs[2 * i], xs[2 * i + 1]) for i in range(n)]
    out = []
    for idx, (a, b) in enumerate(segs):
        if idx % 2 == 0:           # send to the left
            out.append((-b, -a))
        else:
            out.append((a, b))
    return out


# ---------------------------------------------------------------------------
#  Benchmarks
# ---------------------------------------------------------------------------
def bench_one_side(avg_runs: int = 5) -> None:
    print("=== One-Sided DP Benchmarks ===")
    for n in [100, 200, 400]:
        times, states = [], []
        for _ in range(avg_runs):
            segs = _random_segments_right(n)
            h = 0.0
            L = max(tour_length(a, b, h) for a, b in segs) + 1e-6
            t0 = time.perf_counter()
            prefix, _suffix, C = dp_one_side(segs, h, L)
            times.append(time.perf_counter() - t0)
            states.append(len(C))
        print(
            f"n={n:4d} | |C|≈{statistics.mean(states):.0f} "
            f"| t_avg={statistics.mean(times):.4f}s "
            f"| t_std={statistics.stdev(times):.4f}s"
        )


def bench_full_line(avg_runs: int = 5) -> None:
    print("\n=== Full-Line DP Benchmarks ===")
    for n in [100, 200]:
        times = []
        for _ in range(avg_runs):
            segs = _alternating_left_right(n)
            h = 0.0
            # generous battery so instance is feasible
            L = max(tour_length(a, b, h) for a, b in segs) * 2 + 1e-6
            t0 = time.perf_counter()
            _ = dp_full_line(segs, h, L)
            times.append(time.perf_counter() - t0)
        print(
            f"n={n:4d} | t_avg={statistics.mean(times):.4f}s "
            f"| t_std={statistics.stdev(times):.4f}s"
        )


if __name__ == "__main__":
    random.seed(42)
    bench_one_side()
    bench_full_line()
