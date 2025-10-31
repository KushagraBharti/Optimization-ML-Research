"""Benchmark harness for the reference one-sided DP solver (DPOS)."""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from typing import List, Sequence, Tuple

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import dpos
from coverage_planning.common.constants import DEFAULT_SEED, RNG_SEEDS, TOL_NUM

TOL = TOL_NUM
BENCH_SEED = RNG_SEEDS.get("bench", DEFAULT_SEED)


def parse_float_or_tuple(value: str) -> float | Tuple[float, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    floats = tuple(float(p) for p in parts)
    if len(floats) == 1:
        return floats[0]
    return floats


def percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 1:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    weight = idx - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def gen_one_side_segments(
    rng: random.Random,
    *,
    count: int,
    min_len: float,
    max_len: float,
    min_gap: float,
    max_gap: float,
) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    cursor = rng.uniform(0.5, 1.5)
    for _ in range(count):
        length = rng.uniform(min_len, max_len)
        start = cursor
        end = start + length
        segs.append((start, end))
        cursor = end + rng.uniform(min_gap, max_gap)
    return segs


def generate_instance(
    rng: random.Random,
    *,
    h: float,
    k_max: int,
    alpha: float,
    margin: float,
) -> Tuple[List[Tuple[float, float]], float]:
    count = rng.randint(1, k_max)
    segments = gen_one_side_segments(
        rng,
        count=count,
        min_len=0.6,
        max_len=3.0,
        min_gap=0.5,
        max_gap=2.0,
    )
    max_atomic = max(tour_length(a, b, h) for a, b in segments)
    L = max(max_atomic * alpha, max_atomic + margin)
    if L - max_atomic < margin * 0.5:
        L = max_atomic + max(margin, 1.0)
    return segments, L


def run_benchmark(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    factor = args.L_factor
    if isinstance(factor, tuple):
        if len(factor) == 1:
            alpha = factor[0]
            margin = 1.5
        else:
            alpha, margin = factor[:2]
    else:
        alpha = float(factor)
        margin = 1.5

    durations: List[float] = []
    candidate_counts: List[int] = []
    sigma_values: List[float] = []
    failures = 0

    total_runs = args.warmup + args.n
    for iteration in range(total_runs):
        try:
            segments, L = generate_instance(
                rng, h=args.h, k_max=args.k, alpha=alpha, margin=margin
            )
        except Exception:
            failures += 1
            continue

        try:
            start = time.perf_counter()
            Sigma, candidates = dpos(segments, h=args.h, L=L)
            elapsed = time.perf_counter() - start
            if not candidates:
                raise RuntimeError("Empty candidate set")
            if len(Sigma) != len(candidates):
                raise RuntimeError("Sigma/C size mismatch")
            cand_count = len(candidates)
            sigma_last = Sigma[-1]
        except Exception:
            failures += 1
            continue

        if iteration >= args.warmup:
            durations.append(elapsed)
            candidate_counts.append(cand_count)
            sigma_values.append(sigma_last)

    durations.sort()
    total_time = sum(durations)
    mean = statistics.fmean(durations) if durations else float("nan")
    median = statistics.median(durations) if durations else float("nan")
    p90 = percentile(durations, 0.9)
    p99 = percentile(durations, 0.99)
    min_v = durations[0] if durations else float("nan")
    max_v = durations[-1] if durations else float("nan")

    summary = (
        f"total_time={total_time:.6f},mean={mean:.6f},median={median:.6f},"
        f"p90={p90:.6f},p99={p99:.6f},min={min_v:.6f},max={max_v:.6f},"
        f"failures={failures}"
    )
    print(summary)

    if candidate_counts:
        candidate_counts_sorted = sorted(candidate_counts)
        sigma_sorted = sorted(sigma_values)
        candidates_mean = statistics.fmean(candidate_counts)
        candidates_p90 = percentile(candidate_counts_sorted, 0.9)
        sigma_mean = statistics.fmean(sigma_values)
        sigma_p90 = percentile(sigma_sorted, 0.9)
        print(
            f"candidates_mean={candidates_mean:.3f},candidates_p90={candidates_p90:.3f},"
            f"sigma_mean={sigma_mean:.6f},sigma_p90={sigma_p90:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DPOS one-sided reference algorithm.")
    parser.add_argument("--n", type=int, default=10000, help="Number of timed iterations.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations.")
    parser.add_argument("--seed", type=int, default=BENCH_SEED, help="Deterministic RNG seed.")
    parser.add_argument("--k", type=int, default=5, help="Maximum segments per instance.")
    parser.add_argument(
        "--h", type=float, default=2.5, help="Sensor altitude h (applied to all instances)."
    )
    parser.add_argument(
        "--L_factor",
        type=parse_float_or_tuple,
        default=parse_float_or_tuple("1.12,1.5"),
        help="Scaling factors for L. Provide 'alpha' or 'alpha,margin'.",
    )
    args = parser.parse_args()
    if isinstance(args.L_factor, str):
        args.L_factor = parse_float_or_tuple(args.L_factor)
    run_benchmark(args)


if __name__ == "__main__":
    main()
