"""Benchmark harness for the reference full-line DP solver."""

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
from coverage_planning.algs.reference import dp_full, dpos
from coverage_planning.common.constants import DEFAULT_SEED, EPS_GEOM, RNG_SEEDS, TOL_NUM

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
    start_offset: float,
) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    cursor = start_offset
    for _ in range(count):
        length = rng.uniform(min_len, max_len)
        start = cursor
        end = start + length
        segs.append((start, end))
        cursor = end + rng.uniform(min_gap, max_gap)
    return segs


def reflect_left_segments(segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    return [(-b, -a) for a, b in reversed(segments)]


def generate_instance(
    rng: random.Random,
    *,
    h: float,
    k_max: int,
    alpha: float,
    bridge_scale: float,
) -> Tuple[List[Tuple[float, float]], float]:
    total_segments = rng.randint(2, k_max)
    left_count = rng.randint(1, total_segments - 1)
    right_count = total_segments - left_count

    bridge_mode = (rng.random() < 0.5)

    min_len = 0.7
    max_len = 2.8
    min_gap = 0.5
    max_gap = 2.4

    if bridge_mode:
        left_offset = rng.uniform(0.4, 1.0)
        right_offset = rng.uniform(0.4, 1.0)
    else:
        left_offset = rng.uniform(1.0, 2.5)
        right_offset = rng.uniform(1.0, 2.5)

    left_pos = gen_one_side_segments(
        rng,
        count=left_count,
        min_len=min_len,
        max_len=max_len,
        min_gap=min_gap,
        max_gap=max_gap,
        start_offset=left_offset,
    )
    right_pos = gen_one_side_segments(
        rng,
        count=right_count,
        min_len=min_len,
        max_len=max_len,
        min_gap=min_gap,
        max_gap=max_gap,
        start_offset=right_offset,
    )

    left_segments = reflect_left_segments(left_pos)
    segments = left_segments + right_pos

    farthest = max(max(abs(a), abs(b)) for a, b in segments)
    max_atomic = max(tour_length(a, b, h) for a, b in segments)
    base_boundary = 2.0 * math.hypot(farthest, h)

    bridge_candidate = tour_length(left_segments[-1][0], right_pos[0][1], h)
    L_bridge = bridge_candidate * bridge_scale
    L = max(
        max_atomic * alpha,
        base_boundary * 1.05,
        L_bridge if bridge_mode else 0.0,
    )
    return segments, L


def baseline_costs(
    segments: List[Tuple[float, float]],
    *,
    h: float,
    L: float,
) -> Tuple[float, float, float]:
    left = [seg for seg in segments if seg[1] <= 0.0 + EPS_GEOM]
    right = [seg for seg in segments if seg[0] >= 0.0 - EPS_GEOM]

    cost_left = 0.0
    cost_right = 0.0

    if left:
        left_ref = reflect_left_segments(left)
        Sigma_L, _ = dpos(left_ref, h=h, L=L)
        cost_left = Sigma_L[-1]
    if right:
        Sigma_R, _ = dpos(right, h=h, L=L)
        cost_right = Sigma_R[-1]

    return cost_left + cost_right, cost_left, cost_right


def run_benchmark(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    factor = args.L_factor
    if isinstance(factor, tuple):
        if len(factor) == 1:
            alpha = factor[0]
            bridge_scale = factor[0] + 0.05
        else:
            alpha, bridge_scale = factor[:2]
    else:
        alpha = float(factor)
        bridge_scale = alpha + 0.05

    durations: List[float] = []
    improvement_values: List[float] = []
    improvement_positive: List[float] = []
    failures = 0

    total_runs = args.warmup + args.n
    for iteration in range(total_runs):
        try:
            segments, L = generate_instance(
                rng, h=args.h, k_max=args.k, alpha=alpha, bridge_scale=bridge_scale
            )
        except Exception:
            failures += 1
            continue

        farthest = max(max(abs(a), abs(b)) for a, b in segments)
        min_feasible = 2.0 * math.hypot(farthest, args.h)
        if L < min_feasible * 1.02:
            L = min_feasible * 1.02

        try:
            baseline, _, _ = baseline_costs(segments, h=args.h, L=L)
            start = time.perf_counter()
            cost_full, _ = dp_full(segments, h=args.h, L=L)
            elapsed = time.perf_counter() - start
            if baseline <= 0.0:
                raise RuntimeError("Baseline cost non-positive")
            improvement = (baseline - cost_full) / baseline
        except Exception:
            failures += 1
            continue

        if iteration >= args.warmup:
            durations.append(elapsed)
            improvement_values.append(improvement)
            if improvement > TOL:
                improvement_positive.append(improvement)

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

    if improvement_values:
        improvements_sorted = sorted(improvement_values)
        impr_mean = statistics.fmean(improvement_values)
        impr_p90 = percentile(improvements_sorted, 0.9)
        win_fraction = (
            len(improvement_positive) / len(improvement_values) if improvement_values else 0.0
        )
        if improvement_positive:
            improvements_pos_sorted = sorted(improvement_positive)
            impr_pos_p90 = percentile(improvements_pos_sorted, 0.9)
            impr_pos_mean = statistics.fmean(improvement_positive)
        else:
            impr_pos_mean = float("nan")
            impr_pos_p90 = float("nan")
        print(
            f"improvement_mean={impr_mean:.6f},improvement_p90={impr_p90:.6f},"
            f"bridge_win_fraction={win_fraction:.3f},"
            f"improvement_win_mean={impr_pos_mean:.6f},improvement_win_p90={impr_pos_p90:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DP full-line reference algorithm.")
    parser.add_argument("--n", type=int, default=10000, help="Number of timed iterations.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations.")
    parser.add_argument("--seed", type=int, default=BENCH_SEED, help="Deterministic RNG seed.")
    parser.add_argument("--k", type=int, default=6, help="Maximum segments per instance.")
    parser.add_argument(
        "--h", type=float, default=2.5, help="Sensor altitude h (applied to all instances)."
    )
    parser.add_argument(
        "--L_factor",
        type=parse_float_or_tuple,
        default=parse_float_or_tuple("1.10,1.15"),
        help="Scaling factors for L. Provide 'alpha' or 'alpha,bridge_scale'.",
    )
    args = parser.parse_args()
    if isinstance(args.L_factor, str):
        args.L_factor = parse_float_or_tuple(args.L_factor)
    run_benchmark(args)


if __name__ == "__main__":
    main()
