from __future__ import annotations

import argparse
import math
import statistics
import time
from typing import Dict, Iterable, List, Tuple
import random

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import dp_full, dpos, gs, gsp
from coverage_planning.common.constants import RNG_SEEDS, seed_everywhere


def _gen_segments(
    rng,
    *,
    n: int,
    span: float,
    min_len: float,
    min_gap: float,
    left_fraction: float = 0.5,
) -> List[Tuple[float, float]]:
    def _one_side(k: int) -> List[Tuple[float, float]]:
        segs: List[Tuple[float, float]] = []
        cursor = rng.uniform(0.0, min_gap)
        for _ in range(k):
            length = rng.uniform(min_len, min_len * 3.0)
            start = cursor
            end = min(start + length, span)
            segs.append((start, end))
            cursor = end + rng.uniform(min_gap, min_gap * 3.0)
        return segs

    left_count = max(1, int(n * left_fraction))
    right_count = max(1, n - left_count)
    left = [(-b, -a) for a, b in reversed(_one_side(left_count))]
    right = _one_side(right_count)
    return left + right


def _gen_instance(rng, mode: str, size: str, h: float):
    if mode == "gsp":
        span = 10.0 if size == "small" else 25.0
        left = rng.uniform(-span, -span / 5.0)
        right = rng.uniform(span / 5.0, span)
        seg = (left, right)
        L = tour_length(seg[0], seg[1], h) * (1.05 if size == "small" else 1.1)
        return seg, L

    n = 6 if size == "small" else 16
    segments = _gen_segments(
        rng,
        n=n,
        span=60.0 if size == "small" else 120.0,
        min_len=0.6 if size == "small" else 1.0,
        min_gap=0.6 if size == "small" else 1.0,
        left_fraction=0.4 if size == "small" else 0.5,
    )
    max_atomic = max(tour_length(a, b, h) for a, b in segments)
    if mode == "gs":
        L = max_atomic * (1.35 if size == "small" else 1.5)
    elif mode == "dpos":
        segments = [(a, b) for a, b in segments if a >= -1e-9]
        L = max_atomic * 1.35
    else:  # full DP
        L = max_atomic * (1.25 if size == "small" else 1.35)
    return segments, L


def _percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = (len(sorted_values) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[hi]
    weight = idx - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro benchmark for reference solvers.")
    parser.add_argument("--n", type=int, default=2000, help="Number of timed iterations.")
    parser.add_argument("--seed", type=int, default=RNG_SEEDS["bench"])
    parser.add_argument("--h", type=float, default=3.0)
    parser.add_argument("--mode", choices=["gs", "gsp", "dpos", "full"], default="gs")
    parser.add_argument("--size", choices=["small", "medium"], default="small")
    args = parser.parse_args()

    seed_everywhere(args.seed)
    rng = random.Random(args.seed)

    warmup = 100
    durations: List[float] = []
    candidate_sizes: List[int] = []
    total_runs = warmup + args.n

    for iteration in range(total_runs):
        instance, L = _gen_instance(rng, args.mode, args.size, args.h)
        try:
            start = time.perf_counter()
            if args.mode == "gs":
                _, tours = gs(instance, h=args.h, L=L)
                _ = sum(len(tour) for tour in tours)  # touch result
            elif args.mode == "gsp":
                gsp(instance, h=args.h, L=L)
            elif args.mode == "dpos":
                debug: Dict[str, object] = {}
                dpos(instance, h=args.h, L=L, debug=debug)
                candidate_sizes.append(debug["candidate_count"])
            else:  # full
                debug: Dict[str, object] = {}
                dp_full(instance, h=args.h, L=L, debug=debug)
                left = debug["left"]["candidate_count"]
                tail = debug["tail"]["candidate_count"]
                candidate_sizes.append(left + tail)
            elapsed = time.perf_counter() - start
        except Exception:
            if iteration >= warmup:
                candidate_sizes.append(0)
            continue

        if iteration >= warmup:
            durations.append(elapsed)

    durations.sort()
    total_time = sum(durations)
    mean = statistics.fmean(durations) if durations else float("nan")
    median = statistics.median(durations) if durations else float("nan")
    p50 = median
    p90 = _percentile(durations, 0.9)
    p99 = _percentile(durations, 0.99)
    summary = (
        f"mode={args.mode} size={args.size} runs={len(durations)} "
        f"total={total_time:.6f}s mean={mean:.6f}s "
        f"p50={p50:.6f}s p90={p90:.6f}s p99={p99:.6f}s "
        f"min={durations[0]:.6f}s max={durations[-1]:.6f}s"
        if durations
        else "No successful runs recorded."
    )
    print(summary)
    if candidate_sizes:
        avg_candidates = statistics.fmean(candidate_sizes)
        print(f"candidate_stats mean={avg_candidates:.2f} max={max(candidate_sizes)}")


if __name__ == "__main__":
    main()
