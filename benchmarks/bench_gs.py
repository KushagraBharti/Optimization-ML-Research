from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from typing import Iterable, List, Sequence, Tuple

try:
    from coverage_planning.algs.geometry import EPS, tour_length
    from coverage_planning.algs.reference import greedy_min_tours_ref
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.algs.geometry import EPS, tour_length
    from coverage_planning.algs.reference import greedy_min_tours_ref


TOL = 1e-6


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


def check_cover_exact(
    segments: Sequence[Tuple[float, float]],
    tours: Sequence[Tuple[float, float]],
    *,
    tol: float = 1e-7,
) -> None:
    for a, b in segments:
        pieces = []
        for p, q in tours:
            lo, hi = (p, q) if p <= q else (q, p)
            if hi < a - tol or lo > b + tol:
                continue
            pieces.append((max(lo, a), min(hi, b)))
        if not pieces:
            raise AssertionError(f"Segment [{a}, {b}] not covered")
        pieces.sort()
        coverage = pieces[0][0]
        if coverage > a + tol:
            raise AssertionError(f"Gap before {a}")
        coverage = pieces[0][1]
        for start, end in pieces[1:]:
            if start > coverage + tol:
                raise AssertionError("Gap detected within segment")
            coverage = max(coverage, end)
        if coverage < b - tol:
            raise AssertionError(f"Segment [{a}, {b}] not fully covered")


def check_tours_feasible(
    h: float, L: float, tours: Sequence[Tuple[float, float]], *, tol: float = TOL
) -> None:
    for idx, (p, q) in enumerate(tours):
        length = tour_length(min(p, q), max(p, q), h)
        if length > L + tol:
            raise AssertionError(
                f"Tour #{idx + 1} length {length:.9f} exceeds limit {L:.9f}"
            )


def tour_length_sum(h: float, tours: Iterable[Tuple[float, float]]) -> float:
    return sum(tour_length(min(p, q), max(p, q), h) for p, q in tours)


def gen_one_side_segments(
    rng: random.Random,
    count: int,
    *,
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
        seg_start = cursor
        seg_end = seg_start + length
        segs.append((seg_start, seg_end))
        cursor = seg_end + rng.uniform(min_gap, max_gap)
    return segs


def generate_instance(
    rng: random.Random,
    *,
    k_max: int,
    h: float,
    alpha: float,
    beta: float,
) -> Tuple[List[Tuple[float, float]], float]:
    if k_max < 2:
        raise ValueError("k must be at least 2")
    k = rng.randint(2, k_max)
    left_count = rng.randint(1, k - 1)
    right_count = k - left_count

    min_len = 0.8
    max_len = 3.2
    min_gap = 0.6
    max_gap = 2.4

    left_offset = rng.uniform(min_gap, max_gap)
    right_offset = rng.uniform(min_gap, max_gap)

    pos_left = gen_one_side_segments(
        rng,
        left_count,
        min_len=min_len,
        max_len=max_len,
        min_gap=min_gap,
        max_gap=max_gap,
        start_offset=left_offset,
    )
    pos_right = gen_one_side_segments(
        rng,
        right_count,
        min_len=min_len,
        max_len=max_len,
        min_gap=min_gap,
        max_gap=max_gap,
        start_offset=right_offset,
    )

    left_segments = [(-b, -a) for a, b in reversed(pos_left)]
    segments = left_segments + pos_right

    farthest = max(max(abs(a), abs(b)) for a, b in segments)
    avg_length = sum(b - a for a, b in segments) / len(segments)
    baseline = 2.0 * math.hypot(farthest, h)
    L = max(
        baseline * alpha,
        baseline * 1.05,
    ) + beta * avg_length
    return segments, L


def run_benchmark(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    alpha: float
    beta: float
    factor = args.L_factor
    if isinstance(factor, tuple):
        if len(factor) == 1:
            alpha = factor[0]
            beta = 0.3
        else:
            alpha, beta = factor[:2]
    else:
        alpha = float(factor)
        beta = 0.3

    durations: List[float] = []
    tour_counts: List[int] = []
    total_lengths: List[float] = []
    failures = 0

    total_runs = args.warmup + args.n
    for iteration in range(total_runs):
        instance_generated = False
        for _ in range(40):
            segments, L = generate_instance(
                rng, k_max=args.k, h=args.h, alpha=alpha, beta=beta
            )
            farthest = max(max(abs(a), abs(b)) for a, b in segments)
            if 2.0 * math.hypot(farthest, args.h) <= L + EPS:
                instance_generated = True
                break
        if not instance_generated:
            failures += 1
            continue

        try:
            start = time.perf_counter()
            count, tours = greedy_min_tours_ref(segments, h=args.h, L=L)
            elapsed = time.perf_counter() - start

            check_tours_feasible(args.h, L, tours)
            check_cover_exact(segments, tours)

            total_len = tour_length_sum(args.h, tours)
        except Exception:
            failures += 1
            continue

        if iteration >= args.warmup:
            durations.append(elapsed)
            tour_counts.append(count)
            total_lengths.append(total_len)

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

    if tour_counts:
        tour_counts_sorted = sorted(tour_counts)
        total_lengths_sorted = sorted(total_lengths)
        tours_mean = statistics.fmean(tour_counts)
        tours_p90 = percentile(tour_counts_sorted, 0.9)
        lengths_mean = statistics.fmean(total_lengths)
        lengths_p90 = percentile(total_lengths_sorted, 0.9)
        print(
            f"tours_mean={tours_mean:.3f},tours_p90={tours_p90:.3f},"
            f"totallen_mean={lengths_mean:.6f},totallen_p90={lengths_p90:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GS / MinTours reference algorithm.")
    parser.add_argument("--n", type=int, default=10000, help="Number of timed iterations.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic RNG seed.")
    parser.add_argument("--k", type=int, default=5, help="Maximum segments per instance.")
    parser.add_argument(
        "--h", type=float, default=3.0, help="Sensor altitude h (applied to all instances)."
    )
    parser.add_argument(
        "--L_factor",
        type=parse_float_or_tuple,
        default=parse_float_or_tuple("1.12,0.35"),
        help="Scaling factors for L. Provide 'alpha' or 'alpha,beta'.",
    )
    args = parser.parse_args()

    if isinstance(args.L_factor, str):
        args.L_factor = parse_float_or_tuple(args.L_factor)
    run_benchmark(args)


if __name__ == "__main__":
    main()
