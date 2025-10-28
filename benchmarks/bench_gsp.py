from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from typing import Iterable, List, Sequence, Tuple

try:
    from coverage_planning.algs.geometry import EPS, tour_length
    from coverage_planning.algs.reference import greedy_min_length_one_segment_ref
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.algs.geometry import EPS, tour_length
    from coverage_planning.algs.reference import greedy_min_length_one_segment_ref


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


def tour_length_sum(h: float, tours: Iterable[Tuple[float, float]]) -> float:
    return sum(tour_length(min(p, q), max(p, q), h) for p, q in tours)


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
                raise AssertionError("Gap detected within segment coverage")
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


def generate_one_tour_instance(
    rng: random.Random,
    *,
    h: float,
    alpha: float,
    span_factor: float,
) -> Tuple[Tuple[float, float], float]:
    sign = rng.choice([-1, 1])
    span = max(1.0, span_factor)
    base = rng.uniform(1.0, 4.0 * span)
    length = rng.uniform(0.8, 2.6 * span)
    if sign < 0:
        seg = (-base - length, -base)
    else:
        seg = (base, base + length)
    direct = tour_length(seg[0], seg[1], h)
    farthest = max(abs(seg[0]), abs(seg[1]))
    L = max(direct * alpha, 2.0 * math.hypot(farthest, h) * 1.02)
    return seg, L


def generate_two_tour_instance(
    rng: random.Random,
    *,
    h: float,
    alpha: float,
    span_factor: float,
) -> Tuple[Tuple[float, float], float]:
    for _ in range(100):
        span = max(1.0, span_factor)
        left_extent = rng.uniform(3.0, 8.0 * span)
        right_extent = rng.uniform(2.5, 6.5 * span)
        left = -left_extent
        right = right_extent
        len_left = tour_length(left, 0.0, h)
        len_right = tour_length(0.0, right, h)
        direct = tour_length(left, right, h)
        L = max(len_left, len_right) * alpha
        farthest = max(abs(left), abs(right))
        L = max(L, 2.0 * math.hypot(farthest, h) * 1.02)
        if direct > L + 0.5 and len_left <= L + EPS and len_right <= L + EPS:
            return (left, right), L
    raise RuntimeError("Failed to generate two-tour instance")


def run_benchmark(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    factor = args.L_factor
    if isinstance(factor, tuple):
        if len(factor) == 1:
            alpha_one = factor[0]
            alpha_two = factor[0] + 0.05
        else:
            alpha_one = factor[0]
            alpha_two = factor[1]
    else:
        alpha_one = float(factor)
        alpha_two = alpha_one + 0.05

    durations: List[float] = []
    tour_counts: List[int] = []
    total_lengths: List[float] = []
    failures = 0

    total_runs = args.warmup + args.n
    for iteration in range(total_runs):
        target_mode = "one" if iteration % 2 == 0 else "two"
        try:
            if target_mode == "one":
                seg, L = generate_one_tour_instance(
                    rng, h=args.h, alpha=alpha_one, span_factor=args.k / 4.0
                )
            else:
                seg, L = generate_two_tour_instance(
                    rng, h=args.h, alpha=alpha_two, span_factor=args.k / 4.0
                )
        except Exception:
            failures += 1
            continue

        farthest = max(abs(seg[0]), abs(seg[1]))
        if 2.0 * math.hypot(farthest, args.h) > L + EPS:
            L = 2.0 * math.hypot(farthest, args.h) * 1.05

        try:
            start = time.perf_counter()
            count, tours = greedy_min_length_one_segment_ref(seg, h=args.h, L=L)
            elapsed = time.perf_counter() - start
            check_tours_feasible(args.h, L, tours)
            check_cover_exact([seg], tours)
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
    parser = argparse.ArgumentParser(description="Benchmark GSP single-segment reference algorithm.")
    parser.add_argument("--n", type=int, default=10000, help="Number of timed iterations.")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup iterations.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic RNG seed.")
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Maximum crust sweeps (used to cap generated segment spans).",
    )
    parser.add_argument(
        "--h", type=float, default=2.5, help="Sensor altitude h (applied to all instances)."
    )
    parser.add_argument(
        "--L_factor",
        type=parse_float_or_tuple,
        default=parse_float_or_tuple("1.04,1.10"),
        help="Scaling factors for L. Provide 'alpha_one' or 'alpha_one,alpha_two'.",
    )
    args = parser.parse_args()
    if isinstance(args.L_factor, str):
        args.L_factor = parse_float_or_tuple(args.L_factor)
    run_benchmark(args)


if __name__ == "__main__":
    main()
