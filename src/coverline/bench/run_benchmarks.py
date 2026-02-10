from __future__ import annotations

import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from ..algorithms import (
    solve_min_length_one_side_dpos,
    solve_min_length_two_side,
    solve_min_tours_gs,
    solve_min_tours_gs_linear,
)
from ..model import Instance
from ..geometry import tour_length
from ..oracles.exact_small import exact_min_length_one_side, exact_min_length_two_side


@dataclass(frozen=True)
class BenchConfig:
    seed: int
    repeats: int
    one_side_sizes: tuple[int, ...]
    two_side_sizes: tuple[int, ...]
    min_tours_sizes: tuple[int, ...]


def _config(mode: str) -> BenchConfig:
    if mode == "quick":
        return BenchConfig(
            seed=20260210,
            repeats=3,
            one_side_sizes=(4, 8, 12),
            two_side_sizes=(4, 8),
            min_tours_sizes=(6, 12, 20),
        )
    if mode == "deep":
        return BenchConfig(
            seed=20260210,
            repeats=24,
            one_side_sizes=(8, 16, 24, 32, 40, 48, 56),
            two_side_sizes=(8, 12, 16, 20, 24),
            min_tours_sizes=(20, 40, 60, 80, 100),
        )
    raise ValueError(f"Unknown benchmark mode: {mode}")


def _random_one_side_instance(rng: random.Random, n: int, h: float = 4.0, L: float = 20.0) -> Instance:
    segments: list[tuple[float, float]] = []
    x = rng.uniform(0.2, 1.0)
    for _ in range(n):
        seg_len = rng.uniform(0.18, 0.35)
        gap = rng.uniform(0.05, 0.2)
        a = x
        b = x + seg_len
        segments.append((a, b))
        x = b + gap
    max_b = segments[-1][1]
    min_feasible_L = (2.0 * (max_b * max_b + h * h) ** 0.5) + 0.25
    single_tour = tour_length(segments[0][0], segments[-1][1], h)
    L_eff = min_feasible_L + 0.6
    if L_eff >= single_tour - 0.05:
        L_eff = max(min_feasible_L + 0.01, single_tour * 0.9)
    return Instance.from_iterable(h=h, L=L_eff, segments=segments)


def _random_two_side_instance(rng: random.Random, n: int, h: float = 4.0, L: float = 20.0) -> Instance:
    half = max(1, n // 2)
    left: list[tuple[float, float]] = []
    x = -rng.uniform(1.0, 1.8)
    for _ in range(half):
        seg_len = rng.uniform(0.2, 0.45)
        gap = rng.uniform(0.08, 0.25)
        b = x
        a = b - seg_len
        left.append((a, b))
        x = a - gap
    left = list(reversed(left))

    right: list[tuple[float, float]] = []
    x = rng.uniform(1.0, 1.8)
    for _ in range(n - half):
        seg_len = rng.uniform(0.2, 0.45)
        gap = rng.uniform(0.08, 0.25)
        a = x
        b = a + seg_len
        right.append((a, b))
        x = b + gap
    all_segments = left + right
    far = max(abs(all_segments[0][0]), abs(all_segments[-1][1]))
    min_feasible_L = (2.0 * (far * far + h * h) ** 0.5) + 0.25
    single_tour = tour_length(all_segments[0][0], all_segments[-1][1], h)
    L_eff = min_feasible_L + 0.7
    if L_eff >= single_tour - 0.05:
        L_eff = max(min_feasible_L + 0.01, single_tour * 0.9)
    return Instance.from_iterable(h=h, L=L_eff, segments=all_segments)


def _random_min_tours_instance(rng: random.Random, n: int, h: float = 4.0, L: float = 30.0) -> Instance:
    segments: list[tuple[float, float]] = []
    x = -1.0 - (0.28 * n)
    for _ in range(n):
        seg_len = rng.uniform(0.12, 0.28)
        gap = rng.uniform(0.04, 0.16)
        a = x + gap
        b = a + seg_len
        segments.append((a, b))
        x = b
    far = max(abs(segments[0][0]), abs(segments[-1][1]))
    min_feasible_L = (2.0 * (far * far + h * h) ** 0.5) + 0.25
    single_tour = tour_length(segments[0][0], segments[-1][1], h)
    L_eff = min_feasible_L + 0.5
    if L_eff >= single_tour - 0.05:
        L_eff = max(min_feasible_L + 0.01, single_tour * 0.9)
    return Instance.from_iterable(h=h, L=L_eff, segments=segments)


def _time_call(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    t1 = time.perf_counter()
    return result, (t1 - t0)


def run_benchmarks(out_dir: Path, mode: str = "deep") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _config(mode)
    rng = random.Random(cfg.seed)

    rows_one_side: list[dict[str, float | int | str]] = []
    rows_two_side: list[dict[str, float | int | str]] = []
    rows_min_tours: list[dict[str, float | int | str]] = []

    for n in cfg.one_side_sizes:
        fast_times = []
        oracle_times = []
        for _ in range(cfg.repeats):
            instance = _random_one_side_instance(rng, n=n)
            fast_sol, t_fast = _time_call(solve_min_length_one_side_dpos, instance)
            oracle_sol, t_oracle = _time_call(exact_min_length_one_side, instance)
            fast_times.append(t_fast)
            oracle_times.append(t_oracle)
            rows_one_side.append(
                {
                    "n": n,
                    "solver": "dpos",
                    "time_s": t_fast,
                    "tour_count": fast_sol.tour_count,
                    "total_length": fast_sol.total_length,
                    "oracle_gap": fast_sol.total_length - oracle_sol.total_length,
                }
            )
            rows_one_side.append(
                {
                    "n": n,
                    "solver": "oracle_exact",
                    "time_s": t_oracle,
                    "tour_count": oracle_sol.tour_count,
                    "total_length": oracle_sol.total_length,
                    "oracle_gap": 0.0,
                }
            )

    for n in cfg.two_side_sizes:
        for _ in range(cfg.repeats):
            instance = _random_two_side_instance(rng, n=n)
            fast_sol, t_fast = _time_call(solve_min_length_two_side, instance)
            oracle_sol, t_oracle = _time_call(exact_min_length_two_side, instance)
            rows_two_side.append(
                {
                    "n": n,
                    "solver": "two_side_fast",
                    "time_s": t_fast,
                    "tour_count": fast_sol.tour_count,
                    "total_length": fast_sol.total_length,
                    "oracle_gap": fast_sol.total_length - oracle_sol.total_length,
                }
            )
            rows_two_side.append(
                {
                    "n": n,
                    "solver": "two_side_oracle",
                    "time_s": t_oracle,
                    "tour_count": oracle_sol.tour_count,
                    "total_length": oracle_sol.total_length,
                    "oracle_gap": 0.0,
                }
            )

    for n in cfg.min_tours_sizes:
        for _ in range(cfg.repeats):
            instance = _random_min_tours_instance(rng, n=n)
            log_sol, t_log = _time_call(solve_min_tours_gs, instance)
            lin_sol, t_lin = _time_call(solve_min_tours_gs_linear, instance)
            rows_min_tours.append(
                {
                    "n": n,
                    "solver": "gs_log",
                    "time_s": t_log,
                    "tour_count": log_sol.tour_count,
                    "total_length": log_sol.total_length,
                    "count_diff_vs_linear": log_sol.tour_count - lin_sol.tour_count,
                }
            )
            rows_min_tours.append(
                {
                    "n": n,
                    "solver": "gs_linear",
                    "time_s": t_lin,
                    "tour_count": lin_sol.tour_count,
                    "total_length": lin_sol.total_length,
                    "count_diff_vs_linear": 0,
                }
            )

    def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out_dir / "one_side_benchmark.csv", rows_one_side)
    write_csv(out_dir / "two_side_benchmark.csv", rows_two_side)
    write_csv(out_dir / "min_tours_benchmark.csv", rows_min_tours)

    # Summary plot: median runtime by n.
    def median_by(rows: list[dict], solver_name: str) -> tuple[list[int], list[float]]:
        buckets: dict[int, list[float]] = {}
        for r in rows:
            if r["solver"] != solver_name:
                continue
            buckets.setdefault(int(r["n"]), []).append(float(r["time_s"]))
        xs = sorted(buckets.keys())
        ys = [statistics.median(buckets[x]) for x in xs]
        return xs, ys

    fig, ax = plt.subplots(figsize=(8, 4))
    x, y = median_by(rows_one_side, "dpos")
    if x:
        ax.plot(x, y, marker="o", label="DPOS one-side")
    x, y = median_by(rows_one_side, "oracle_exact")
    if x:
        ax.plot(x, y, marker="o", label="Oracle one-side")
    x, y = median_by(rows_two_side, "two_side_fast")
    if x:
        ax.plot(x, y, marker="s", label="Two-side fast")
    x, y = median_by(rows_two_side, "two_side_oracle")
    if x:
        ax.plot(x, y, marker="s", label="Two-side oracle")
    ax.set_xlabel("n segments")
    ax.set_ylabel("median time (s)")
    ax.set_title(f"Coverline Benchmarks ({mode})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_summary.png", dpi=160)
    plt.close(fig)

    summary_lines = [
        f"# Benchmark Summary ({mode})",
        "",
        f"- Seed: `{cfg.seed}`",
        f"- Repeats: `{cfg.repeats}`",
        f"- One-side rows: `{len(rows_one_side)}`",
        f"- Two-side rows: `{len(rows_two_side)}`",
        f"- MinTours rows: `{len(rows_min_tours)}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "mode": mode,
        "seed": cfg.seed,
        "repeats": cfg.repeats,
        "rows": {
            "one_side": len(rows_one_side),
            "two_side": len(rows_two_side),
            "min_tours": len(rows_min_tours),
        },
        "out_dir": str(out_dir),
    }
