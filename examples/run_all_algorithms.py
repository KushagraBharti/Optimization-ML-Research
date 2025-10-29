from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

try:
    from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
    from coverage_planning.algs.geometry import tour_length
    from coverage_planning.algs.reference import (
        dp_full,
        dpos,
        gsp,
        gs,
    )
except ImportError:  # pragma: no cover - layout fallback
    import sys
    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
    from coverage_planning.algs.geometry import tour_length
    from coverage_planning.algs.reference import (
        dp_full,
        dpos,
        gsp,
        gs,
    )


TOL = TOL_NUM


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
        for u, v in tours:
            lo, hi = (u, v) if u <= v else (v, u)
            if hi < a - tol or lo > b + tol:
                continue
            pieces.append((max(lo, a), min(hi, b)))
        if not pieces:
            raise AssertionError(f"Segment [{a}, {b}] not covered by tours")
        pieces.sort()
        coverage = pieces[0][0]
        if coverage > a + tol:
            raise AssertionError(f"Left end {a} not covered (gap of {coverage - a})")
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


@dataclass
class ExampleResult:
    name: str
    detail_lines: List[str]

    def emit(self) -> None:
        print(f"=== {self.name} ===")
        for line in self.detail_lines:
            print(line)
        print()


def run_gs_example() -> ExampleResult:
    segments = [(-9.0, -6.0), (-2.0, -1.0), (1.5, 2.5), (6.0, 9.0)]
    h = 3.0
    L = 28.0
    try:
        count, tours = gs(segments, h=h, L=L)
    except ValueError as exc:
        return ExampleResult(
            "GS / MinTours",
            [
                f"Segments: {segments}",
                f"h={h}, L={L}",
                f"Infeasible instance: {exc}",
            ],
        )

    check_tours_feasible(h, L, tours)
    check_cover_exact(segments, tours)

    total_len = tour_length_sum(h, tours)
    detail = [
        f"Segments: {segments}",
        f"h={h}, L={L}",
        f"tours={count}, endpoints={tours}",
        f"total_length={total_len:.6f}",
    ]
    return ExampleResult("GS / MinTours", detail)


def run_gsp_example() -> ExampleResult:
    seg = (-7.0, 5.0)
    h = 2.5
    L = 21.5
    try:
        count, tours = gsp(seg, h=h, L=L)
    except ValueError as exc:
        return ExampleResult(
            "GSP / Single Segment",
            [
                f"Segment: {seg}",
                f"h={h}, L={L}",
                f"Infeasible instance: {exc}",
            ],
        )

    check_tours_feasible(h, L, tours)
    check_cover_exact([seg], tours)
    total_len = tour_length_sum(h, tours)

    detail = [
        f"Segment: {seg}",
        f"h={h}, L={L}",
        f"tours={count}, sweeps={tours}",
        f"total_length={total_len:.6f}",
    ]
    return ExampleResult("GSP / Single Segment", detail)


def run_dpos_example() -> ExampleResult:
    segments = [(0.0, 2.0), (3.0, 3.5), (6.0, 8.0)]
    h = 2.0
    L = 17.5
    try:
        Sigma, candidates = dpos(segments, h=h, L=L)
    except ValueError as exc:
        return ExampleResult(
            "DPOS / One-Sided",
            [
                f"Segments: {segments}",
                f"h={h}, L={L}",
                f"Infeasible instance: {exc}",
            ],
        )

    sigma_last = Sigma[-1]
    detail = [
        f"Segments: {segments}",
        f"h={h}, L={L}",
        f"|C|={len(candidates)}, candidates={candidates}",
        f"Sigma*(b_last)={sigma_last:.6f}",
    ]
    return ExampleResult("DPOS / One-Sided", detail)


def run_full_line_example() -> ExampleResult:
    segments = [
        (-8.89835723442975, -6.9393559339752535),
        (-5.307185703922216, -3.7870960811960925),
        (2.3179655663366483, 3.307328152086481),
        (4.770532804219864, 6.2435507360996265),
    ]
    h = 2.5
    L = 18.626939647203475
    try:
        cost_full, tours = dp_full(segments, h=h, L=L)
    except ValueError as exc:
        return ExampleResult(
            "Full-Line DP",
            [
                f"Segments: {segments}",
                f"h={h}, L={L}",
                f"Infeasible instance: {exc}",
            ],
        )

    left = [seg for seg in segments if seg[1] <= 0.0]
    right = [seg for seg in segments if seg[0] >= 0.0]

    baseline = 0.0
    if left:
        Sigma_left, _ = dpos([(-b, -a) for a, b in reversed(left)], h=h, L=L)
        baseline += Sigma_left[-1]
    if right:
        Sigma_right, _ = dpos(right, h=h, L=L)
        baseline += Sigma_right[-1]

    if baseline <= 0.0:
        raise AssertionError("Baseline cost unexpectedly non-positive")

    detail = [
        f"Segments: {segments}",
        f"h={h}, L={L}",
        f"baseline_cost={baseline:.6f}",
        f"full_cost={cost_full:.6f}",
        f"bridge_improvement={(baseline - cost_full) / baseline * 100.0:.3f}%",
        f"tours_output={tours if tours else 'N/A (cost-only reference)'}",
    ]

    # Monotonicity sanity check
    L_up = 1.2 * L
    cost_looser, _ = dp_full(segments, h=h, L=L_up)
    if cost_looser > cost_full + TOL:
        raise AssertionError(
            f"Monotonicity violated: cost({L_up})={cost_looser} > cost({L})={cost_full}"
        )
    detail.append(f"monotonicity_check: cost@1.2L={cost_looser:.6f}")

    return ExampleResult("Full-Line DP", detail)


def main() -> None:
    runners = [
        run_gs_example,
        run_gsp_example,
        run_dpos_example,
        run_full_line_example,
    ]
    for fn in runners:
        result = fn()
        result.emit()


if __name__ == "__main__":
    main()
