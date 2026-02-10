from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from ..candidates import SegmentLocator, build_candidate_sets_one_side
from ..geometry import feasible_tour, solve_maximal_left_endpoint, tour_length
from ..model import EPS, Instance, Segment, Solution, Tour


def _round_key(x: float, digits: int = 12) -> float:
    return round(x, digits)


@dataclass(frozen=True)
class DPDecision:
    point: float
    prev_point: float | None
    tour_left: float
    tour_right: float
    tour_length: float
    case: int
    rule: str


@dataclass(frozen=True)
class OneSideDPArtifacts:
    candidates: tuple[float, ...]
    cost: dict[float, float]
    decision: dict[float, DPDecision]
    normalized_sign: int


class _PointResolver:
    def __init__(self, points: list[float]) -> None:
        self.points = sorted(points)
        self.key_to_point = {_round_key(p): p for p in self.points}

    def resolve(self, x: float, tol: float = 1e-6) -> float:
        k = _round_key(x)
        if k in self.key_to_point:
            return self.key_to_point[k]
        i = bisect_left(self.points, x)
        for j in (i - 1, i):
            if 0 <= j < len(self.points):
                if abs(self.points[j] - x) <= tol:
                    return self.points[j]
        raise KeyError(f"Point {x} was not found in candidate set.")


def _solve_dpos_positive(instance: Instance) -> tuple[Solution, OneSideDPArtifacts]:
    if instance.is_empty:
        sol = Solution.from_tours([], metadata={"algorithm": "dpos_one_side"})
        artifacts = OneSideDPArtifacts(candidates=(), cost={}, decision={}, normalized_sign=1)
        return sol, artifacts
    if instance.a1 < -EPS:
        raise ValueError("_solve_dpos_positive expects non-negative one-side instance.")

    csets = build_candidate_sets_one_side(instance)
    candidates = list(csets.union)
    # Ensure every b_i is represented (robustness under floating point keying).
    for s in instance.segments:
        if all(abs(s.b - c) > 1e-7 for c in candidates):
            candidates.append(s.b)
    candidates = sorted(set(candidates))

    locator = SegmentLocator(instance.segments)
    resolver = _PointResolver(candidates)

    cost: dict[float, float] = {}
    decision: dict[float, DPDecision] = {}

    def sigma_at(point: float) -> float:
        p = resolver.resolve(point)
        return cost[p]

    for ck in candidates:
        loc_k = locator.locate(ck)
        if loc_k.kind != "on_segment" or loc_k.segment_index is None:
            continue
        jk = loc_k.segment_index
        left = solve_maximal_left_endpoint(ck, instance.h, instance.L)

        best = float("inf")
        best_dec: DPDecision | None = None

        if left <= instance.segments[0].a + EPS:
            length = tour_length(instance.segments[0].a, ck, instance.h)
            best = length
            best_dec = DPDecision(
                point=ck,
                prev_point=None,
                tour_left=instance.segments[0].a,
                tour_right=ck,
                tour_length=length,
                case=1,
                rule="case1_full_prefix",
            )
            cost[ck] = best
            decision[ck] = best_dec
            continue

        loc_left = locator.locate(left)
        case = 2
        j_prime = 0
        c_prime = None

        if loc_left.kind == "in_gap":
            if loc_left.right_segment_index is None:
                raise RuntimeError("Gap location missing right segment index.")
            j_prime = loc_left.right_segment_index
            case = 2
        elif loc_left.kind == "on_segment" and loc_left.segment_index is not None:
            j_prime = loc_left.segment_index
            c_prime = left
            try:
                _ = resolver.resolve(left)
                case = 3
            except KeyError:
                case = 2
        else:
            raise RuntimeError("Unexpected left-endpoint location for DPOS.")

        # Case 3 option: maximal tour from ck to c_prime.
        if case == 3 and c_prime is not None:
            cp = resolver.resolve(c_prime)
            if cp in cost:
                length = tour_length(cp, ck, instance.h)
                candidate = length + cost[cp]
                if candidate < best - EPS:
                    best = candidate
                    best_dec = DPDecision(
                        point=ck,
                        prev_point=cp,
                        tour_left=cp,
                        tour_right=ck,
                        tour_length=length,
                        case=3,
                        rule="case3_maximal_link",
                    )

        # Segment-left endpoint options from Equation (1).
        j_start = j_prime + (1 if case == 3 else 0)
        for j in range(j_start, jk + 1):
            left_endpoint = instance.segments[j].a
            length = tour_length(left_endpoint, ck, instance.h)
            if length > instance.L + EPS:
                continue
            prev = None if j == 0 else resolver.resolve(instance.segments[j - 1].b)
            prev_cost = 0.0 if prev is None else cost[prev]
            candidate = length + prev_cost
            if candidate < best - EPS:
                best = candidate
                best_dec = DPDecision(
                    point=ck,
                    prev_point=prev,
                    tour_left=left_endpoint,
                    tour_right=ck,
                    tour_length=length,
                    case=case,
                    rule=f"case{case}_segment_left_j{j+1}",
                )

        if best_dec is None:
            raise RuntimeError(f"DPOS found no valid transition for candidate {ck}.")

        cost[ck] = best
        decision[ck] = best_dec

    target = resolver.resolve(instance.bn)
    if target not in cost:
        raise RuntimeError("DPOS did not compute objective for bn.")

    tours_rev: list[Tour] = []
    cursor: float | None = target
    guard = 0
    while cursor is not None:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("DPOS backtracking loop guard triggered.")
        dec = decision[cursor]
        tours_rev.append(
            Tour(
                left=dec.tour_left,
                right=dec.tour_right,
                length=dec.tour_length,
                maximal=abs(dec.tour_length - instance.L) <= 1e-6,
                tag=dec.rule,
            )
        )
        cursor = dec.prev_point

    tours = list(reversed(tours_rev))
    solution = Solution.from_tours(
        tours,
        metadata={
            "algorithm": "dpos_one_side",
            "target": target,
            "candidate_count": len(candidates),
        },
    )
    artifacts = OneSideDPArtifacts(
        candidates=tuple(candidates),
        cost=cost,
        decision=decision,
        normalized_sign=1,
    )
    return solution, artifacts


def solve_min_length_one_side_dpos_with_artifacts(instance: Instance) -> tuple[Solution, OneSideDPArtifacts]:
    if instance.is_empty:
        return _solve_dpos_positive(instance)
    if not instance.one_side():
        raise ValueError("DPOS expects one-sided instance.")

    normalized, sign = instance.normalize_to_right_side()
    normalized_solution, artifacts = _solve_dpos_positive(normalized)

    if sign == 1:
        return normalized_solution, artifacts

    mapped_tours = []
    for t in normalized_solution.tours:
        mapped_left = -t.right
        mapped_right = -t.left
        mapped_tours.append(
            Tour(
                left=mapped_left,
                right=mapped_right,
                length=t.length,
                maximal=t.maximal,
                tag=t.tag,
            )
        )
    mapped_solution = Solution.from_tours(
        mapped_tours,
        metadata={
            **normalized_solution.metadata,
            "normalized_sign": sign,
        },
    )
    mapped_artifacts = OneSideDPArtifacts(
        candidates=tuple(sorted((-c for c in artifacts.candidates))),
        cost={-k: v for k, v in artifacts.cost.items()},
        decision={
            -k: DPDecision(
                point=-k,
                prev_point=None if d.prev_point is None else -d.prev_point,
                tour_left=-d.tour_right,
                tour_right=-d.tour_left,
                tour_length=d.tour_length,
                case=d.case,
                rule=d.rule,
            )
            for k, d in artifacts.decision.items()
        },
        normalized_sign=sign,
    )
    return mapped_solution, mapped_artifacts


def solve_min_length_one_side_dpos(instance: Instance) -> Solution:
    solution, _ = solve_min_length_one_side_dpos_with_artifacts(instance)
    return solution


def solve_min_length_one_side_dpos_subinstance(
    base: Instance,
    left: float | None = None,
    right: float | None = None,
) -> Solution:
    clipped = base.clipped(left=left, right=right)
    return solve_min_length_one_side_dpos(clipped)
