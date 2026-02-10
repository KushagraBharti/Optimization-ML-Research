from __future__ import annotations

from bisect import bisect_left
from functools import lru_cache

from ..candidates import SegmentLocator, build_candidate_sets_one_side
from ..coverage import subtract_interval_from_segments
from ..geometry import feasible_tour, tour_length
from ..model import EPS, Instance, Segment, Solution, Tour
from ..algorithms.gs_min_tours import solve_min_tours_gs


def _round_key(x: float, digits: int = 12) -> float:
    return round(x, digits)


class _CandidateResolver:
    def __init__(self, points: list[float]) -> None:
        self.points = sorted(points)
        self.key_to_point = {_round_key(x): x for x in self.points}

    def resolve(self, x: float, tol: float = 1e-6) -> float:
        k = _round_key(x)
        if k in self.key_to_point:
            return self.key_to_point[k]
        i = bisect_left(self.points, x)
        for j in (i - 1, i):
            if 0 <= j < len(self.points):
                if abs(self.points[j] - x) <= tol:
                    return self.points[j]
        raise KeyError(f"Point {x} not in resolver.")


def exact_min_length_one_side(instance: Instance) -> Solution:
    if instance.is_empty:
        return Solution.from_tours([], metadata={"oracle": "exact_one_side"})
    if not instance.one_side():
        raise ValueError("exact_min_length_one_side expects one-sided instance.")

    normalized, sign = instance.normalize_to_right_side()
    if normalized.is_empty:
        return Solution.from_tours([], metadata={"oracle": "exact_one_side"})

    cset = list(build_candidate_sets_one_side(normalized).union)
    for s in normalized.segments:
        if all(abs(s.b - x) > 1e-7 for x in cset):
            cset.append(s.b)
    cset = sorted(set(cset))
    resolver = _CandidateResolver(cset)
    locator = SegmentLocator(normalized.segments)

    @lru_cache(maxsize=None)
    def solve_cost(x_key: float) -> float:
        x = resolver.resolve(x_key)
        loc_x = locator.locate(x)
        if loc_x.kind != "on_segment" or loc_x.segment_index is None:
            return float("inf")
        jk = loc_x.segment_index
        best = float("inf")

        # Candidate-left transitions (feasible for exhaustive enumeration).
        for left in cset:
            if left >= x - EPS:
                break
            loc_left = locator.locate(left)
            if loc_left.kind != "on_segment":
                continue
            length = tour_length(left, x, normalized.h)
            if length > normalized.L + EPS:
                continue
            candidate = length + solve_cost(_round_key(left))
            if candidate < best:
                best = candidate

        # Segment-left transitions.
        for j in range(0, jk + 1):
            left = normalized.segments[j].a
            length = tour_length(left, x, normalized.h)
            if length > normalized.L + EPS:
                continue
            if j == 0:
                prev_cost = 0.0
            else:
                prev = resolver.resolve(normalized.segments[j - 1].b)
                prev_cost = solve_cost(_round_key(prev))
            candidate = length + prev_cost
            if candidate < best:
                best = candidate
        return best

    choice: dict[float, tuple[float | None, float, float]] = {}

    def reconstruct_choice(x: float) -> None:
        if _round_key(x) in choice:
            return
        loc_x = locator.locate(x)
        if loc_x.kind != "on_segment" or loc_x.segment_index is None:
            raise RuntimeError("Reconstruction failed: endpoint not on segment.")
        jk = loc_x.segment_index
        best = solve_cost(_round_key(x))
        picked: tuple[float | None, float, float] | None = None

        for left in cset:
            if left >= x - EPS:
                break
            loc_left = locator.locate(left)
            if loc_left.kind != "on_segment":
                continue
            length = tour_length(left, x, normalized.h)
            if length > normalized.L + EPS:
                continue
            candidate = length + solve_cost(_round_key(left))
            if abs(candidate - best) <= 1e-6:
                picked = (left, left, length)
                break

        if picked is None:
            for j in range(0, jk + 1):
                left = normalized.segments[j].a
                length = tour_length(left, x, normalized.h)
                if length > normalized.L + EPS:
                    continue
                prev = None if j == 0 else resolver.resolve(normalized.segments[j - 1].b)
                prev_cost = 0.0 if prev is None else solve_cost(_round_key(prev))
                candidate = length + prev_cost
                if abs(candidate - best) <= 1e-6:
                    picked = (prev, left, length)
                    break
        if picked is None:
            raise RuntimeError("Reconstruction failed: no matching transition.")
        choice[_round_key(x)] = picked
        prev, _, _ = picked
        if prev is not None:
            reconstruct_choice(prev)

    target = resolver.resolve(normalized.bn)
    _ = solve_cost(_round_key(target))
    reconstruct_choice(target)

    tours_rev: list[Tour] = []
    cursor: float | None = target
    guard = 0
    while cursor is not None:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("Oracle reconstruction loop guard triggered.")
        prev, left, length = choice[_round_key(cursor)]
        tours_rev.append(
            Tour(
                left=left,
                right=cursor,
                length=length,
                maximal=abs(length - normalized.L) <= 1e-6,
                tag="oracle_one_side",
            )
        )
        cursor = prev
    tours = list(reversed(tours_rev))

    if sign == -1:
        mapped = [
            Tour(
                left=-t.right,
                right=-t.left,
                length=t.length,
                maximal=t.maximal,
                tag=t.tag,
            )
            for t in tours
        ]
        return Solution.from_tours(mapped, metadata={"oracle": "exact_one_side", "sign": sign})
    return Solution.from_tours(tours, metadata={"oracle": "exact_one_side", "sign": sign})


def exact_min_length_two_side(instance: Instance) -> Solution:
    if instance.is_empty:
        return Solution.from_tours([], metadata={"oracle": "exact_two_side"})
    if instance.one_side():
        return exact_min_length_one_side(instance)

    left = instance.clipped(right=0.0)
    right = instance.clipped(left=0.0)

    independent = Solution.from_tours(
        [*exact_min_length_one_side(left).tours, *exact_min_length_one_side(right).tours],
        metadata={"oracle": "exact_two_side", "case": "independent"},
    )
    best = independent

    if not left.is_empty and not right.is_empty:
        c_left = [x for x in build_candidate_sets_one_side(left).union if x < -EPS]
        c_right = [x for x in build_candidate_sets_one_side(right).union if x > EPS]

        left_cache: dict[float, Solution] = {}
        right_cache: dict[float, Solution] = {}
        for p in c_left:
            for q in c_right:
                if not feasible_tour(p, q, instance.h, instance.L):
                    continue
                if p not in left_cache:
                    left_cache[p] = exact_min_length_one_side(instance.clipped(right=p))
                if q not in right_cache:
                    right_cache[q] = exact_min_length_one_side(instance.clipped(left=q))
                center = Tour(
                    left=p,
                    right=q,
                    length=tour_length(p, q, instance.h),
                    maximal=False,
                    tag="oracle_center",
                )
                candidate = Solution.from_tours(
                    [*left_cache[p].tours, center, *right_cache[q].tours],
                    metadata={"oracle": "exact_two_side", "case": "interior", "pair": (p, q)},
                )
                if candidate.total_length < best.total_length - EPS:
                    best = candidate
    return best


def exact_min_tours_small(instance: Instance) -> Solution:
    """Small-instance exact reference for MinTours.

    For bounded tests we use the proven-optimal GS implementation as the exact
    reference and verify independently via exhaustive MinLength checks.
    """
    return solve_min_tours_gs(instance)


def exhaustive_grid_instances(
    h: float,
    L: float,
    x_values: list[float],
    max_segments: int = 3,
) -> list[Instance]:
    out: list[Instance] = []
    vals = sorted(x_values)
    segs: list[tuple[float, float]] = []

    def rec(start: int, k: int) -> None:
        if k > 0:
            out.append(Instance.from_iterable(h=h, L=L, segments=segs))
        if k >= max_segments:
            return
        n = len(vals)
        for i in range(start, n):
            for j in range(i, n):
                if i > start and segs and vals[i] <= segs[-1][1] + EPS:
                    continue
                candidate = (vals[i], vals[j])
                if segs and candidate[0] <= segs[-1][1] + EPS:
                    continue
                segs.append(candidate)
                rec(j + 1, k + 1)
                segs.pop()

    rec(0, 0)
    return out
