from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from ..candidates import SegmentLocator, build_candidate_sets_one_side
from ..geometry import dist_origin, solve_maximal_right_endpoint, tour_length
from ..model import EPS, Instance, Solution, Tour
from .dpos_one_side import solve_min_length_one_side_dpos


@dataclass(frozen=True)
class TwoSideCaseResult:
    name: str
    solution: Solution


def _round_key(x: float, digits: int = 12) -> float:
    return round(x, digits)


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
            if 0 <= j < len(self.points) and abs(self.points[j] - x) <= tol:
                return self.points[j]
        raise KeyError(f"Point {x} is not in resolver.")


@dataclass(frozen=True)
class _SuffixDecision:
    point: float
    next_point: float | None
    tour_left: float
    tour_right: float
    tour_length: float
    case: int
    rule: str


@dataclass(frozen=True)
class _OneSideSuffixArtifacts:
    instance: Instance
    candidates: tuple[float, ...]
    states: tuple[float, ...]
    cost: dict[float, float]
    decision: dict[float, _SuffixDecision]
    candidate_resolver: _PointResolver
    state_resolver: _PointResolver


def _combine_solutions(name: str, left: Solution, middle: Tour | None, right: Solution) -> Solution:
    tours = list(left.tours)
    if middle is not None:
        tours.append(middle)
    tours.extend(right.tours)
    return Solution.from_tours(
        tours,
        metadata={
            "algorithm": "min_length_two_side",
            "case": name,
            "left_tours": left.tour_count,
            "right_tours": right.tour_count,
            "middle": None if middle is None else (middle.left, middle.right),
        },
    )


def _independent_case(instance: Instance) -> TwoSideCaseResult:
    left = solve_min_length_one_side_dpos(instance.clipped(right=0.0))
    right = solve_min_length_one_side_dpos(instance.clipped(left=0.0))
    combined = _combine_solutions("independent", left, None, right)
    return TwoSideCaseResult(name="independent", solution=combined)


def _build_suffix_artifacts_positive(
    instance: Instance,
    query_points: list[float] | None = None,
) -> _OneSideSuffixArtifacts:
    if instance.is_empty:
        resolver = _PointResolver([])
        return _OneSideSuffixArtifacts(
            instance=instance,
            candidates=(),
            states=(),
            cost={},
            decision={},
            candidate_resolver=resolver,
            state_resolver=resolver,
        )
    if not instance.one_side() or instance.a1 < -EPS:
        raise ValueError("Suffix DP expects a one-sided instance on the non-negative side.")

    csets = build_candidate_sets_one_side(instance)
    candidates = list(csets.union)
    if query_points is not None:
        candidates.extend(query_points)
    for s in instance.segments:
        if all(abs(s.b - c) > 1e-7 for c in candidates):
            candidates.append(s.b)
    candidates = sorted(set(candidates))
    candidate_resolver = _PointResolver(candidates)

    locator = SegmentLocator(instance.segments)
    states_map: dict[float, float] = {}

    def add_state(x: float) -> None:
        k = _round_key(x)
        if k not in states_map:
            states_map[k] = x

    for s in instance.segments:
        add_state(s.a)
    for c in candidates:
        add_state(c)

    queue = list(states_map.values())
    guard = 0
    while queue:
        guard += 1
        if guard > 200_000:
            raise RuntimeError("Suffix DP state-construction loop guard triggered.")
        point = queue.pop()
        right = solve_maximal_right_endpoint(left=point, h=instance.h, L=instance.L)
        if right >= instance.bn - EPS:
            continue
        loc_right = locator.locate(right)
        if loc_right.kind != "on_segment":
            continue
        k = _round_key(right)
        if k not in states_map:
            states_map[k] = right
            queue.append(right)

    states = sorted(states_map.values())
    state_resolver = _PointResolver(states)
    n = len(instance.segments)

    cost: dict[float, float] = {}
    decision: dict[float, _SuffixDecision] = {}

    for point in reversed(states):
        point = state_resolver.resolve(point)
        loc_point = locator.locate(point)
        if loc_point.kind != "on_segment" or loc_point.segment_index is None:
            continue
        jk = loc_point.segment_index
        right = solve_maximal_right_endpoint(left=point, h=instance.h, L=instance.L)

        best = float("inf")
        best_dec: _SuffixDecision | None = None

        if right >= instance.bn - EPS:
            length = tour_length(point, instance.bn, instance.h)
            best = length
            best_dec = _SuffixDecision(
                point=point,
                next_point=None,
                tour_left=point,
                tour_right=instance.bn,
                tour_length=length,
                case=1,
                rule="case1_full_suffix",
            )
            cost[point] = best
            decision[point] = best_dec
            continue

        loc_right = locator.locate(right)
        case = 2
        j_prime: int | None = None
        c_prime: float | None = None

        if loc_right.kind == "in_gap":
            if loc_right.left_segment_index is None:
                raise RuntimeError("Gap location missing left segment index in suffix DP.")
            j_prime = loc_right.left_segment_index
        elif loc_right.kind == "on_segment" and loc_right.segment_index is not None:
            j_prime = loc_right.segment_index
            try:
                c_prime = state_resolver.resolve(right)
                case = 3
            except KeyError:
                case = 2
        else:
            raise RuntimeError("Unexpected right-endpoint location in suffix DP.")

        if case == 3 and c_prime is not None:
            recurse_point: float | None = c_prime
            if abs(c_prime - instance.segments[j_prime].b) <= 1e-6:
                if j_prime == n - 1:
                    recurse_point = None
                else:
                    recurse_point = state_resolver.resolve(instance.segments[j_prime + 1].a)
            if recurse_point is not None and recurse_point not in cost:
                raise RuntimeError("Suffix DP maximal transition target missing from cost table.")
            length = tour_length(point, c_prime, instance.h)
            tail = 0.0 if recurse_point is None else cost[recurse_point]
            candidate_value = length + tail
            if candidate_value < best - EPS:
                best = candidate_value
                best_dec = _SuffixDecision(
                    point=point,
                    next_point=recurse_point,
                    tour_left=point,
                    tour_right=c_prime,
                    tour_length=length,
                    case=3,
                    rule="case3_maximal_link",
                )

        if j_prime is None:
            raise RuntimeError("Suffix DP missing j_prime.")
        j_upper = j_prime - 1 if case == 3 else j_prime
        for j in range(jk, j_upper + 1):
            right_endpoint = instance.segments[j].b
            length = tour_length(point, right_endpoint, instance.h)
            if length > instance.L + EPS:
                continue
            next_point = None
            tail_cost = 0.0
            if j < n - 1:
                next_point = state_resolver.resolve(instance.segments[j + 1].a)
                if next_point not in cost:
                    raise RuntimeError("Suffix DP tail state is missing from cost table.")
                tail_cost = cost[next_point]
            candidate_value = length + tail_cost
            if candidate_value < best - EPS:
                best = candidate_value
                best_dec = _SuffixDecision(
                    point=point,
                    next_point=next_point,
                    tour_left=point,
                    tour_right=right_endpoint,
                    tour_length=length,
                    case=case,
                    rule=f"case{case}_segment_right_j{j + 1}",
                )

        if best_dec is None:
            raise RuntimeError(f"Suffix DP found no transition for state {point}.")
        cost[point] = best
        decision[point] = best_dec

    for c in candidates:
        resolved = candidate_resolver.resolve(c)
        if resolved not in cost:
            raise RuntimeError(f"Suffix DP did not compute cost for candidate {c}.")

    return _OneSideSuffixArtifacts(
        instance=instance,
        candidates=tuple(candidates),
        states=tuple(states),
        cost=cost,
        decision=decision,
        candidate_resolver=candidate_resolver,
        state_resolver=state_resolver,
    )


def _suffix_cost(artifacts: _OneSideSuffixArtifacts, start: float) -> float:
    point = artifacts.candidate_resolver.resolve(start)
    return artifacts.cost[point]


def _suffix_solution(
    artifacts: _OneSideSuffixArtifacts,
    start: float,
    tag_prefix: str,
) -> Solution:
    point = artifacts.candidate_resolver.resolve(start)
    tours: list[Tour] = []
    cursor: float | None = point
    guard = 0
    while cursor is not None:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("Suffix DP reconstruction loop guard triggered.")
        dec = artifacts.decision[cursor]
        tours.append(
            Tour(
                left=dec.tour_left,
                right=dec.tour_right,
                length=dec.tour_length,
                maximal=abs(dec.tour_length - artifacts.instance.L) <= 1e-6,
                tag=f"{tag_prefix}:{dec.rule}",
            )
        )
        cursor = dec.next_point
    return Solution.from_tours(
        tours,
        metadata={
            "algorithm": "dpos_one_side_topdown_suffix",
            "start": point,
            "state_count": len(artifacts.states),
        },
    )


def _map_mirrored_suffix_to_left(mirrored_solution: Solution) -> Solution:
    mapped = [
        Tour(
            left=-t.right,
            right=-t.left,
            length=t.length,
            maximal=t.maximal,
            tag="two_side_left_suffix",
        )
        for t in mirrored_solution.tours
    ]
    mapped = list(reversed(mapped))
    return Solution.from_tours(mapped, metadata={"algorithm": "two_side_left_suffix_mirror"})


def _interior_case(instance: Instance) -> TwoSideCaseResult | None:
    left_instance = instance.clipped(right=0.0)
    right_instance = instance.clipped(left=0.0)
    if left_instance.is_empty or right_instance.is_empty:
        return None

    # Lemma 9 candidates.
    c_left = [x for x in build_candidate_sets_one_side(left_instance).union if x < -EPS]
    c_right = [x for x in build_candidate_sets_one_side(right_instance).union if x > EPS]
    if not c_left or not c_right:
        return None

    # Restrict to candidates within maximal tours touching O' as endpoint.
    try:
        q_max0 = solve_maximal_right_endpoint(left=0.0, h=instance.h, L=instance.L)
    except ValueError:
        return None
    p_min0 = -q_max0
    c_left = [p for p in c_left if p >= p_min0 - EPS]
    c_right = [q for q in c_right if q <= q_max0 + EPS]

    if not c_left or not c_right:
        return None

    left_mirrored = left_instance.mirrored_x()
    left_suffix = _build_suffix_artifacts_positive(left_mirrored, query_points=[-p for p in c_left])
    right_suffix = _build_suffix_artifacts_positive(right_instance, query_points=c_right)

    left_tail: dict[float, float] = {p: _suffix_cost(left_suffix, -p) for p in c_left}
    c_right = sorted(c_right)
    right_value: list[float] = []
    for q in c_right:
        right_cost = _suffix_cost(right_suffix, q)
        right_value.append(right_cost + dist_origin(q, instance.h) + q)

    prefix_best_value: list[float] = []
    prefix_best_q: list[float] = []
    for idx, q in enumerate(c_right):
        val = right_value[idx]
        if idx == 0 or val < prefix_best_value[idx - 1] - EPS:
            prefix_best_value.append(val)
            prefix_best_q.append(q)
        else:
            prefix_best_value.append(prefix_best_value[idx - 1])
            prefix_best_q.append(prefix_best_q[idx - 1])

    best_total = float("inf")
    best_pair: tuple[float, float] | None = None
    pair_queries = 0

    for p in c_left:
        try:
            q_upper = solve_maximal_right_endpoint(left=p, h=instance.h, L=instance.L)
        except ValueError:
            continue
        if q_upper <= EPS:
            continue
        idx = bisect_right(c_right, q_upper + EPS) - 1
        if idx < 0:
            continue
        pair_queries += 1
        q = prefix_best_q[idx]
        candidate_total = left_tail[p] + (dist_origin(p, instance.h) - p) + prefix_best_value[idx]
        if candidate_total < best_total - EPS:
            best_total = candidate_total
            best_pair = (p, q)

    if best_pair is None:
        return None

    best_p, best_q = best_pair
    left_mirrored_solution = _suffix_solution(left_suffix, -best_p, tag_prefix="left_suffix")
    left_solution = _map_mirrored_suffix_to_left(left_mirrored_solution)
    right_solution = _suffix_solution(right_suffix, best_q, tag_prefix="right_suffix")
    center_len = tour_length(best_p, best_q, instance.h)
    center_tour = Tour(
        left=best_p,
        right=best_q,
        length=center_len,
        maximal=abs(center_len - instance.L) <= 1e-6,
        tag="two_side_interior",
    )
    best = _combine_solutions("interior_projection", left_solution, center_tour, right_solution)

    meta = dict(best.metadata)
    meta["decomposition"] = "theorem10_topdown"
    meta["pair"] = best_pair
    meta["c_left"] = len(c_left)
    meta["c_right"] = len(c_right)
    meta["pair_queries"] = pair_queries
    meta["left_suffix_states"] = len(left_suffix.states)
    meta["right_suffix_states"] = len(right_suffix.states)
    best = Solution(
        tours=best.tours,
        total_length=best.total_length,
        tour_count=best.tour_count,
        metadata=meta,
    )
    return TwoSideCaseResult(name="interior_projection", solution=best)


def solve_min_length_two_side(instance: Instance) -> Solution:
    if instance.is_empty:
        return Solution.from_tours([], metadata={"algorithm": "min_length_two_side"})
    if instance.one_side():
        sol = solve_min_length_one_side_dpos(instance)
        meta = dict(sol.metadata)
        meta["algorithm"] = "min_length_two_side"
        meta["case"] = "one_side_reduction"
        return Solution(
            tours=sol.tours,
            total_length=sol.total_length,
            tour_count=sol.tour_count,
            metadata=meta,
        )

    candidates: list[TwoSideCaseResult] = [_independent_case(instance)]
    interior = _interior_case(instance)
    if interior is not None:
        candidates.append(interior)

    best = min(candidates, key=lambda r: r.solution.total_length)
    meta = dict(best.solution.metadata)
    meta["case_candidates"] = [c.name for c in candidates]
    return Solution(
        tours=best.solution.tours,
        total_length=best.solution.total_length,
        tour_count=best.solution.tour_count,
        metadata=meta,
    )


def solve_min_length_two_side_reference(instance: Instance) -> Solution:
    """Reference solver for small instances."""
    return solve_min_length_two_side(instance)
