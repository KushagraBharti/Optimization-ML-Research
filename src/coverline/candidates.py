from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .geometry import solve_maximal_left_endpoint
from .model import EPS, Instance, Segment


def _round_key(x: float, digits: int = 12) -> float:
    return round(x, digits)


def unique_sorted_eps(values: list[float]) -> list[float]:
    if not values:
        return []
    vals = sorted(values)
    out = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > EPS:
            out.append(v)
    return out


@dataclass(frozen=True)
class LocateResult:
    kind: str
    segment_index: int | None = None
    left_segment_index: int | None = None
    right_segment_index: int | None = None


class SegmentLocator:
    def __init__(self, segments: tuple[Segment, ...]) -> None:
        self.segments = segments
        self.a = [s.a for s in segments]
        self.b = [s.b for s in segments]

    def locate(self, x: float) -> LocateResult:
        n = len(self.segments)
        if n == 0:
            return LocateResult(kind="empty")
        if x < self.a[0] - EPS:
            return LocateResult(kind="left_of_all")
        if x > self.b[-1] + EPS:
            return LocateResult(kind="right_of_all")

        j = bisect_right(self.a, x) - 1
        if j >= 0 and x <= self.b[j] + EPS:
            return LocateResult(kind="on_segment", segment_index=j)

        right_idx = bisect_left(self.a, x)
        left_idx = right_idx - 1
        return LocateResult(
            kind="in_gap",
            left_segment_index=left_idx if left_idx >= 0 else None,
            right_segment_index=right_idx if right_idx < n else None,
        )


@dataclass(frozen=True)
class CandidateTransition:
    ck: float
    jk: int
    maximal_left: float
    case: int
    j_prime: int
    c_prime: float


@dataclass(frozen=True)
class CandidateSets:
    by_index: dict[int, tuple[float, ...]]
    union: tuple[float, ...]


def build_candidate_sets_one_side(instance: Instance) -> CandidateSets:
    if not instance.one_side():
        raise ValueError("Candidate construction expects one-sided instance.")
    normalized, sign = instance.normalize_to_right_side()
    segments = normalized.segments
    locator = SegmentLocator(segments)

    by_idx: dict[int, tuple[float, ...]] = {}
    all_values: list[float] = []

    for i, s in enumerate(segments):
        chain: list[float] = [s.b]
        current = s.b
        guard = 0
        while True:
            guard += 1
            if guard > 10_000:
                raise RuntimeError("Candidate chain loop guard triggered.")
            left = solve_maximal_left_endpoint(current, normalized.h, normalized.L)
            if left <= segments[0].a + EPS:
                break
            loc = locator.locate(left)
            if loc.kind == "in_gap":
                break
            if loc.kind != "on_segment":
                break
            chain.append(left)
            current = left

        chain = unique_sorted_eps(chain)
        if sign == -1:
            mapped = tuple(sorted((-x for x in chain)))
        else:
            mapped = tuple(chain)
        by_idx[i] = mapped
        all_values.extend(mapped)

    all_values = unique_sorted_eps(all_values)
    return CandidateSets(by_index=by_idx, union=tuple(all_values))


def _map_to_normalized_x(x: float, sign: int) -> float:
    return x if sign == 1 else -x


def _map_from_normalized_x(x: float, sign: int) -> float:
    return x if sign == 1 else -x


def build_candidate_transitions_one_side(instance: Instance) -> tuple[tuple[float, ...], dict[float, CandidateTransition]]:
    if not instance.one_side():
        raise ValueError("Transitions require one-sided instance.")
    normalized, sign = instance.normalize_to_right_side()
    locator = SegmentLocator(normalized.segments)
    csets = build_candidate_sets_one_side(instance)

    # Work in normalized coordinate system for case logic.
    c_norm = sorted(_map_to_normalized_x(x, sign) for x in csets.union)
    c_norm = unique_sorted_eps(c_norm)
    c_norm_set = {_round_key(x) for x in c_norm}

    transitions_norm: dict[float, CandidateTransition] = {}
    for ck in c_norm:
        loc_k = locator.locate(ck)
        if loc_k.kind != "on_segment" or loc_k.segment_index is None:
            raise RuntimeError(f"Candidate point {ck} is not on any segment.")
        jk = loc_k.segment_index
        left = solve_maximal_left_endpoint(ck, normalized.h, normalized.L)
        if left <= normalized.segments[0].a + EPS:
            transitions_norm[ck] = CandidateTransition(
                ck=ck,
                jk=jk,
                maximal_left=left,
                case=1,
                j_prime=0,
                c_prime=normalized.segments[0].a,
            )
            continue
        loc_left = locator.locate(left)
        if loc_left.kind == "in_gap":
            if loc_left.right_segment_index is None:
                raise RuntimeError("Unexpected gap location with no right segment.")
            j_prime = loc_left.right_segment_index
            c_prime = normalized.segments[j_prime].a
            transitions_norm[ck] = CandidateTransition(
                ck=ck,
                jk=jk,
                maximal_left=left,
                case=2,
                j_prime=j_prime,
                c_prime=c_prime,
            )
            continue
        if loc_left.kind == "on_segment" and loc_left.segment_index is not None:
            j_prime = loc_left.segment_index
            c_prime = left
            case = 3 if _round_key(c_prime) in c_norm_set else 2
            transitions_norm[ck] = CandidateTransition(
                ck=ck,
                jk=jk,
                maximal_left=left,
                case=case,
                j_prime=j_prime,
                c_prime=c_prime,
            )
            continue
        raise RuntimeError("Unexpected transition classification.")

    if sign == 1:
        transitions = transitions_norm
        union = tuple(c_norm)
    else:
        union = tuple(sorted(_map_from_normalized_x(x, sign) for x in c_norm))
        transitions = {}
        for ck_norm, tr in transitions_norm.items():
            ck = _map_from_normalized_x(ck_norm, sign)
            # Segment indexes remain in normalized order; map to original order.
            n = len(normalized.segments)
            jk = (n - 1) - tr.jk
            j_prime = (n - 1) - tr.j_prime
            transitions[ck] = CandidateTransition(
                ck=ck,
                jk=jk,
                maximal_left=_map_from_normalized_x(tr.maximal_left, sign),
                case=tr.case,
                j_prime=j_prime,
                c_prime=_map_from_normalized_x(tr.c_prime, sign),
            )
    return union, transitions
