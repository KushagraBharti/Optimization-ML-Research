from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import pytest

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.algs.reference import dp_full, dpos, gs, gsp
from coverage_planning.common.constants import EPS_GEOM, RNG_SEEDS, TOL_NUM, seed_everywhere


seed_everywhere(RNG_SEEDS["tests"])


def _contiguous_segment_indices(
    segments: Sequence[Tuple[float, float]],
    start_idx: int,
    p: float,
    q: float,
) -> List[int]:
    covered: List[int] = []
    for idx in range(start_idx, len(segments)):
        a, b = segments[idx]
        if a < p - EPS_GEOM:
            continue
        if b > q + EPS_GEOM:
            break
        covered.append(idx)
    return covered


def test_gs_mask_contracts() -> None:
    segments = [
        (-9.0, -6.5),
        (-5.0, -3.0),
        (-1.5, -0.2),
        (0.5, 1.2),
        (2.0, 3.4),
        (5.0, 7.2),
    ]
    h = 2.8
    L = tour_length(segments[0][0], segments[-1][1], h) * 1.1

    count, tours = gs(segments, h=h, L=L)
    assert count == len(tours)

    uncovered_idx = 0
    for tour in tours:
        p, q = tour
        length = tour_length(p, q, h)
        assert length <= L + TOL_NUM

        covered_indices = _contiguous_segment_indices(segments, uncovered_idx, p, q)
        assert covered_indices, "tour must cover at least one new segment"

        # Mask recall: tour must start at the left boundary of the first uncovered segment.
        assert math.isclose(p, segments[uncovered_idx][0], abs_tol=EPS_GEOM)

        # Maximality: q should be the furthest segment endpoint reachable under L.
        expected_q = segments[covered_indices[0]][1]
        for idx in covered_indices:
            candidate_q = segments[idx][1]
            if tour_length(p, candidate_q, h) <= L + TOL_NUM:
                expected_q = candidate_q
            else:
                break
        assert math.isclose(q, expected_q, abs_tol=5 * TOL_NUM)

        uncovered_idx = covered_indices[-1] + 1

    assert uncovered_idx == len(segments)


def test_gsp_mask_case_consistency() -> None:
    seg = (-6.5, 6.0)
    h = 2.5

    tight_L = 2.0 * math.hypot(max(abs(seg[0]), abs(seg[1])), h)
    sweep = [tight_L * (1.0 + delta) for delta in (-1e-6, 0.0, 1e-6, 1e-4, 1e-3)]

    for L in sweep:
        try:
            count, tours = gsp(seg, h=h, L=L)
        except ValueError:
            continue


        assert tours, "at least one tour must be returned"
        for tour in tours:
            p, q = tour
            assert tour_length(p, q, h) <= L + TOL_NUM

        if count == 1:
            p, q = tours[0]
            assert p <= 0.0 + EPS_GEOM and q >= 0.0 - EPS_GEOM
        elif count == 2:
            left, right = sorted(tours, key=lambda t: t[0])
            assert abs(left[1]) <= EPS_GEOM
            assert abs(right[0]) <= EPS_GEOM
        else:  # pragma: no cover - unexpected output
            pytest.fail("GSP returned more than two tours")


def test_dpos_candidate_legality() -> None:
    segments = [
        (0.0, 1.2),
        (2.0, 2.8),
        (4.2, 5.1),
        (7.0, 7.6),
    ]
    h = 1.5
    L = max(tour_length(a, b, h) for a, b in segments) * 1.2
    debug: Dict[str, object] = {}
    Sigma, candidates = dpos(segments, h=h, L=L, debug=debug)

    assert debug["candidate_count"] == len(candidates)
    assert Sigma[-1] > 0.0

    for c in candidates:
        p = find_maximal_p(c, h, L)
        length = tour_length(p, c, h)
        assert length <= L + TOL_NUM



def test_dp_full_bridge_legality() -> None:
    segments = [
        (-8.0, -6.2),
        (-4.0, -2.8),
        (1.0, 2.2),
        (3.5, 4.6),
    ]
    h = 2.0
    L = max(tour_length(a, b, h) for a, b in segments) * 1.25
    debug: Dict[str, object] = {}
    cost, tours = dp_full(segments, h=h, L=L, debug=debug)

    assert tours == []
    assert cost > 0.0

    left_dbg = debug["left"]
    right_dbg = debug["right"]
    tail_dbg = debug["tail"]

    assert left_dbg["candidate_count"] > 0
    assert right_dbg["candidate_count"] > 0
    assert tail_dbg["candidate_count"] > 0

    left_candidates = left_dbg["candidates"]
    tail_candidates = tail_dbg["candidates"]

    bridge_checks = debug["bridge_checked_pairs"]
    min_expected = max(1, len(left_candidates) * len(tail_candidates) // 4)
    assert bridge_checks >= min_expected

    feasible = any(
        tour_length(-P, q, h) <= L + TOL_NUM
        for P in left_candidates
        for q in tail_candidates
    )
    if bridge_checks > 0:
        assert feasible
