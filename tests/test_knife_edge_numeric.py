from __future__ import annotations

def _length_close(length: float, L: float, delta: float) -> bool:
    tol = max(5 * TOL_NUM, delta * 10)
    return abs(length - L) <= tol


import math
from typing import Dict, Iterable, Tuple

import pytest

from coverage_planning.algs.geometry import find_maximal_p, tour_length
from coverage_planning.algs.reference import dp_full, dpos, gs, gsp
from coverage_planning.common.constants import EPS_GEOM, RNG_SEEDS, TOL_NUM, seed_everywhere


seed_everywhere(RNG_SEEDS["tests"])

DELTAS = [1e-9, 1e-8, 1e-6, 1e-4, 1e-3]


def _thresholds(segments: Iterable[Tuple[float, float]], h: float) -> Iterable[float]:
    for a, b in segments:
        yield tour_length(a, b, h)
        yield 2.0 * math.hypot(a, h)
        yield 2.0 * math.hypot(b, h)


@pytest.mark.parametrize("delta", DELTAS)
def test_gs_knife_edges(delta: float) -> None:
    segments = [(-5.0, -3.0), (0.5, 1.5), (3.0, 4.4)]
    h = 1.8
    for base in _thresholds(segments, h):
        for sign in (-1.0, 1.0):
            L = base + sign * delta
            if L <= 0:
                continue
            try:
                _, tours = gs(segments, h=h, L=L)
            except (ValueError, RuntimeError):
                continue
            for p, q in tours:
                length = tour_length(p, q, h)
                assert length <= L + TOL_NUM
                if length >= L - 1e-4:
                    assert _length_close(length, L, delta)


@pytest.mark.parametrize("delta", DELTAS)
def test_gsp_knife_edges(delta: float) -> None:
    seg = (-7.0, 4.5)
    h = 2.2
    for base in _thresholds([seg], h):
        for sign in (-1.0, 1.0):
            L = base + sign * delta
            if L <= 0:
                continue
            try:
                _, tours = gsp(seg, h=h, L=L)
            except (ValueError, RuntimeError):
                continue
            for p, q in tours:
                length = tour_length(p, q, h)
                assert length <= L + TOL_NUM
                if length >= L - 1e-4:
                    assert _length_close(length, L, delta)


@pytest.mark.parametrize("delta", DELTAS)
def test_dpos_knife_edges(delta: float) -> None:
    segments = [(0.0, 1.1), (2.0, 3.0), (4.5, 5.4)]
    h = 1.2
    for base in _thresholds(segments, h):
        for sign in (-1.0, 1.0):
            L = base + sign * delta
            if L <= 0:
                continue
            debug: Dict[str, object] = {}
            try:
                _, candidates = dpos(segments, h=h, L=L, debug=debug)
            except (ValueError, RuntimeError):
                continue
            for c in candidates:
                p = find_maximal_p(c, h, L)
                length = tour_length(p, c, h)
                assert length <= L + TOL_NUM
                if length >= L - 1e-4:
                    assert _length_close(length, L, delta)


@pytest.mark.parametrize("delta", DELTAS)
def test_dp_full_knife_edges(delta: float) -> None:
    segments = [(-6.0, -4.0), (-2.0, -1.0), (1.0, 2.0), (3.5, 5.0)]
    h = 2.4
    for base in _thresholds(segments, h):
        for sign in (-1.0, 1.0):
            L = base + sign * delta
            if L <= 0:
                continue
            debug: Dict[str, object] = {}
            try:
                cost, _ = dp_full(segments, h=h, L=L, debug=debug)
            except (ValueError, RuntimeError):
                continue
            assert math.isfinite(cost)
            left_candidates = debug["left"].get("candidates", [])
            tail_candidates = debug["tail"].get("candidates", [])
            for info in (debug["left"], debug["right"], debug["tail"]):
                assert info["candidate_count"] >= 0
                assert info["table_size"] >= 0
            if left_candidates and tail_candidates:
                feasible = False
                for P in left_candidates:
                    p_world = -P
                    for q in tail_candidates:
                        length = tour_length(p_world, q, h)
                        if length <= L + TOL_NUM:
                            feasible = True
                            if length >= L - 1e-4:
                                assert _length_close(length, L, delta)
                            break
                    if feasible:
                        break
                if debug["bridge_checked_pairs"] > 0:
                    assert feasible

