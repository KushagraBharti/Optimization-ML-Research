from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import pytest

from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import dp_full, dpos
from coverage_planning.common.constants import RNG_SEEDS, seed_everywhere


seed_everywhere(RNG_SEEDS["tests"])


def _gen_disjoint_segments(
    rng: random.Random,
    n: int,
    *,
    x_min: float,
    x_max: float,
    min_len: float,
    min_gap: float,
) -> List[Tuple[float, float]]:
    anchors = sorted(rng.uniform(x_min, x_max) for _ in range(2 * n))
    segments: List[Tuple[float, float]] = []
    cursor = anchors[0]
    for idx in range(n):
        length = rng.uniform(min_len, min_len * 3.0)
        start = max(cursor, x_min)
        end = min(start + length, x_max - min_gap * (n - idx))
        if segments and start < segments[-1][1] + min_gap:
            start = segments[-1][1] + min_gap
            end = start + length
        segments.append((start, end))
        cursor = end + min_gap
    return segments


@pytest.mark.parametrize("n", [5, 10, 20, 40])
def test_dpos_complexity_envelope(n: int) -> None:
    rng = random.Random(10_123)
    instances = 3
    kappa_candidates = 6.0
    kappa_transitions = 8.0
    for _ in range(instances):
        segments = _gen_disjoint_segments(
            rng,
            n,
            x_min=0.0,
            x_max=120.0,
            min_len=0.8,
            min_gap=0.8,
        )
        h = rng.uniform(1.5, 4.5)
        max_atomic = max(tour_length(a, b, h) for a, b in segments)
        L = max_atomic * 1.35
        debug: Dict[str, object] = {}
        _, candidates = dpos(segments, h=h, L=L, debug=debug)
        candidate_count = debug["candidate_count"]
        transitions = debug["transitions_total"]
        assert candidate_count == len(candidates)
        assert candidate_count <= kappa_candidates * n * math.log1p(n)
        assert transitions <= kappa_transitions * max(1, n) * max(1, candidate_count)


@pytest.mark.parametrize("n", [5, 10, 20, 40])
def test_dp_full_complexity_envelope(n: int) -> None:
    rng = random.Random(44_221)
    instances = 3
    for _ in range(instances):
        segments_left = _gen_disjoint_segments(
            rng,
            max(1, n // 2),
            x_min=0.0,
            x_max=60.0,
            min_len=0.7,
            min_gap=0.6,
        )
        segments_right = _gen_disjoint_segments(
            rng,
            n - len(segments_left),
            x_min=0.0,
            x_max=60.0,
            min_len=0.7,
            min_gap=0.6,
        )
        segments = [(-b, -a) for a, b in reversed(segments_left)] + [
            (a, b) for a, b in segments_right
        ]
        h = rng.uniform(1.5, 4.0)
        max_atomic = max(tour_length(a, b, h) for a, b in segments)
        L = max_atomic * 1.25
        debug: Dict[str, object] = {}
        cost, _ = dp_full(segments, h=h, L=L, debug=debug)
        assert math.isfinite(cost)
        left_info = debug["left"]
        right_info = debug["right"]
        tail_info = debug["tail"]
        for info in (left_info, right_info, tail_info):
            assert info["candidate_count"] <= 8.0 * n * math.log1p(n)
            assert info["transitions_total"] <= 16.0 * n * math.log1p(n)
        checked = debug["bridge_checked_pairs"]
        if checked:
            limit = len(left_info.get("candidates", [])) * len(
                tail_info.get("candidates", [])
            )
            assert checked <= max(1, 2 * limit)
