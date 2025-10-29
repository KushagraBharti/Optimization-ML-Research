from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Tuple

from coverage_planning.common.constants import EPS_GEOM

SideMode = Literal["one_side", "both_sides", "mixed"]


@dataclass
class GenConfig:
    n_min: int = 2
    n_max: int = 100
    x_span: float = 100.0
    h_min: float = 5.0
    h_max: float = 100.0
    L_margin: float = 1.2  # ensure L >= 2h*L_margin sometimes tighter below
    side_mode: SideMode = "mixed"  # control sides of O'
    min_gap: float = 1e-4
    allow_touch_merge: bool = True
    seed: int | None = None


def _disjointify(
    segs: List[Tuple[float, float]], min_gap: float, merge_touches: bool
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for a, b in sorted(segs, key=lambda s: s[0]):
        if merge_touches and out and a <= out[-1][1] + min_gap:
            shift = out[-1][1] + min_gap - a
            a += shift
            b += shift
        if b - a < EPS_GEOM:
            b = a + EPS_GEOM
        out.append((a, b))
    return out


def sample_instance(cfg: GenConfig) -> tuple[list[tuple[float, float]], float, float]:
    if cfg.seed is not None:
        random.seed(cfg.seed)

    n = random.randint(cfg.n_min, cfg.n_max)
    xs = sorted(random.uniform(0.0, cfg.x_span) for _ in range(2 * n))
    segs = [(xs[2 * i], xs[2 * i + 1]) for i in range(n)]

    # place segments by side_mode
    if cfg.side_mode == "one_side":
        segs = [(a, b) for (a, b) in segs]  # keep on right
    elif cfg.side_mode == "both_sides":
        # reflect alternating to the left
        segs = [((-b, -a) if i % 2 == 0 else (a, b)) for i, (a, b) in enumerate(segs)]
    else:  # "mixed"
        segs = [((-b, -a) if random.random() < 0.4 else (a, b)) for (a, b) in segs]

    segs = _disjointify(segs, cfg.min_gap, cfg.allow_touch_merge)

    h = random.uniform(cfg.h_min, cfg.h_max)
    # choose L: sometimes generous, sometimes tight
    # crude lower bound: 2h (p=q=0)
    tight = random.random() < 0.35
    if tight:
        L = 2.05 * h  # near feasibility edge
    else:
        # large enough to cover the convex hull in one tour (often)
        a_min = min(a for a, _ in segs)
        b_max = max(b for _, b in segs)
        from coverage_planning.algs.geometry import tour_length

        L = max(2.05 * h, tour_length(a_min, b_max, h) * 1.05)

    return segs, h, L
