"""Synthetic instance generation utilities for coverage planning datasets.

The dataset scripts rely on two public helpers:

``FamilyConfig`` – declaratively specifies geometric ranges and RNG knobs.
``draw_family`` – samples an :class:`~coverage_planning.data.schemas.Instance`
                 according to the requested family name.

The implementation intentionally favours deterministic, reproducible sampling
paths so that dataset shards can be regenerated exactly when seeded with the
same ``numpy.random.Generator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from coverage_planning.algs.geometry import tour_length
from coverage_planning.data.schemas import Instance

FloatRange = Tuple[float, float]
Segments = List[Tuple[float, float]]

# Default sample counts across families.
MIN_SEGMENTS = 2
MAX_SEGMENTS = 7


@dataclass(frozen=True)
class FamilyConfig:
    """Configuration bundle for instance families."""

    min_gap: float
    min_len: float
    max_len: float
    h_range: FloatRange
    L_mode: str = "mixed"  # tight, roomy, mixed
    side_mix: Tuple[float, float, float] = (0.5, 0.3, 0.2)  # right, left, straddle
    tight_probability: float = 0.5
    use_extrapolation: bool = False
    max_segments: int = MAX_SEGMENTS

    def __post_init__(self) -> None:
        if not (0.0 < self.min_len < self.max_len):
            raise ValueError("min_len must be positive and < max_len")
        if self.min_gap <= 0.0:
            raise ValueError("min_gap must be positive")
        if len(self.side_mix) != 3:
            raise ValueError("side_mix must contain three probabilities")
        total = sum(self.side_mix)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("side_mix must contain finite positive weights")
        if self.max_segments < MIN_SEGMENTS:
            raise ValueError("max_segments must be >= MIN_SEGMENTS")
        if self.h_range[0] <= 0.0 or self.h_range[1] <= self.h_range[0]:
            raise ValueError("h_range must contain positive ascending bounds")

    def normalised_side_mix(self) -> Tuple[float, float, float]:
        total = sum(self.side_mix)
        return tuple(weight / total for weight in self.side_mix)


def _rng_float(rng, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _rng_int(rng, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi))


def _build_monotone_segments(
    count: int,
    *,
    length_range: FloatRange,
    gap_range: FloatRange,
    rng,
) -> Segments:
    segments: Segments = []
    cursor = 0.0
    for idx in range(count):
        length = _rng_float(rng, *length_range)
        start = cursor
        end = start + length
        segments.append((start, end))
        if idx + 1 < count:
            gap = _rng_float(rng, *gap_range)
            cursor = end + gap
    return segments


def _translate(segments: Segments, delta: float) -> Segments:
    return [(a + delta, b + delta) for a, b in segments]


def _reflect_to_left(segments: Segments, *, min_gap: float) -> Segments:
    reflected = [(-b, -a) for a, b in segments]
    reflected.sort(key=lambda seg: seg[0])
    if not reflected:
        return reflected
    terminal = reflected[-1][1]
    offset = terminal + min_gap
    return [(a - offset, b - offset) for a, b in reflected]


def _generate_right_segments(family: str, config: FamilyConfig, rng, count: int) -> Segments:
    min_gap = config.min_gap
    len_lo = config.min_len
    len_hi = config.max_len * (1.2 if config.use_extrapolation else 1.0)

    if family == "uniform":
        gap_hi = min_gap * 2.5
        segments = _build_monotone_segments(
            count,
            length_range=(len_lo, len_hi),
            gap_range=(min_gap, gap_hi),
            rng=rng,
        )
        return segments

    if family == "clustered":
        gap_hi = max(min_gap * 1.2, min_gap + 1e-6)
        local_len_hi = max(len_lo * 1.5, len_lo + 1e-6)
        segments = _build_monotone_segments(
            count,
            length_range=(len_lo * 0.6, min(local_len_hi, len_hi)),
            gap_range=(min_gap * 0.5, gap_hi),
            rng=rng,
        )
        centre = _rng_float(rng, 0.0, config.max_len * 0.5)
        return _translate(segments, centre)

    if family == "step_gap":
        segments = _build_monotone_segments(
            count,
            length_range=(len_lo, len_hi),
            gap_range=(min_gap, min_gap * 1.6),
            rng=rng,
        )
        if count >= 2:
            idx = _rng_int(rng, 1, count)
            boost = min_gap * _rng_float(rng, 3.5, 6.0)
            for i in range(idx, count):
                start, end = segments[i]
                segments[i] = (start + boost, end + boost)
        return segments

    # Default catch-all mirrors uniform behaviour.
    return _generate_right_segments("uniform", config, rng, count)


def _merge_straddle_segments(
    config: FamilyConfig,
    left_segments: Segments,
    right_segments: Segments,
    rng,
) -> Segments:
    if not left_segments:
        return right_segments
    if not right_segments:
        return _reflect_to_left(left_segments, min_gap=config.min_gap)

    left_reflected = _reflect_to_left(left_segments, min_gap=config.min_gap)
    right_shift = right_segments[0][0]
    right_adjusted = _translate(
        right_segments, config.min_gap - right_shift if right_shift > 0.0 else 0.0
    )

    combined = left_reflected + right_adjusted
    combined.sort(key=lambda seg: seg[0])
    return combined


def _choose_orientation(family: str, config: FamilyConfig, rng) -> str:
    if family == "straddlers":
        return "straddle"
    right_w, left_w, straddle_w = config.normalised_side_mix()
    pick = _rng_float(rng, 0.0, 1.0)
    if pick <= right_w:
        return "right"
    if pick <= right_w + left_w:
        return "left"
    return "straddle"


def _sample_segment_count(config: FamilyConfig, rng, *, min_count: int = MIN_SEGMENTS) -> int:
    hi = max(min_count + 1, config.max_segments + 1)
    return _rng_int(rng, min_count, hi)


def _determine_altitude(config: FamilyConfig, rng) -> float:
    return _rng_float(rng, *config.h_range)


def _determine_L(
    segments: Sequence[Tuple[float, float]],
    h: float,
    config: FamilyConfig,
    rng,
) -> float:
    max_atomic = max(tour_length(a, b, h) for a, b in segments)
    farthest = max(max(abs(a), abs(b)) for a, b in segments)
    base_min = 2.0 * math.hypot(farthest, h)

    def tight_multiplier() -> float:
        return _rng_float(rng, 1.02, 1.10)

    def roomy_multiplier() -> float:
        return _rng_float(rng, 1.25, 1.60)

    mode = config.L_mode
    if mode == "mixed":
        mode = "tight" if _rng_float(rng, 0.0, 1.0) < config.tight_probability else "roomy"

    multiplier = tight_multiplier() if mode == "tight" else roomy_multiplier()
    L = max(max_atomic * multiplier, base_min * 1.02)
    return float(L)


def draw_family(name: str, config: FamilyConfig, rng) -> Instance:
    """Sample an instance from ``name`` given ``config`` and RNG ``rng``."""

    family = name.lower().strip()
    if family not in {"uniform", "clustered", "step_gap", "straddlers"}:
        raise ValueError(f"Unknown instance family: {name!r}")

    orientation = _choose_orientation(family, config, rng)
    total_segments = _sample_segment_count(config, rng)

    if orientation == "straddle":
        left_count = max(1, _rng_int(rng, 1, total_segments))
        right_count = max(1, total_segments - left_count)
        left_segments_pos = _generate_right_segments(family, config, rng, left_count)
        right_segments = _generate_right_segments(family, config, rng, right_count)
        segments = _merge_straddle_segments(config, left_segments_pos, right_segments, rng)
    else:
        segments_pos = _generate_right_segments(family, config, rng, total_segments)
        if orientation == "right":
            segments = segments_pos
        else:
            segments = _reflect_to_left(segments_pos, min_gap=config.min_gap)

    h = _determine_altitude(config, rng)
    L = _determine_L(segments, h, config, rng)
    return Instance(segments=tuple(segments), h=h, L=L)
