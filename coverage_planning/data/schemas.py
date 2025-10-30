from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from coverage_planning.algs.geometry import tour_length
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM

SCHEMA_VERSION = "1.0"
HASH_PRECISION = 12


def _round_float(value: float, precision: int = HASH_PRECISION) -> float:
    rounded = round(value, precision)
    if abs(rounded) < 10.0 ** (-precision):
        return 0.0
    # Coerce -0.0 to +0.0 for stability
    if rounded == 0.0:
        return 0.0
    return rounded


def _canonical_segments(
    segments: Sequence[Tuple[float, float]],
    precision: int = HASH_PRECISION,
) -> Tuple[Tuple[float, float], ...]:
    canon = tuple(sorted((_round_float(a, precision), _round_float(b, precision)) for a, b in segments))
    for idx, (a, b) in enumerate(canon):
        if not a < b - EPS_GEOM:
            raise ValueError(f"segment #{idx} degenerates: ({a}, {b})")
        if idx > 0:
            prev_a, prev_b = canon[idx - 1]
            if prev_b > a + EPS_GEOM:
                raise ValueError("segments must be disjoint in canonical form")
    return canon


def canonical_instance_payload(instance: "Instance", precision: int = HASH_PRECISION) -> Dict[str, Any]:
    return {
        "segments": [[_round_float(a, precision), _round_float(b, precision)] for a, b in instance.segments],
        "h": _round_float(instance.h, precision),
        "L": _round_float(instance.L, precision),
    }


def compute_instance_hash(
    segments: Sequence[Tuple[float, float]],
    h: float,
    L: float,
    precision: int = HASH_PRECISION,
) -> str:
    payload = {
        "segments": [[_round_float(a, precision), _round_float(b, precision)] for a, b in segments],
        "h": _round_float(h, precision),
        "L": _round_float(L, precision),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True, slots=True)
class Instance:
    segments: Tuple[Tuple[float, float], ...]
    h: float
    L: float

    def __post_init__(self) -> None:
        canon = tuple(sorted(tuple(seg) for seg in self.segments))
        for idx, (a, b) in enumerate(canon):
            if not a < b - EPS_GEOM:
                raise ValueError(f"segment #{idx} has non-positive measure: ({a}, {b})")
            if idx > 0 and canon[idx - 1][1] > a + EPS_GEOM:
                raise ValueError("segments must be disjoint; merge-touch handled upstream")
        object.__setattr__(self, "segments", canon)

    def canonical_segments(self, precision: int = HASH_PRECISION) -> Tuple[Tuple[float, float], ...]:
        return _canonical_segments(self.segments, precision=precision)


@dataclass(frozen=True, slots=True)
class GoldLabel:
    tours: Tuple[Tuple[float, float], ...] | None
    cost: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tours is not None:
            tours_tuple = tuple(tuple(tour) for tour in self.tours)
        else:
            tours_tuple = None
        object.__setattr__(self, "tours", tours_tuple)
        if not math.isfinite(self.cost):
            raise ValueError("gold cost must be finite")


@dataclass(frozen=True, slots=True)
class NearOptimalLabel:
    tours: Tuple[Tuple[float, float], ...] | None
    cost: float
    gap_pct: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tours is not None:
            tours_tuple = tuple(tuple(tour) for tour in self.tours)
        else:
            tours_tuple = None
        object.__setattr__(self, "tours", tours_tuple)
        if self.gap_pct < -TOL_NUM:
            raise ValueError("gap_pct must be non-negative")


@dataclass(frozen=True, slots=True)
class CandidateSetMeta:
    count: int
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("candidate count cannot be negative")
        object.__setattr__(self, "tags", tuple(self.tags))


def _default_python_version() -> str:
    return platform.python_version()


def _default_commit() -> str:
    return os.environ.get("COVERAGE_PLANNING_COMMIT", "UNKNOWN")


@dataclass(frozen=True, slots=True)
class Sample:
    instance: Instance
    gold: GoldLabel
    split_tag: str = "unspecified"
    seed: int = 0
    code_commit: str = field(default_factory=_default_commit)
    python_version: str = field(default_factory=_default_python_version)
    near_opt: Tuple[NearOptimalLabel, ...] = field(default_factory=tuple)
    candidate_meta: CandidateSetMeta | None = None
    schema_version: str = SCHEMA_VERSION
    hash_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "near_opt", tuple(self.near_opt))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version mismatch: {self.schema_version} != {SCHEMA_VERSION}")
        hash_id = compute_instance_hash(
            self.instance.segments,
            self.instance.h,
            self.instance.L,
            precision=HASH_PRECISION,
        )
        object.__setattr__(self, "hash_id", hash_id)


def sample_to_dict(sample: Sample) -> Dict[str, Any]:
    def tours_to_list(tours: Tuple[Tuple[float, float], ...] | None) -> List[List[float]] | None:
        if tours is None:
            return None
        return [[float(a), float(b)] for a, b in tours]

    return {
        "schema_version": sample.schema_version,
        "code_commit": sample.code_commit,
        "python_version": sample.python_version,
        "split_tag": sample.split_tag,
        "seed": sample.seed,
        "hash_id": sample.hash_id,
        "instance": {
            "segments": [[float(a), float(b)] for a, b in sample.instance.segments],
            "h": float(sample.instance.h),
            "L": float(sample.instance.L),
        },
        "gold": {
            "tours": tours_to_list(sample.gold.tours),
            "cost": float(sample.gold.cost),
            "meta": sample.gold.meta,
        },
        "near_opt": [
            {
                "tours": tours_to_list(opt.tours),
                "cost": float(opt.cost),
                "gap_pct": float(opt.gap_pct),
                "meta": opt.meta,
            }
            for opt in sample.near_opt
        ],
        "candidate_meta": None
        if sample.candidate_meta is None
        else {"count": sample.candidate_meta.count, "tags": list(sample.candidate_meta.tags)},
    }


def validate_sample(sample: Sample) -> None:
    segments = sample.instance.segments
    h = sample.instance.h
    L = sample.instance.L

    # Ensure canonical hash matches stored hash.
    expected_hash = compute_instance_hash(segments, h, L, precision=HASH_PRECISION)
    if expected_hash != sample.hash_id:
        raise ValueError("hash_id mismatch")

    # Geometry feasibility checks.
    tours = sample.gold.tours
    if tours is not None:
        for idx, (p, q) in enumerate(tours):
            length = tour_length(p, q, h)
            if length > L + TOL_NUM:
                raise ValueError(f"tour #{idx} exceeds battery budget: length={length}, L={L}")
        for seg_idx, (a, b) in enumerate(segments):
            pieces: List[Tuple[float, float]] = []
            for p, q in tours:
                lo, hi = (p, q) if p <= q else (q, p)
                if hi < a - EPS_GEOM or lo > b + EPS_GEOM:
                    continue
                pieces.append((max(lo, a), min(hi, b)))
            if not pieces:
                raise ValueError(f"segment #{seg_idx} is not covered by gold tours")
            pieces.sort()
            coverage_start, coverage_end = pieces[0]
            if coverage_start > a + EPS_GEOM:
                raise ValueError(f"segment #{seg_idx} is not covered by gold tours")
            for start, end in pieces[1:]:
                if start > coverage_end + EPS_GEOM:
                    raise ValueError(f"segment #{seg_idx} is not covered by gold tours")
                coverage_end = max(coverage_end, end)
            if coverage_end < b - EPS_GEOM:
                raise ValueError(f"segment #{seg_idx} is not covered by gold tours")

    # Near-optimal labels must respect the same tolerance envelopes.
    for idx, opt in enumerate(sample.near_opt):
        if opt.tours is not None:
            for tour_idx, (p, q) in enumerate(opt.tours):
                if tour_length(p, q, h) > L + TOL_NUM:
                    raise ValueError(
                        f"near-opt label #{idx} tour #{tour_idx} exceeds battery limit"
                    )
        if opt.gap_pct < -TOL_NUM:
            raise ValueError("near-opt gap must be non-negative within tolerance")


__all__ = [
    "SCHEMA_VERSION",
    "HASH_PRECISION",
    "Instance",
    "GoldLabel",
    "NearOptimalLabel",
    "CandidateSetMeta",
    "Sample",
    "canonical_instance_payload",
    "compute_instance_hash",
    "sample_to_dict",
    "validate_sample",
]
