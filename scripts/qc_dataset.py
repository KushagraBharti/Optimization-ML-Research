#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from coverage_planning.algs.geometry import tour_length
from coverage_planning.algs.reference import dpos, gs, gsp
from coverage_planning.algs.reference.dp_full_line_ref import dp_full_line_ref
from coverage_planning.common.constants import EPS_GEOM, TOL_NUM
from coverage_planning.data.schemas import (
    GoldLabel,
    Instance,
    NearOptimalLabel,
    Sample,
    validate_sample,
)
from eval.metrics import candidate_size_summary

try:
    import pyarrow.parquet as pq  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pq = None  # type: ignore


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_parquet(path: Path) -> List[Dict[str, object]]:
    if pq is None:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required to read parquet files")
    table = pq.read_table(path)
    return table.to_pylist()  # type: ignore[no-any-return]


def dict_to_sample(data: Dict[str, object]) -> Sample:
    inst_data = data["instance"]
    instance = Instance(
        segments=tuple(tuple(seg) for seg in inst_data["segments"]),
        h=float(inst_data["h"]),
        L=float(inst_data["L"]),
    )
    gold_data = data["gold"]
    tours = gold_data["tours"]
    gold = GoldLabel(
        tours=None if tours is None else tuple(tuple(t) for t in tours),
        cost=float(gold_data["cost"]),
        meta=dict(gold_data.get("meta", {})),
    )
    near_list = []
    for item in data.get("near_opt", []):
        tours_opt = item["tours"]
        near_list.append(
            NearOptimalLabel(
                tours=None if tours_opt is None else tuple(tuple(t) for t in tours_opt),
                cost=float(item["cost"]),
                gap_pct=float(item.get("gap_pct", 0.0)),
                meta=dict(item.get("meta", {})),
            )
        )
    return Sample(
        instance=instance,
        gold=gold,
        near_opt=tuple(near_list),
        split_tag=str(data.get("split_tag", "unknown")),
        seed=int(data.get("seed", 0)),
        code_commit=str(data.get("code_commit", "UNKNOWN")),
        python_version=str(data.get("python_version", "UNKNOWN")),
    )


def load_samples(directory: Path) -> List[Sample]:
    samples: List[Sample] = []
    for path in sorted(directory.iterdir()):
        if path.suffix == ".jsonl":
            records = load_jsonl(path)
        elif path.suffix == ".parquet":
            records = load_parquet(path)
        elif path.name == "manifest.json":
            continue
        else:
            continue
        for record in records:
            sample = dict_to_sample(record)
            validate_sample(sample)
            samples.append(sample)
    if not samples:
        raise RuntimeError(f"No dataset files found in {directory}")
    return samples


GRID_EPS = max(EPS_GEOM * 10.0, 1e-8)


def coverage_signature(tours: Sequence[Tuple[float, float]]) -> frozenset[Tuple[int, int]]:
    def quant(x: float) -> int:
        return int(round(x / GRID_EPS))

    pairs = []
    for p, q in tours:
        lo, hi = (p, q) if p <= q else (q, p)
        pairs.append((quant(lo), quant(hi)))
    return frozenset(pairs)


def verify_tours(instance: Instance, tours: Sequence[Tuple[float, float]]) -> None:
    segs = sorted(instance.segments, key=lambda seg: seg[0])
    min_seg = segs[0][0]
    max_seg = segs[-1][1]
    for idx, (p, q) in enumerate(tours):
        length = tour_length(p, q, instance.h)
        if length > instance.L * 1.1:
            raise AssertionError(f"tour #{idx} exceeds battery limit")
        lo, hi = (p, q) if p <= q else (q, p)
        if lo < min_seg - EPS_GEOM - TOL_NUM or hi > max_seg + EPS_GEOM + TOL_NUM:
            raise AssertionError(f"tour #{idx} extends outside hull")

    for a, b in segs:
        coverage: List[Tuple[float, float]] = []
        for p, q in tours:
            lo, hi = (p, q) if p <= q else (q, p)
            if hi < a - EPS_GEOM or lo > b + EPS_GEOM:
                continue
            coverage.append((max(lo, a), min(hi, b)))
        if not coverage:
            raise AssertionError(f"segment [{a}, {b}] not covered")
        coverage.sort()
        start, end = coverage[0]
        if start > a + EPS_GEOM:
            raise AssertionError(f"gap before segment {a}")
        current = end
        for lo, hi in coverage[1:]:
            if lo > current + EPS_GEOM:
                raise AssertionError(f"gap detected within segment [{a}, {b}]")
            current = max(current, hi)
        if current < b - EPS_GEOM:
            raise AssertionError(f"segment [{a}, {b}] not fully covered")


def recompute_cost(sample: Sample, objective: str) -> float:
    segments = list(sample.instance.segments)
    h = sample.instance.h
    L = sample.instance.L
    if objective == "min_tours":
        count, _ = gs(segments, h, L)
        return float(count)
    if len(segments) == 1:
        _, tours = gsp(segments[0], h, L)
        return sum(tour_length(p, q, h) for p, q in tours)
    cost, _ = dp_full_line_ref(segments, h, L)
    return cost


def check_near_opt(sample: Sample, max_gap: float) -> float:
    gold_signature = coverage_signature(list(sample.gold.tours or []))
    gold_cost = sample.gold.cost
    max_observed = 0.0
    signatures = {gold_signature}
    for idx, label in enumerate(sample.near_opt):
        tours = list(label.tours or [])
        verify_tours(sample.instance, tours)
        total = sum(tour_length(p, q, sample.instance.h) for p, q in tours)
        if abs(total - label.cost) > 1e-6:
            raise AssertionError(f"near-opt #{idx} cost mismatch")
        gap_pct = (label.cost - gold_cost) / max(gold_cost, 1e-9)
        if gap_pct > max_gap + 1e-6:
            raise AssertionError(f"near-opt #{idx} exceeds gap tolerance")
        signature = coverage_signature(tours)
        if signature in signatures:
            raise AssertionError(f"near-opt #{idx} duplicates coverage signature")
        signatures.add(signature)
        max_observed = max(max_observed, gap_pct)
    return max_observed


def infer_objective(sample: Sample, default: str) -> str:
    meta_obj = sample.gold.meta.get("objective")
    if isinstance(meta_obj, str):
        return meta_obj
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality control for generated datasets.")
    parser.add_argument("--path", required=True, help="Dataset directory generated by gen_dataset.py")
    parser.add_argument("--sample_per_bucket", type=int, default=1000)
    parser.add_argument("--objective", choices=["min_length", "min_tours", "auto"], default="auto")
    parser.add_argument("--max_gap_pct", type=float, default=0.03)
    args = parser.parse_args()

    directory = Path(args.path)
    samples = load_samples(directory)
    rng = random.Random(1337)

    split_counts = Counter()
    dp_sizes: List[int] = []
    hash_split: Dict[str, str] = {}
    bucket_counts = Counter()
    max_gap_pct = 0.0
    for sample in samples:
        split = sample.split_tag
        if sample.hash_id in hash_split and hash_split[sample.hash_id] != split:
            raise AssertionError(f"hash collision across splits: {sample.hash_id}")
        hash_split[sample.hash_id] = split
        split_counts[split] += 1

        objective = infer_objective(sample, args.objective if args.objective != "auto" else "min_length")
        tours = list(sample.gold.tours or [])
        if tours:
            verify_tours(sample.instance, tours)
            recomputed = sum(tour_length(p, q, sample.instance.h) for p, q in tours)
            if abs(recomputed - sample.gold.cost) > 1e-6:
                raise AssertionError("gold cost mismatch with tours")
        if sample.near_opt:
            observed = check_near_opt(sample, args.max_gap_pct)
            max_gap_pct = max(max_gap_pct, observed)
        dp_meta = sample.gold.meta.get("dp_meta", {})
        if isinstance(dp_meta, dict):
            total_candidates = sum(int(dp_meta.get(key, 0)) for key in ("C_left", "C_right", "C_tail"))
            if total_candidates > 0:
                dp_sizes.append(total_candidates)

        for tag in sample.gold.meta.get("bucket_tags", []):
            if isinstance(tag, str):
                bucket_counts[tag] += 1

    # Optimality re-check
    per_bucket_samples: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        tags = sample.gold.meta.get("bucket_tags", [])
        if not tags:
            per_bucket_samples["_"].append(sample)
        else:
            for tag in tags:
                if isinstance(tag, str):
                    per_bucket_samples[tag].append(sample)

    for tag, group in per_bucket_samples.items():
        subset = rng.sample(group, min(len(group), args.sample_per_bucket))
        for sample in subset:
            objective = infer_objective(sample, "min_length")
            expected = sample.gold.cost
            recomputed = recompute_cost(sample, objective)
            if abs(recomputed - expected) > 1e-5:
                raise AssertionError(f"Optimality mismatch for hash {sample.hash_id} in bucket {tag}")

    dp_summary = candidate_size_summary(dp_sizes)

    print(f"Split counts: {dict(split_counts)}")
    if dp_summary.get("count", 0) > 0:
        print(
            "|C| stats: "
            f"mean={dp_summary['mean']:.2f}, "
            f"p50={dp_summary['p50']:.2f}, "
            f"p95={dp_summary['p95']:.2f}, "
            f"max={dp_summary['max']:.2f}"
        )
    else:
        print("|C| stats: count=0")
    print("QC PASS")
    print(f"Samples checked: {len(samples)}")
    print(f"Buckets: {dict(bucket_counts)}")
    print(f"Max near-opt gap observed: {max_gap_pct:.5f}")


if __name__ == "__main__":
    main()
