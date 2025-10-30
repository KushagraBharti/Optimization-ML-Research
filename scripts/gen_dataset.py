#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from coverage_planning.common.constants import DEFAULT_SEED, seed_everywhere
from coverage_planning.data.gen_instances import FamilyConfig, draw_family
from coverage_planning.data.io_utils import (
    canonicalize_and_hash,
    write_manifest,
    write_samples_jsonl,
    write_samples_parquet,
)
from coverage_planning.data.labelers import label_gold, label_near_optimal, make_sample
from coverage_planning.data.schemas import Sample
from eval.metrics import candidate_size_summary

SHARD_CAPACITY = 50_000


@dataclass
class DatasetStats:
    counts_per_split: Dict[str, int]
    family_counts: Counter
    bucket_counts: Counter
    bridge_counts: Counter
    dp_candidate_sizes: List[int]
    near_opt_counts: List[int]
    failures: Counter
    skipped: Counter

    def __init__(self, splits: Iterable[str]) -> None:
        self.counts_per_split = {split: 0 for split in splits}
        self.family_counts = Counter()
        self.bucket_counts = Counter()
        self.bridge_counts = Counter()
        self.dp_candidate_sizes = []
        self.near_opt_counts = []
        self.failures = Counter()
        self.skipped = Counter()

    def record_sample(self, split: str, family: str, gold_meta: Dict[str, object], near_opt_count: int) -> None:
        self.counts_per_split[split] += 1
        self.family_counts[family] += 1
        for tag in gold_meta.get("bucket_tags", []):
            if isinstance(tag, str):
                self.bucket_counts[tag] += 1
        dp_meta = gold_meta.get("dp_meta", {})
        if isinstance(dp_meta, dict):
            total_candidates = sum(int(dp_meta.get(key, 0)) for key in ("C_left", "C_right", "C_tail"))
            if total_candidates > 0:
                self.dp_candidate_sizes.append(total_candidates)
            mode = dp_meta.get("best_mode")
            if isinstance(mode, str):
                self.bridge_counts[mode] += 1
        self.near_opt_counts.append(near_opt_count)


def parse_range(value: str, *, expected: int) -> Tuple[float, ...]:
    parts = [float(v.strip()) for v in value.split(",") if v.strip()]
    if len(parts) != expected:
        raise argparse.ArgumentTypeError(f"expected {expected} comma-separated values, got {value!r}")
    return tuple(parts)


def build_family_config(args: argparse.Namespace) -> FamilyConfig:
    h_lo, h_hi = args.h_range
    side_mix = args.side_mix
    return FamilyConfig(
        min_gap=args.min_gap,
        min_len=args.min_len,
        max_len=args.max_len,
        h_range=(h_lo, h_hi),
        L_mode=args.L_mode,
        side_mix=side_mix,
        tight_probability=args.tight_probability,
        use_extrapolation=args.use_extrapolation,
    )


def assign_split(hash_id: str, thresholds: List[Tuple[str, float]]) -> str:
    value = int(hash_id[:16], 16) / float(0xFFFFFFFFFFFFFFFF)
    for split, threshold in thresholds:
        if value <= threshold:
            return split
    return thresholds[-1][0]


def flush_shard(
    out_dir: Path,
    split: str,
    shard_index: int,
    samples: List[Sample],
    *,
    fmt: str,
) -> None:
    if not samples:
        return
    filename = f"{split}_{shard_index:03d}.{ 'parquet' if fmt == 'parquet' else 'jsonl' }"
    target = out_dir / filename
    if fmt == "parquet":
        write_samples_parquet(target, samples)
    else:
        write_samples_jsonl(target, samples)
    samples.clear()


def compute_thresholds(targets: Dict[str, int]) -> List[Tuple[str, float]]:
    total = sum(targets.values())
    if total <= 0:
        raise ValueError("Total target count must be positive")
    cumulative = 0.0
    thresholds = []
    for split, count in targets.items():
        cumulative += count / total
        thresholds.append((split, cumulative))
    thresholds[-1] = (thresholds[-1][0], 1.0)
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate covering-plan dataset with provenance.")
    parser.add_argument("--out", required=True, help="Output directory for dataset shards and manifest.")
    parser.add_argument("--n_train", type=int, default=50000)
    parser.add_argument("--n_test", type=int, default=5000)
    parser.add_argument("--n_shifted", type=int, default=5000)
    parser.add_argument("--n_extrap", type=int, default=5000)
    parser.add_argument("--n_stress", type=int, default=2000)
    parser.add_argument("--families", type=str, default="uniform,clustered,step_gap,straddlers")
    parser.add_argument("--objective", choices=["min_length", "min_tours"], default="min_length")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--near_k", type=int, default=3)
    parser.add_argument("--near_gap_pct", type=float, default=0.03)
    parser.add_argument("--format", choices=["jsonl", "parquet"], default=None)
    parser.add_argument("--min_gap", type=float, default=0.5)
    parser.add_argument("--min_len", type=float, default=1.0)
    parser.add_argument("--max_len", type=float, default=40.0)
    parser.add_argument("--h_range", type=str, default="5.0,40.0", metavar="LO,HI")
    parser.add_argument("--L_mode", choices=["tight", "roomy", "mixed"], default="mixed")
    parser.add_argument("--side_mix", type=str, default="0.4,0.4,0.2", metavar="ONE,TWO,STRAD")
    parser.add_argument("--tight_probability", type=float, default=0.5)
    parser.add_argument("--use_extrapolation", action="store_true")
    parser.add_argument("--workers", type=int, default=0, help="reserved for future use; generation is sequential.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.h_range = parse_range(args.h_range, expected=2)
    args.side_mix = parse_range(args.side_mix, expected=3)
    families = [name.strip() for name in args.families.split(",") if name.strip()]
    if not families:
        raise ValueError("At least one family must be specified")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt is None:
        try:  # prefer parquet when available
            import pyarrow  # noqa: F401

            fmt = "parquet"
        except ImportError:
            fmt = "jsonl"
    if fmt == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pyarrow is required for parquet output") from exc

    targets = {
        "train": args.n_train,
        "test": args.n_test,
        "shifted": args.n_shifted,
        "extrap": args.n_extrap,
        "stress": args.n_stress,
    }
    thresholds = compute_thresholds(targets)

    seed_everywhere(args.seed)
    rng = np.random.default_rng(args.seed)
    family_config = build_family_config(args)

    stats = DatasetStats(targets.keys())
    buffers: Dict[str, List[Sample]] = defaultdict(list)
    shard_indexes = {split: 0 for split in targets}
    seen_hashes: set[str] = set()

    family_cycle = cycle(families)
    total_target = sum(targets.values())
    attempts = 0
    max_attempts = total_target * 100

    while any(stats.counts_per_split[split] < targets[split] for split in targets):
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Maximum attempts exceeded while attempting to fill splits")

        family = next(family_cycle)
        try:
            instance = draw_family(family, family_config, rng)
        except Exception:
            stats.failures["generation"] += 1
            continue

        hash_id = canonicalize_and_hash(instance.segments, instance.h, instance.L)
        split = assign_split(hash_id, thresholds)
        if stats.counts_per_split[split] >= targets[split]:
            stats.skipped["full_split"] += 1
            continue
        if hash_id in seen_hashes:
            stats.skipped["duplicate"] += 1
            continue

        try:
            gold = label_gold(instance, objective=args.objective, family=family)
            near_opt = label_near_optimal(
                instance,
                gold,
                objective=args.objective,
                k=args.near_k,
                max_gap_pct=args.near_gap_pct,
                rng=rng,
            )
            sample = make_sample(instance, gold, near_opt, split_tag=split, seed=args.seed)
        except Exception:
            stats.failures["labeling"] += 1
            continue

        seen_hashes.add(sample.hash_id)
        stats.record_sample(split, family, gold.meta, len(near_opt))
        buffers[split].append(sample)

        if len(buffers[split]) >= SHARD_CAPACITY:
            flush_shard(out_dir, split, shard_indexes[split], buffers[split], fmt=fmt)
            shard_indexes[split] += 1

    for split, buffer in buffers.items():
        flush_shard(out_dir, split, shard_indexes[split], buffer, fmt=fmt)

    manifest = {
        "counts_per_split": stats.counts_per_split,
        "family_counts": dict(stats.family_counts),
        "bucket_counts": dict(stats.bucket_counts),
        "bridge_counts": dict(stats.bridge_counts),
        "dp_candidate_summary": candidate_size_summary(stats.dp_candidate_sizes),
        "near_opt_mean": float(np.mean(stats.near_opt_counts)) if stats.near_opt_counts else 0.0,
        "failures": dict(stats.failures),
        "skipped": dict(stats.skipped),
        "config": {
            "families": families,
            "objective": args.objective,
            "seed": args.seed,
            "near_k": args.near_k,
            "near_gap_pct": args.near_gap_pct,
            "format": fmt,
            "family_config": family_config.__dict__,
        },
    }

    write_manifest(out_dir, manifest)

    print("Generation complete.")
    for split, count in stats.counts_per_split.items():
        print(f"  {split}: {count}")
    print(f"Failures: {dict(stats.failures)}")
    if stats.dp_candidate_sizes:
        summary = candidate_size_summary(stats.dp_candidate_sizes)
        print(
            "Mean |C|: "
            f"{summary['mean']:.2f} (p95={summary['p95']:.2f}, max={summary['max']:.2f})"
        )


if __name__ == "__main__":
    main()
