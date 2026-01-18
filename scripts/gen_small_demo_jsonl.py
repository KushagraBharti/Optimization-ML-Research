#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from coverage_planning.common.constants import DEFAULT_SEED, seed_everywhere
from coverage_planning.data.gen_instances import FamilyConfig, draw_family
from coverage_planning.data.io_utils import write_samples_jsonl
from coverage_planning.data.labelers import label_gold, label_near_optimal, make_sample
from coverage_planning.data.schemas import Sample

DEFAULT_FAMILIES: Tuple[str, ...] = ("uniform", "clustered", "step_gap", "straddlers")
OUTPUT_MIN_LENGTH = Path("data/demo_raw_minlength.jsonl")
OUTPUT_MIN_TOURS = Path("data/demo_raw_mintours.jsonl")


def _build_default_family_config() -> FamilyConfig:
    return FamilyConfig(
        min_gap=0.5,
        min_len=1.0,
        max_len=40.0,
        h_range=(5.0, 40.0),
        L_mode="mixed",
        side_mix=(0.4, 0.4, 0.2),
        tight_probability=0.5,
        use_extrapolation=False,
    )


def _summarise(samples: Sequence[Sample]) -> Tuple[Counter, Counter]:
    family_counts: Counter = Counter()
    bucket_counts: Counter = Counter()
    for sample in samples:
        meta = sample.gold.meta if isinstance(sample.gold.meta, dict) else {}
        family = meta.get("family")
        if isinstance(family, str):
            family_counts[family] += 1
        for tag in meta.get("bucket_tags", []):
            if isinstance(tag, str):
                bucket_counts[tag] += 1
    return family_counts, bucket_counts


def _generate_samples(
    *,
    objective: str,
    families: Iterable[str],
    count: int,
    rng: np.random.Generator,
    seed: int,
) -> List[Sample]:
    config = _build_default_family_config()
    produced: List[Sample] = []
    seen_hashes: set[str] = set()
    family_iter = cycle(list(families))
    attempts = 0
    max_attempts = max(count * 50, 1000)

    while len(produced) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to generate {count} samples for {objective} after {attempts} attempts"
            )

        family = next(family_iter)
        try:
            instance = draw_family(family, config, rng)
        except Exception:
            continue

        try:
            gold = label_gold(instance, objective=objective, family=family)
            near_opt = label_near_optimal(
                instance,
                gold,
                objective=objective,
                rng=rng,
            )
            sample = make_sample(
                instance,
                gold,
                near_opt,
                split_tag=f"demo_{objective}",
                seed=seed,
            )
        except Exception:
            continue

        if sample.hash_id in seen_hashes:
            continue

        seen_hashes.add(sample.hash_id)
        produced.append(sample)

    return produced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate small raw JSONL demo datasets for milestone inspection."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for RNG and Sample records.")
    parser.add_argument("--count", type=int, default=200, help="Samples per objective.")
    parser.add_argument(
        "--families",
        type=str,
        default=",".join(DEFAULT_FAMILIES),
        help="Comma separated family names to cycle through.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    family_list = [name.strip() for name in args.families.split(",") if name.strip()]
    if not family_list:
        raise ValueError("No valid families supplied.")

    seed_everywhere(args.seed)
    rng = np.random.default_rng(args.seed)

    min_length_samples = _generate_samples(
        objective="min_length",
        families=family_list,
        count=args.count,
        rng=rng,
        seed=args.seed,
    )
    min_tours_samples = _generate_samples(
        objective="min_tours",
        families=family_list,
        count=args.count,
        rng=rng,
        seed=args.seed,
    )

    write_samples_jsonl(OUTPUT_MIN_LENGTH, min_length_samples)
    write_samples_jsonl(OUTPUT_MIN_TOURS, min_tours_samples)

    min_length_families, min_length_buckets = _summarise(min_length_samples)
    min_tours_families, min_tours_buckets = _summarise(min_tours_samples)

    print("Demo raw datasets generated:")
    print(f"- {OUTPUT_MIN_LENGTH}: {len(min_length_samples)} samples")
    if min_length_families:
        preview = ", ".join(f"{fam}:{cnt}" for fam, cnt in min_length_families.most_common(4))
        print(f"  Families: {preview}")
    if min_length_buckets:
        preview = ", ".join(f"{tag}:{cnt}" for tag, cnt in min_length_buckets.most_common(4))
        print(f"  Bucket tags: {preview}")

    print(f"- {OUTPUT_MIN_TOURS}: {len(min_tours_samples)} samples")
    if min_tours_families:
        preview = ", ".join(f"{fam}:{cnt}" for fam, cnt in min_tours_families.most_common(4))
        print(f"  Families: {preview}")
    if min_tours_buckets:
        preview = ", ".join(f"{tag}:{cnt}" for tag, cnt in min_tours_buckets.most_common(4))
        print(f"  Bucket tags: {preview}")


if __name__ == "__main__":
    main()
