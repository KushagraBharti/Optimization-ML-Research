#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from typing import List

from coverage_planning.common.constants import DEFAULT_SEED, seed_everywhere
from coverage_planning.data.gen_instances import GenConfig, sample_instance
from coverage_planning.data.io_utils import write_jsonl_gz
from coverage_planning.data.labelers import (
    label_min_length_full_line,
    label_min_length_one_side,
    label_min_tours,
)
from coverage_planning.data.schemas import Instance, Sample
from coverage_planning.data.solvers import SolverProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coverage-planning dataset")
    parser.add_argument("--out", type=str, required=True, help="output .jsonl.gz path")
    parser.add_argument("--count", type=int, default=5000, help="#instances to sample")
    parser.add_argument(
        "--objective",
        type=str,
        choices=["MinTours", "MinLength_OneSide", "MinLength_FullLine"],
        default="MinLength_FullLine",
    )
    parser.add_argument(
        "--solver", type=str, choices=["heuristic", "reference"], default="heuristic"
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-tag", type=str, default="unspecified")
    args = parser.parse_args()

    seed_everywhere(args.seed)
    random.seed(args.seed)

    provider = SolverProvider(family=args.solver)
    cfg = GenConfig(seed=None)

    def label_one(idx: int) -> Sample:
        segs, h, L = sample_instance(cfg)
        inst = Instance(segments=tuple(segs), h=h, L=L)

        if args.objective == "MinTours":
            return label_min_tours(inst, provider, split_tag=args.split_tag, seed=args.seed + idx)
        if args.objective == "MinLength_OneSide":
            segs_pos = [(a, b) for (a, b) in segs if a >= 0 and b >= 0]
            inst_pos = Instance(segments=tuple(segs_pos), h=h, L=L)
            return label_min_length_one_side(
                inst_pos, provider, split_tag=args.split_tag, seed=args.seed + idx
            )
        return label_min_length_full_line(
            inst, provider, split_tag=args.split_tag, seed=args.seed + idx
        )

    records: List[Sample] = [label_one(i) for i in range(args.count)]
    write_jsonl_gz(args.out, records)
    print(
        f"Wrote {args.count} records to {args.out} "
        f"(solver={args.solver}, objective={args.objective}, split={args.split_tag})"
    )


if __name__ == "__main__":
    main()
