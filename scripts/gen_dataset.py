#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, random
from typing import List
from dataclasses import asdict

from coverage_planning.data.gen_instances import GenConfig, sample_instance
from coverage_planning.data.solvers import SolverProvider
from coverage_planning.data.schemas import Instance, LabeledExample
from coverage_planning.data.labelers import (
    label_min_tours, label_min_length_one_side, label_min_length_full_line
)
from coverage_planning.data.io_utils import write_jsonl_gz

def main():
    ap = argparse.ArgumentParser(description="Generate coverage-planning dataset")
    ap.add_argument("--out", type=str, required=True, help="output .jsonl.gz path")
    ap.add_argument("--count", type=int, default=5000, help="#instances")
    ap.add_argument("--objective", type=str, choices=["MinTours","MinLength_OneSide","MinLength_FullLine"], default="MinLength_FullLine")
    ap.add_argument("--solver", type=str, choices=["heuristic","reference"], default="heuristic")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)

    provider = SolverProvider(family=args.solver)
    cfg = GenConfig(seed=None)  # we randomize per-instance

    def label_one() -> LabeledExample:
        segs, h, L = sample_instance(cfg)
        inst = Instance(segments=segs, h=h, L=L, seed=None, meta={})
        
        if args.objective == "MinTours":
            return label_min_tours(inst, provider)
        elif args.objective == "MinLength_OneSide":
            # ensure we’re really one-sided
            segs_pos = [(a,b) for (a,b) in segs if a >= 0 and b >= 0]
            inst = Instance(segments=segs_pos, h=h, L=L, seed=None, meta={"filtered_to_one_side": True})
            return label_min_length_one_side(inst, provider)
        else:
            return label_min_length_full_line(inst, provider)

    records: List[LabeledExample] = [label_one() for _ in range(args.count)]
    write_jsonl_gz(args.out, records)
    print(f"Wrote {args.count} records to {args.out} using solver={args.solver} objective={args.objective}")

if __name__ == "__main__":
    main()
