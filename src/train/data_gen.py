# src/train/data_gen.py

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

from src.solvers.greedy import cover_min_tours, tour_length
from src.solvers.dp import cover_min_length

def random_instance(
    n: int,
    max_coord: float = 100.0,
    h: float = 1.0
) -> Tuple[List[Tuple[float, float]], float]:
    # Generate n disjoint segments on [0, max_coord], then choose L around full span.
    coords = sorted(random.uniform(0, max_coord) for _ in range(2 * n))
    raw = [(coords[2*i], coords[2*i+1]) for i in range(n)]
    segments: List[Tuple[float, float]] = []
    last_end = -1e9
    for a, b in sorted(raw):
        if a <= last_end:
            a = last_end + random.random()*(max_coord*0.01 + 1e-3)
        if b <= a:
            b = a + random.random()*(max_coord*0.01 + 1e-3)
        segments.append((a, b))
        last_end = b

    p0, q0 = segments[0][0], segments[-1][1]
    L_full = tour_length(p0, q0, h)
    L = random.uniform(0.5 * L_full, 1.5 * L_full)
    return segments, L

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instances", type=int, default=1000)
    parser.add_argument("--min-n",       type=int, default=5)
    parser.add_argument("--max-n",       type=int, default=30)
    parser.add_argument("--max-coord",   type=float, default=100.0)
    parser.add_argument("--h",           type=float, default=1.0)
    parser.add_argument("--output",      type=str,   default="data/raw/train.jsonl")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as fout:
        for _ in range(args.n_instances):
            n = random.randint(args.min_n, args.max_n)
            segments, L = random_instance(n, args.max_coord, args.h)
            gs = cover_min_tours(segments, L, args.h)
            dp = cover_min_length(segments, L, args.h)
            fout.write(json.dumps({
                "segments": segments,
                "L": L,
                "gs_tours": gs,
                "dp_tours": dp
            }) + "\n")

if __name__ == "__main__":
    main()
