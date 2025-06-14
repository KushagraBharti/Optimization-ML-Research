#!/usr/bin/env python3
# examples/run_all.py  –  smoke tests for all four algorithms

from __future__ import annotations
import time
from typing import List, Tuple

import coverage_planning.utils as utils
from coverage_planning.greedy import greedy_min_tours
from coverage_planning.gsp import greedy_min_length_one_segment
from coverage_planning.dp_1side import dp_one_side
from coverage_planning.dp_both import dp_full_line

utils.VERBOSE = True
SEP = "=" * 80


def hdr(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}\n")


# ---------------- 1. Greedy MinTours --------------------------------------
def run_greedy() -> None:
    hdr("Algorithm 1 – Greedy MinTours")
    h, L = 2.0, 17.0
    segs = [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)]

    print("Input segments :", segs)
    t0 = time.perf_counter()
    count, tours = greedy_min_tours(segs, h, L)
    dt = time.perf_counter() - t0

    print(f"Result: {count} tour(s)")
    for i, (p, q) in enumerate(tours, 1):
        print(f"  Tour {i}: covers [{min(p,q):.2f}, {max(p,q):.2f}]")
    print(f"Elapsed: {dt:.6f} s")


# ---------------- 2. GSP ---------------------------------------------------
def run_gsp() -> None:
    hdr("Algorithm 2 – GSP Single-Segment")
    h, L = 2.0, 13.0
    seg = (2.0, 6.0)

    print("Input segment  :", seg)
    t0 = time.perf_counter()
    count, tours = greedy_min_length_one_segment(seg, h, L)
    dt = time.perf_counter() - t0

    print(f"Result: {count} tour(s)")
    for i, (p, q) in enumerate(tours, 1):
        print(f"  Tour {i}: covers [{min(p,q):.2f}, {max(p,q):.2f}]")
    print(f"Elapsed: {dt:.6f} s")


# ---------------- 3. One-sided DP -----------------------------------------
def run_dp_one_side() -> None:
    hdr("Algorithm 3 – One-Sided DP")
    h, L = 2.0, 19.0
    segs = [(1.0, 2.0), (4.0, 5.0), (7.0, 9.0)]

    print("Input segments :", segs)
    t0 = time.perf_counter()
    pref, _suf, C = dp_one_side(segs, h, L)
    dt = time.perf_counter() - t0

    print(f"Optimal total distance : {pref[-1]:.4f}")
    print(f"|C| = {len(C)}")
    print(f"Elapsed: {dt:.6f} s")


# ---------------- 4. Full-line DP -----------------------------------------
def run_dp_full_line() -> None:
    hdr("Algorithm 4 – Full-Line DP")
    h, L = 2.0, 19.0
    segs = [(-5.0, -3.0), (-2.0, -1.0), (1.0, 2.0), (4.0, 5.0), (7.0, 9.0)]

    print("Input segments :", segs)
    t0 = time.perf_counter()
    cost = dp_full_line(segs, h, L)
    dt = time.perf_counter() - t0

    print(f"Optimal total distance : {cost:.4f}")
    print(f"Elapsed: {dt:.6f} s")


def main() -> None:
    run_greedy()
    run_gsp()
    run_dp_one_side()
    run_dp_full_line()


if __name__ == "__main__":
    main()
