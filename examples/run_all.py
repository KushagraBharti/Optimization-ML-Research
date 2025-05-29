#!/usr/bin/env python3
# examples/run_all.py

import time
import coverage_planning.utils as utils
from coverage_planning.greedy   import greedy_min_tours
from coverage_planning.gsp      import greedy_min_length_one_segment
from coverage_planning.dp_1side import dp_one_side
from coverage_planning.dp_both  import dp_full_line

# Enable detailed internal logging for demonstration
utils.VERBOSE = True

def print_header(title: str):
    sep = "=" * 80
    print(f"\n{sep}\n{title}\n{sep}\n")

def run_greedy_example():
    print_header("Algorithm 1: Greedy MinTours")
    h = 2.0
    segs = [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)]
    # choose L so minimal-length fails then maximal is used
    L = 17.0
    print(f"Input segments: {segs}")
    print(f"Line height y = {h}, Battery limit L = {L:.4f}\n")

    start = time.perf_counter()
    count, tours = greedy_min_tours(segs, h, L)
    elapsed = time.perf_counter() - start

    print(f"Result: {count} tour(s)")
    for idx,(p,q) in enumerate(tours, start=1):
        print(f"  Tour {idx}: from x={p:.4f} to x={q:.4f}")
    print(f"\nTime elapsed: {elapsed:.6f} seconds")

def run_gsp_example():
    print_header("Algorithm 2: GSP Single-Segment MinLength")
    h = 2.0
    seg = (2.0, 6.0)
    # choose L between full cover and split threshold
    L = 13.0
    print(f"Input segment: {seg}")
    print(f"Line height y = {h}, Battery limit L = {L:.4f}\n")

    start = time.perf_counter()
    count, tours = greedy_min_length_one_segment(seg, h, L)
    elapsed = time.perf_counter() - start

    print(f"Result: {count} tour(s)")
    for idx,(p,q) in enumerate(tours, start=1):
        print(f"  Tour {idx}: from x={p:.4f} to x={q:.4f}")
    print(f"\nTime elapsed: {elapsed:.6f} seconds")

def run_dp_one_side_example():
    print_header("Algorithm 3: One-Sided MinLength DP (DPOS)")
    h = 2.0
    segs = [(1.0, 2.0), (4.0, 5.0), (7.0, 9.0)]
    L = 19.0
    print(f"Input segments: {segs}")
    print(f"Line height y = {h}, Battery limit L = {L:.4f}\n")

    start = time.perf_counter()
    dp_vals, backptr = dp_one_side(segs, h, L, side_label="RIGHT SIDE")
    elapsed = time.perf_counter() - start

    print(f"Optimal total distance: {dp_vals[-1]:.4f}")
    print(f"Number of DP states (|C|): {len(dp_vals)}\n")
    print("Backpointer descriptions for each candidate:")
    for c in sorted(backptr):
        print(f"  c={c:.2f} → {backptr[c]}")
    print(f"\nTime elapsed: {elapsed:.6f} seconds")

def run_dp_full_line_example():
    print_header("Algorithm 4: Full-Line MinLength DP")
    h = 2.0
    segs = [(-5.0, -3.0), (-2.0, -1.0), (1.0, 2.0), (4.0, 5.0), (7.0, 9.0)]
    L = 19.0
    print(f"Input segments: {segs}")
    print(f"Line height y = {h}, Battery limit L = {L:.4f}\n")

    start = time.perf_counter()
    cost = dp_full_line(segs, h, L)
    elapsed = time.perf_counter() - start

    print(f"Optimal total distance: {cost:.4f}")
    print(f"\nTime elapsed: {elapsed:.6f} seconds")

def main():
    run_greedy_example()
    run_gsp_example()
    run_dp_one_side_example()
    run_dp_full_line_example()

if __name__ == "__main__":
    main()
