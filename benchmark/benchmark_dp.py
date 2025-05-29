# benchmarks/benchmark_dp_extended.py

import time, statistics
import random
from coverage_planning.dp_1side import dp_one_side, generate_candidates_one_side
from coverage_planning.dp_both import dp_full_line
from coverage_planning.utils import tour_length

def random_segments(n, span=100.0):
    xs = sorted(random.uniform(0, span) for _ in range(2*n))
    return [(xs[2*i], xs[2*i+1]) for i in range(n)]

def bench_one_side(avg_runs=5):
    print("=== One‐Sided DP Extended Benchmarks ===")
    for n in [100, 200, 400]:
        times, states = [], []
        for _ in range(avg_runs):
            segs = random_segments(n)
            h = 0.0
            L = max(tour_length(a,b,h) for a,b in segs) + 1e-6
            start = time.perf_counter()
            dp, _ = dp_one_side(segs, h, L)
            times.append(time.perf_counter() - start)
            states.append(len(dp))
        print(f"n={n:4d} | states≈{statistics.mean(states):.0f} "
              f"| t_avg={statistics.mean(times):.4f}s "
              f"| t_std={statistics.stdev(times):.4f}s")

def bench_full_line(avg_runs=5):
    print("\n=== Full‐Line DP Extended Benchmarks ===")
    for n in [100, 200]:
        times = []
        for _ in range(avg_runs):
            segs = random_segments(n)
            # alternate left/right
            segs = [(-b,-a) if i%2==0 else (a,b) 
                    for i,(a,b) in enumerate(segs)]
            h = 0.0
            L = max(tour_length(a,b,h) for a,b in segs)*2 + 1e-6
            start = time.perf_counter()
            cost = dp_full_line(segs, h, L)
            times.append(time.perf_counter() - start)
        print(f"n={n:4d} | t_avg={statistics.mean(times):.4f}s "
              f"| t_std={statistics.stdev(times):.4f}s")

if __name__ == "__main__":
    bench_one_side()
    bench_full_line()
