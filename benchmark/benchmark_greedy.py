# benchmarks/benchmark_greedy_extended.py

import time, statistics, random
from coverage_planning.greedy import greedy_min_tours
from coverage_planning.utils import tour_length

def random_uniform(n, span=100.0):
    xs = sorted(random.uniform(0, span) for _ in range(2*n))
    return [(xs[2*i], xs[2*i+1]) for i in range(n)]

def random_clustered(n, span=100.0, clusters=5):
    segs = []
    for _ in range(clusters):
        center = random.uniform(0, span)
        for _ in range(n//clusters):
            a = center + random.uniform(-5, 5)
            b = a + random.uniform(0.1, 5)
            segs.append((min(a,b), max(a,b)))
    return sorted(segs, key=lambda x: x[0])

def bench_greedy(dist_fn, label, sizes=[1000,5000,10000], runs=3):
    print(f"\n=== Greedy ({label}) Benchmarks ===")
    for n in sizes:
        times = []
        for _ in range(runs):
            segs = dist_fn(n)
            h = 0.0
            x_min = min(a for a,_ in segs)
            x_max = max(b for _,b in segs)
            L = tour_length(x_min, x_max, h) + 1e-6
            start = time.perf_counter()
            count, _ = greedy_min_tours(segs, h, L)
            times.append(time.perf_counter() - start)
        print(f"n={n:6d} | t_avg={statistics.mean(times):.4f}s "
              f"| t_std={statistics.stdev(times):.4f}s")

if __name__ == "__main__":
    bench_greedy(random_uniform,   "Uniform")
    bench_greedy(random_clustered, "Clustered")
    # Worst-case: many tiny segments, large m
    bench_greedy(lambda n: [(i*0.01, i*0.01+0.009) for i in range(n)], "WorstCase")
