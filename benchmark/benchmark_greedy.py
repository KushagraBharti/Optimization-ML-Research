# benchmarks/benchmark_greedy.py
import time
import random
from coverage_planning.greedy import greedy_min_tours
from coverage_planning.utils import tour_length

def random_segments(n: int, span: float = 100.0):
    xs = sorted(random.uniform(0, span) for _ in range(2*n))
    return [(xs[2*i], xs[2*i+1]) for i in range(n)]

def run_benchmark():
    print("Greedy MinTours Benchmarks")
    for n in [1000, 5000, 10000]:
        segs = random_segments(n)
        h = 0.0
        # battery just enough to cover the span of all segments
        x_min = min(a for a,_ in segs)
        x_max = max(b for _,b in segs)
        L = tour_length(x_min, x_max, h) + 1e-6

        start = time.time()
        count, _ = greedy_min_tours(segs, h, L, mode="two_pointer")
        elapsed = time.time() - start
        print(f" n={n:6d} → tours={count:4d}, time={elapsed:.4f}s")

if __name__ == "__main__":
    run_benchmark()
