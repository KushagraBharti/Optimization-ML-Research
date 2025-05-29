# benchmarks/benchmark_dp.py
import time
import random
from coverage_planning.dp_1side import dp_one_side
from coverage_planning.dp_both import dp_full_line
from coverage_planning.utils import tour_length

def random_segments(n: int, span: float = 100.0):
    xs = sorted(random.uniform(0, span) for _ in range(2*n))
    return [(xs[2*i], xs[2*i+1]) for i in range(n)]

def bench_one_side():
    print("One‐Sided DP Benchmarks")
    for n in [50, 100, 200]:
        segs = random_segments(n)
        h = 0.0
        # battery large enough for the hardest single segment
        L = max(tour_length(a, b, h) for a,b in segs) + 1e-6

        start = time.time()
        dp, _ = dp_one_side(segs, h, L)
        elapsed = time.time() - start
        print(f" n={n:4d}, states={len(dp):4d}, time={elapsed:.4f}s")

def bench_full_line():
    print("\nFull‐Line DP Benchmarks")
    for n in [50, 100]:
        segs = random_segments(n)
        # alternate left/right
        segs = [(-b, -a) if i%2==0 else (a, b) for i,(a,b) in enumerate(segs)]
        h = 0.0
        # battery enough for a bridge
        L = max(tour_length(a, b, h) for a,b in segs)*2 + 1e-6

        start = time.time()
        cost = dp_full_line(segs, h, L)
        elapsed = time.time() - start
        print(f" n={n:4d}, cost={cost:.4f}, time={elapsed:.4f}s")

if __name__ == "__main__":
    bench_one_side()
    bench_full_line()
