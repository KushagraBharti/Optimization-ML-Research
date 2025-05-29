# coverage_planning/greedy.py

from typing import List, Tuple
from .utils import tour_length, sort_segments, EPS, log

def greedy_min_tours(
    segments: List[Tuple[float, float]],
    h: float,
    L: float
) -> Tuple[int, List[Tuple[float, float]]]:
    """
    Greedy Strategy for MinTours (both-sides).
    Returns (tour_count, [(p,q),...]).
    """
    # Partition
    right = [(a,b) for (a,b) in segments if b >= 0]
    left  = [(-b,-a) for (a,b) in segments if b <  0]

    tours: List[Tuple[float,float]] = []
    total_count = 0

    def process_side(side_segs: List[Tuple[float,float]], sign: int):
        nonlocal total_count, tours
        rem = sort_segments(side_segs)

        # Step A: any segment that alone exceeds L → one tour
        reachable = []
        for a,b in rem:
            if tour_length(a,b,h) > L + EPS:
                total_count += 1
                tours.append((sign*a, sign*b))
            else:
                reachable.append((a,b))
        rem = reachable

        # Step B: greedy cover remaining
        while rem:
            p0, _ = rem[0]
            _, q0 = rem[-1]
            if tour_length(p0, q0, h) <= L + EPS:
                total_count += 1
                tours.append((sign*p0, sign*q0))
                break
            else:
                a_s, b_s = rem.pop()
                total_count += 1
                tours.append((sign*a_s, sign*b_s))

    process_side(right, +1)
    process_side(left,  -1)
    return total_count, tours
