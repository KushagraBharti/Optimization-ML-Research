#!/usr/bin/env python3
"""examples/run_all.py – smoke‑test for the four reference algorithms.

Run this file directly, or execute `python -m examples.run_all` from the project
root.  It prints small, illustrative outputs and timings for:

  1. Greedy MinTours (Algorithm 1)
  2. GSP single‑segment MinLength (Algorithm 2)
  3. One‑sided DP (DPOS, Algorithm 3)
  4. Full‑line DP (Algorithm 4)
"""

from __future__ import annotations

import time
from typing import List, Tuple, Sequence

import coverage_planning.utils as utils
from coverage_planning.greedy import greedy_min_tours
from coverage_planning.gsp import greedy_min_length_one_segment
from coverage_planning.dp_1side import dp_one_side
from coverage_planning.dp_both import dp_full_line

# Activate verbose internal logging so the user can see the algorithmic traces.
utils.VERBOSE = True

SEP = "=" * 80

def _hdr(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}\n")

# ---------------------------------------------------------------------------
# 1. Greedy MinTours
# ---------------------------------------------------------------------------

def run_greedy_example() -> None:
    _hdr("Algorithm 1 – Greedy MinTours")
    h = 2.0
    segs: List[Tuple[float, float]] = [(1.0, 2.0), (4.0, 5.0), (7.0, 8.0)]
    L = 17.0  # Large enough that the first minimal‑length tour is infeasible

    print(f"Input segments           : {segs}")
    print(f"Line height (y)          : {h}")
    print(f"Battery limit (L)        : {L}\n")

    t0 = time.perf_counter()
    cnt, tours = greedy_min_tours(segs, h, L)
    dt = time.perf_counter() - t0

    print(f"Result: {cnt} tour(s)")
    for i, (p, q) in enumerate(tours, 1):
        left, right = sorted((p, q))
        print(f"  Tour {i:2d}: covers x ∈ [{left:.2f}, {right:.2f}]")
    print(f"Elapsed: {dt:.6f} s")

# ---------------------------------------------------------------------------
# 2. GSP – single segment
# ---------------------------------------------------------------------------

def run_gsp_example() -> None:
    _hdr("Algorithm 2 – GSP Single‑Segment MinLength")
    h = 2.0
    seg = (2.0, 6.0)
    L = 13.0  # Forces a split at projection

    print(f"Input segment            : {seg}")
    print(f"Line height (y)          : {h}")
    print(f"Battery limit (L)        : {L}\n")

    t0 = time.perf_counter()
    cnt, tours = greedy_min_length_one_segment(seg, h, L)
    dt = time.perf_counter() - t0

    print(f"Result: {cnt} tour(s)")
    for i, (p, q) in enumerate(tours, 1):
        left, right = sorted((p, q))
        print(f"  Tour {i:2d}: covers x ∈ [{left:.2f}, {right:.2f}]")
    print(f"Elapsed: {dt:.6f} s")

# ---------------------------------------------------------------------------
# 3. One‑sided DP (DPOS)
# ---------------------------------------------------------------------------

def _extract_dp_output(result: Sequence) -> tuple[List[float], List[float] | None, List[float]]:
    """Helper to accommodate the two possible return signatures of `dp_one_side`.

    * New faithful version returns (prefix_costs, suffix_costs, candidates).
    * If user still has an older (two‑value) variant lying around we degrade
      gracefully.  Raises if the arity is something unexpected.
    """
    if len(result) == 3:
        return result  # type: ignore[return-value]
    if len(result) == 2:
        prefix, cands = result  # type: ignore[assignment]
        return prefix, None, cands
    raise TypeError("Unsupported dp_one_side return signature – expected 2 or 3 values.")


def run_dp_one_side_example() -> None:
    _hdr("Algorithm 3 – One‑Sided MinLength DP (DPOS)")
    h = 2.0
    segs = [(1.0, 2.0), (4.0, 5.0), (7.0, 9.0)]
    L = 19.0

    print(f"Input segments           : {segs}")
    print(f"Line height (y)          : {h}")
    print(f"Battery limit (L)        : {L}\n")

    t0 = time.perf_counter()
    result = dp_one_side(segs, h, L)  # New signature has no side_label kwarg.
    dt = time.perf_counter() - t0

    prefix_costs, _suffix, candidates = _extract_dp_output(result)

    print(f"Optimal total distance   : {prefix_costs[-1]:.4f}")
    print(f"Number of DP states (|C|): {len(candidates)}")
    print("Candidates               :", candidates)
    print(f"Elapsed: {dt:.6f} s")

# ---------------------------------------------------------------------------
# 4. Full‑line DP
# ---------------------------------------------------------------------------

def run_dp_full_line_example() -> None:
    _hdr("Algorithm 4 – Full‑Line MinLength DP")
    h = 2.0
    segs = [(-5.0, -3.0), (-2.0, -1.0), (1.0, 2.0), (4.0, 5.0), (7.0, 9.0)]
    L = 19.0

    print(f"Input segments           : {segs}")
    print(f"Line height (y)          : {h}")
    print(f"Battery limit (L)        : {L}\n")

    t0 = time.perf_counter()
    cost = dp_full_line(segs, h, L)
    dt = time.perf_counter() - t0

    print(f"Optimal total distance   : {cost:.4f}")
    print(f"Elapsed: {dt:.6f} s")

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    run_greedy_example()
    run_gsp_example()
    run_dp_one_side_example()
    run_dp_full_line_example()


if __name__ == "__main__":
    main()
