# Coverage Planning with Drones (1D)

A reference implementation of the paper  
**“Covering Segments on a Line with Drones”**  
(Information Processing Letters, 2025).

This package provides four core algorithms to plan drone tours on a 1D line under battery constraints, along with structured logging and benchmarks.

---

## Project Overview

| Algorithm                                                   | Description                                                                                                          | Time Complexity                                  |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Greedy MinTours** (`greedy_min_tours`)                    | Optimal strategy to minimize the *number* of tours covering disjoint segments, by repeatedly covering the farthest. | \(O(n\log n)\) (sorting dominates; linear scan)   |
| **GSP Single-Segment MinLength** (`greedy_min_length_one_segment`) | 3-step projection algorithm for a single segment, minimizing *total distance* in at most 3 tours.                   | \(O(1)\) (constant-time geometric computations)   |
| **One-Sided MinLength DP** (`dp_one_side`)                  | Exact DP for segments on one side of the base: builds candidate set (\(O(n\,m\log n)\)), then pointer-driven DP (\(O(n\,m)\)). | \(O(n\,m\log n)\), where \(m\) is optimal tour count |
| **Full-Line MinLength DP** (`dp_full_line`)                 | Combines two one-sided DPs + “bridge” tours to handle both sides of the base.                                       | \(O(n\,m\log n + n\,m\log(nm))\approx O(n\,m\log n)\)  |

> **Notation:**  
> \(n\) = number of segments;  
> \(m\) = number of tours in the optimal solution (≤ n).

---

## Features

- **Exact oracles** for optimal tour count and total distance.  
- **Greedy baselines** for instantaneous planning.  
- **Structured debugging**: single-flag `VERBOSE` for step-by-step logs.  
- **Unit tests** covering edge cases (via `pytest`).  
- **Benchmark scripts** for runtime profiling.

---

## Installation

This project is built using Anaconda (highly recommend to follow that)

```bash
conda create -n mlresearch python -y
conda activate mlresearch

# sanity checks
where.exe python    # should point into ...\anaconda3\envs\mlresearch
python --version

pip install -e .[dev]
```

## Usage

Quick demo:

```bash
python examples/run_all.py
```

Benchmarks:

```bash
python benchmark/benchmark_greedy.py
python benchmark/benchmark_dp.py
```

Unit tests:

```bash
pytest
```
