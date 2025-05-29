# Coverage Planning with Drones (1D)

A complete reference implementation of the paper
“Covering Segments on a Line with Drones”
(Information Processing Letters, 2025).

## Highlights

- **Greedy MinTours** (`greedy_min_tours`): Optimal \(O(m+n)\) solution for minimizing the number of tours.
- **GSP for Single Segment** (`greedy_min_length_one_segment`): 3-step greedy for 1 segment, minimizing total distance.
- **One-Sided DP** (`dp_one_side`): Exact MinLength in \(O(n^2 m)\) with full candidate generation.
- **Full-Line DP** (`dp_full_line`): Handles segments on both sides of the base station.

## Install


```bash
pip install -e .
```

## Usage

Quick demo:

```bash
python examples/run_all.py
```

Unit tests:

```bash
pytest
```

Benchmarks:

```bash
python benchmarks/benchmark_greedy.py
python benchmarks/benchmark_dp.py
```
