# Coverage Planning with Drones (1D)

A reference implementation of the paper  
**"Covering Segments on a Line with Drones"**

Bereg, Sergey, et al. "Covering segments on a line with drones." Information Processing Letters, vol. 188, Feb. 2025, p. 106540, https://doi.org/10.1016/j.ipl.2024.106540.

---

## Project Overview

| Algorithm                                                        | Description                                                                                                               | Time Complexity                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Greedy MinTours** (`greedy_min_tours`)                         | Optimal strategy to minimize the *number* of tours covering disjoint segments, by repeatedly covering the farthest point. | \(O(n \log n)\) (sorting dominates; greedy scan)   |
| **GSP Single-Segment MinLength** (`greedy_min_length_one_segment`)| Exact 3-case projection algorithm for covering a single segment with minimum *total distance*.                            | \(O(1)\) (constant-time geometric computations)     |
| **One-Sided MinLength DP** (`dp_one_side`)                       | Dynamic programming for segments on one side of the base: builds candidate set, then pointer-driven DP.                   | \(O(n m \log n)\), where \(m\) = optimal tours      |
| **Full-Line MinLength DP** (`dp_full_line`)                      | Combines two one-sided DPs + bridge tours to handle both sides of the base.                                               | \(O(n m \log n)\)                                   |

> **Notation:**  
> \(n\) = number of segments  
> \(m\) = number of tours in the optimal solution (\(m \leq n\))

## Installation

I recommend using **Anaconda** for environment management:

```bash
conda create -n mlresearch python=3.11 -y
conda activate mlresearch

# sanity checks
where.exe python    # should point into .../anaconda3/envs/mlresearch
python --version

# install package in editable mode with dev dependencies
pip install -e .[dev]
```

## Usage

### Quick Demo

Run the deterministic driver that exercises all reference algorithms:

```bash
python examples/run_all_algorithms.py
```

### Reference vs Legacy Imports

- `coverage_planning.gs`, `coverage_planning.gsp`, `coverage_planning.dpos`, `coverage_planning.dp_full`, `coverage_planning.dp_full_with_plan` now resolve to the **reference, paper-faithful** solvers.
- Legacy heuristics remain available on purpose as `coverage_planning.legacy_gs`, `coverage_planning.legacy_gsp`, `coverage_planning.legacy_dpos`, and `coverage_planning.legacy_dp_full` (or under `coverage_planning.algs.heuristics`). They are fenced off from the data and learning pipelines.

### Shared Constants

Use `coverage_planning.common.constants` for tolerances, seeding, and utilities:

- `EPS_GEOM`, `TOL_NUM` – canonical geometric/ numeric tolerances.
- `DEFAULT_SEED`, `RNG_SEEDS` – deterministic seeds for tests, data, and benchmarks.
- `seed_everywhere(seed)` – helper to initialise Python, NumPy, and (optionally) PyTorch RNGs.

### Full-Line DP Entry Points

- `coverage_planning.dp_full` returns the optimal cost only (fast QC checks).
- `coverage_planning.dp_full_with_plan` returns `(cost, tours, meta)` and is used by the labelers and featurizer.
- `coverage_planning.find_maximal_bridge_p` exposes the bridge-anchor helper for consumers that need direct access to the supporting geometry.

### Learning Surfaces

- `coverage_planning.learn.featurize.featurize_sample` rebuilds legality masks and graph features for imitation learning.
- `coverage_planning.data.labelers.label_gold` generates solver-faithful gold labels; `make_sample` serialises them under the canonical schema.
- See [`docs/REPO_STRUCTURE.md`](docs/REPO_STRUCTURE.md) for a concise overview of how instances flow through generators → labelers → featurisation.

### Benchmarks

Each algorithm ships with a CSV-style benchmark under `benchmarks/`. Increase `--n` to run longer samples:

```bash
python benchmarks/bench_gs.py --n 1000
python benchmarks/bench_gsp.py --n 1000
python benchmarks/bench_dpos.py --n 1000
python benchmarks/bench_dp_full.py --n 1000
```

### Unit Tests

Run the complete pytest suite:

```bash
pytest -q
```

Property-based checks use [Hypothesis](https://hypothesis.readthedocs.io/); if the library is not installed those tests are automatically skipped while unit/oracle checks still execute.

## Repository Structure

`
Optimization-ML-Research/
|-- coverage_planning/
|   |-- algs/
|   |   |-- __init__.py
|   |   |-- geometry.py
|   |   |-- heuristics/  # legacy heuristics
|   |   |-- reference/   # reference solvers
|   |-- common/
|   |   |-- constants.py
|   |-- data/
|   |   |-- gen_instances.py
|   |   |-- labelers.py
|   |   |-- _legacy_solvers_provider.py
|   |-- eval/
|   |   |-- metrics.py
|   |-- learn/
|       |-- featurize.py
|       |-- mask_predicates.py
|-- benchmarks/
|   |-- README.md
|   |-- bench_dp_full.py
|   |-- bench_dpos.py
|   |-- bench_gs.py
|   |-- bench_gsp.py
|   |-- micro_bench_small_medium.py
|-- benchmarks_legacy/
|   |-- README.md
|   |-- benchmark_dp.py
|   |-- benchmark_greedy.py
|-- docs/
|   |-- REPO_STRUCTURE.md
|-- examples/
|   |-- run_all_algorithms.py
|-- scripts/
|   |-- gen_dataset.py
|   |-- qc_dataset.py
|-- tests/
|   |-- test_dp_full_plan_roundtrip.py
|   |-- test_featurize.py
|   |-- test_featurize_perf.py
|   |-- test_mask_predicates.py
|   |-- ...
|-- README.md
|-- LICENSE
|-- pyproject.toml
|-- requirements.txt
`

## Next Directions

This repo currently focuses on **exact classical algorithms** plus reproducible tooling (examples, benchmarks, and guided tests). Planned future work includes:
- **Data generation pipelines** for supervised learning.
- **Graph neural networks (GNNs)** trained on optimal tours.
- **Reinforcement learning agents** for adaptive coverage.

## Reference

Bereg, Sergey, et al. "Covering segments on a line with drones." Information Processing Letters, vol. 188, Feb. 2025, p. 106540, https://doi.org/10.1016/j.ipl.2024.106540.

## Contributing

Contributions (bug fixes, new benchmarks, improved data generation for ML training) are welcome! Please open an issue or submit a pull request.

## License

MIT License © 2025 Kushagra Bharti

See [LICENSE](LICENSE) for details.
