# Coverage Planning with Drones (1D)

A reference implementation of the paper  
**“Covering Segments on a Line with Drones”**  

Bereg, Sergey, et al. “Covering segments on a line with drones.” Information Processing Letters, vol. 188, Feb. 2025, p. 106540, https://doi.org/10.1016/j.ipl.2024.106540. 

---

## 📌 Project Overview

| Algorithm                                                       | Description                                                                                                               | Time Complexity                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| **Greedy MinTours** (`greedy_min_tours`)                         | Optimal strategy to minimize the *number* of tours covering disjoint segments, by repeatedly covering the farthest point. | \(O(n \log n)\) (sorting dominates; greedy scan)   |
| **GSP Single-Segment MinLength** (`greedy_min_length_one_segment`)| Exact 3-case projection algorithm for covering a single segment with minimum *total distance*.                             | \(O(1)\) (constant-time geometric computations)     |
| **One-Sided MinLength DP** (`dp_one_side`)                       | Dynamic programming for segments on one side of the base: builds candidate set, then pointer-driven DP.                   | \(O(n m \log n)\), where \(m\) = optimal tours      |
| **Full-Line MinLength DP** (`dp_full_line`)                      | Combines two one-sided DPs + bridge tours to handle both sides of the base.                                               | \(O(n m \log n)\)                                  |

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
Run the included example script:
```bash
python examples/run_all.py
```

### Benchmarks
Benchmark DP runtimes:
```bash
python benchmark/benchmark_dp.py
```

Benchmark Greedy MinTours on large instances:
```bash
python benchmark/benchmark_greedy.py
```

### Unit Tests
Run all tests with:
```bash
pytest
```

## Repository Structure

```
Optimization-ML-Research/
│
├── coverage_planning/             # Core algorithms
│   ├── algs/
│   │   ├── geometry.py            # Geometry utilities
│   │   ├── heuristics/
│   │   │   ├── gs_mintours.py     # Greedy MinTours (Algorithm 1)
│   │   │   ├── gsp_single.py      # GSP Single Segment (Algorithm 2)
│   │   │   ├── dp_one_side_heur.py# One-Sided DP (Algorithm 3)
│   │   │   ├── dp_full_line_heur.py# Full-Line DP (Algorithm 4)
│
├── benchmark/                     # Benchmark scripts
│   ├── benchmark_dp.py
│   ├── benchmark_greedy.py
│
├── examples/                      # Example scripts
│   ├── run_all.py
│
├── tests/                         # Unit tests (pytest)
│   ├── test_greedy.py
│   ├── test_gsp.py
│   ├── test_dp_1side.py
│   ├── test_dp_both.py
│
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── pyproject.toml                  # Build config
├── requirements.txt                # Dependencies
└── .gitignore
```

## Next Directions

This repo currently focuses on **exact classical algorithms**. Planned future work includes:
- **Data generation pipelines** for supervised learning.
- **Graph neural networks (GNNs)** trained on optimal tours.
- **Reinforcement learning agents** for adaptive coverage.

## Reference

Bereg, Sergey, et al. “Covering segments on a line with drones.” Information Processing Letters, vol. 188, Feb. 2025, p. 106540, https://doi.org/10.1016/j.ipl.2024.106540. 

## Contributing

Contributions (bug fixes, new benchmarks, improved data generation for ML training) are welcome! Please open an issue or submit a pull request.

## License

MIT License © 2025 Kushagra Bharti

See [LICENSE](LICENSE) for details.

