# Coverline: Paper-Faithful Implementation

Implementation target paper: *Covering Segments on a Line with Drones* (Information Processing Letters 188 (2025) 106540).

This repo includes:
- All 4 algorithmic components from the paper (`GS`, `GSP`, `DPOS`, two-sided composition).
- All 4 figure-style demos plus 4 algorithm demos.
- Oracle-based small-instance crosschecks, theorem-structure tests, and benchmark tooling.

## Environment

Python `3.13` is required.

```bash
conda create -n mlresearch python=3.13 -y
conda activate mlresearch
python --version
```

Install:

```bash
python -m pip install -e ".[dev]"
```

If you do not install the package, set:

```bash
$env:PYTHONPATH = "src"   # PowerShell
```

## Run Tests

```bash
pytest
```

Current baseline in this repository: `20 passed`.

## CLI Usage

### Demos

Run all demos (4 figure demos + 4 algorithm demos):

```bash
python -m coverline.cli demo all --out demos/output
```

Run one demo:

```bash
python -m coverline.cli demo figure1 --out demos/output
python -m coverline.cli demo algorithm-dpos --out demos/output
```

### Benchmarks

Quick benchmark:

```bash
python -m coverline.cli benchmark --mode quick --out benchmarks/results
```

Deep benchmark:

```bash
python -m coverline.cli benchmark --mode deep --out benchmarks/results
```

Outputs include CSV files, a runtime summary plot, and `summary.md`.

## Paper Mapping

| Paper part | Implementation |
|---|---|
| Greedy Strategy (GS), Theorems 1-2 | `src/coverline/algorithms/gs_min_tours.py` |
| GSP for one segment, Theorems 3-4 | `src/coverline/algorithms/gsp_one_segment.py` |
| Candidate sets, Lemmas 5-6 | `src/coverline/candidates.py` |
| DPOS / Equation (1), Theorems 7-8 | `src/coverline/algorithms/dpos_one_side.py` |
| Two-sided composition, Lemma 9, Theorem 10 | `src/coverline/algorithms/min_length_two_side.py` |
| Exact small-instance oracle crosschecks | `src/coverline/oracles/exact_small.py` |
| Coverage validator | `src/coverline/coverage.py` |

## The 4 Figure Demos

- `figure1`: notation/coverage decomposition scenario.
- `figure2`: one-segment counterexample where GS-style solution is worse than GSP.
- `figure3`: numeric one-side counterexample from the paper conditions (`h=4`, `z=9`).
- `figure4`: candidate-set construction example (`C_i`, `C_{i-1}`).

Artifacts are written under `demos/output/figure*/`.

## The 4 Algorithm Demos

- `algorithm-gs`: MinTours walkthrough and GS variant equivalence.
- `algorithm-gsp`: one-segment projection-aware decomposition.
- `algorithm-dpos`: one-side DP with candidate/decision export.
- `algorithm-two-side`: full MinLength two-sided case solver.

Artifacts are written under `demos/output/algorithm_*/`.

## Reproducibility

- Deterministic tolerance policy uses `EPS = 1e-9`.
- Benchmarks use fixed random seed `20260210`.
- Demo and benchmark outputs are deterministic for a fixed Python/OS numeric environment.
