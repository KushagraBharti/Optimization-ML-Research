## Reference Benchmarks

All scripts in this directory exercise the **paper-faithful reference solvers**.  Seeds and
numeric tolerances are sourced from `coverage_planning.common.constants` to keep runs
deterministic across environments.

- `bench_gs.py`, `bench_gsp.py` – greedy MinTours / MinLength coverage.
- `bench_dpos.py` – one-sided dynamic program (DPOS).
- `bench_dp_full.py` – full-line dynamic program with bridging.
- `micro_bench_small_medium.py` – mixed harness sweeping small ↔ medium instances.

Example:

```bash
python benchmarks/bench_dpos.py --help
python benchmarks/bench_dpos.py --n 2000 --seed 4242
```

For heuristic-only historical experiments see `../benchmarks_legacy/README.md`.
