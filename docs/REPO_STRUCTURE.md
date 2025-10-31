# Repository Structure

This project is organised around the paper-faithful reference solvers for
1‑dimensional drone coverage and the machinery required to generate labelled
datasets for learning experiments.  The table below highlights the relevant
surfaces.

| Path | Purpose | Key exports |
| ---- | ------- | ----------- |
| `coverage_planning/algs/reference` | Reference implementations of Algorithms 1–4 from the paper. All public solver entry points re-export from here. | `gs`, `gsp`, `dpos`, `dp_full`, `dp_full_with_plan` |
| `coverage_planning/algs/heuristics` | Historical heuristics retained for comparison and legacy benchmarks. | `legacy_*` symbols via package root |
| `coverage_planning/common` | Shared tolerances, RNG seeding, and geometry constants. | `EPS_GEOM`, `TOL_NUM`, `seed_everywhere` |
| `coverage_planning/data` | Dataset plumbing: instance generators, labelers, schema definitions. | `label_gold`, `label_near_optimal`, `make_sample` |
| `coverage_planning/learn` | Transition oracles and featurisation pipeline used for imitation learning. | `transition_*`, `featurize_sample` |
| `coverage_planning/eval` | Metrics consumed by scripts, benchmarks, and QC dashboards. | `candidate_size_summary`, `bridge_benefit` |
| `benchmarks/` | Deterministic benchmarks for the reference solvers. | CLI scripts (`bench_*.py`) |
| `benchmarks_legacy/` | Frozen heuristic benchmarks retained for historical comparisons only. | Legacy CLI scripts |

## Data Flow at a Glance

1. **Instance synthesis** – `coverage_planning.data.gen_instances` (family configs,
   RNG control) produces disjoint segment sets with battery limits.
2. **Gold labelling** – `coverage_planning.data.labelers.label_gold` runs the
   appropriate reference solver (`dp_full_with_plan` for mixed instances) and
   records solver metadata.
3. **Near-opt sampling** – hook provided by `label_near_optimal` (currently a
   conservative stub) for future search-based exploration.
4. **Featurisation** – `coverage_planning.learn.featurize.featurize_sample`
   rebuilds legality masks and graph views used for imitation learning.
5. **Metrics & QC** – scripts under `scripts/` and benchmarks under
   `benchmarks/` rely on `coverage_planning.eval.metrics` for consistent summary
   statistics.

All package imports are absolute and resolve through the `coverage_planning`
namespace, ensuring editable installs (`pip install -e .`) behave identically to
CI deployments.  Legacy heuristics stay available under the
`coverage_planning.algs.heuristics` namespace (and via the `legacy_*` aliases)
but are fenced off from the data and learning stacks.
