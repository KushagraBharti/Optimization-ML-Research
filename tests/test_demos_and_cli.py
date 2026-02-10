from __future__ import annotations

from pathlib import Path

from coverline.bench.run_benchmarks import run_benchmarks
from coverline.cli import main
from coverline.demo.algorithm_examples import run_all_algorithm_demos
from coverline.demo.figure_examples import run_all_figure_demos


def test_run_all_demos_generate_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "demos"
    figures = run_all_figure_demos(out)
    algos = run_all_algorithm_demos(out)
    assert (out / "figures_summary.json").exists()
    assert (out / "algorithms_summary.json").exists()
    assert "figure1" in figures and "figure4" in figures
    assert "algorithm-gs" in algos and "algorithm-two-side" in algos


def test_cli_demo_and_benchmark_quick(tmp_path: Path) -> None:
    demo_out = tmp_path / "demo_cli"
    bench_out = tmp_path / "bench_cli"
    rc_demo = main(["demo", "figure1", "--out", str(demo_out)])
    rc_bench = main(["benchmark", "--mode", "quick", "--out", str(bench_out)])
    assert rc_demo == 0
    assert rc_bench == 0
    assert (demo_out / "figure1" / "figure1.json").exists()
    assert (bench_out / "one_side_benchmark.csv").exists()
    assert (bench_out / "runtime_summary.png").exists()


def test_benchmark_api_quick_mode(tmp_path: Path) -> None:
    out = tmp_path / "bench_api"
    summary = run_benchmarks(out_dir=out, mode="quick")
    assert summary["rows"]["one_side"] > 0
    assert summary["rows"]["two_side"] > 0
    assert (out / "summary.md").exists()
