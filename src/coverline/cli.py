from __future__ import annotations

import argparse
from pathlib import Path

from .bench.run_benchmarks import run_benchmarks
from .demo.algorithm_examples import run_algorithm_demo, run_all_algorithm_demos
from .demo.figure_examples import run_all_figure_demos, run_figure_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coverline", description="Coverline research implementation CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run demos and generate artifacts.")
    demo.add_argument(
        "name",
        choices=[
            "figure1",
            "figure2",
            "figure3",
            "figure4",
            "all-figures",
            "algorithm-gs",
            "algorithm-gsp",
            "algorithm-dpos",
            "algorithm-two-side",
            "all-algorithms",
            "all",
        ],
    )
    demo.add_argument("--out", type=Path, default=Path("demos/output"), help="Output directory.")

    bench = sub.add_parser("benchmark", help="Run benchmark suites.")
    bench.add_argument("--mode", choices=["quick", "deep"], default="deep")
    bench.add_argument("--out", type=Path, default=Path("benchmarks/results"))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "demo":
        out = Path(args.out)
        if args.name in {"figure1", "figure2", "figure3", "figure4"}:
            run_figure_demo(args.name, out)
        elif args.name == "all-figures":
            run_all_figure_demos(out)
        elif args.name in {"algorithm-gs", "algorithm-gsp", "algorithm-dpos", "algorithm-two-side"}:
            run_algorithm_demo(args.name, out)
        elif args.name == "all-algorithms":
            run_all_algorithm_demos(out)
        else:
            run_all_figure_demos(out)
            run_all_algorithm_demos(out)
        return 0

    if args.command == "benchmark":
        run_benchmarks(out_dir=Path(args.out), mode=args.mode)
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
