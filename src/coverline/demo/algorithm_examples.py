from __future__ import annotations

from pathlib import Path

from ..algorithms import (
    solve_min_length_one_segment_gsp,
    solve_min_length_one_side_dpos,
    solve_min_length_one_side_dpos_with_artifacts,
    solve_min_length_two_side,
    solve_min_tours_gs,
    solve_min_tours_gs_linear,
)
from ..coverage import validate_solution
from ..model import Instance
from .common import ensure_dir, plot_instance_with_tours, solution_payload, write_json


def algorithm_gs_demo(out_dir: Path) -> dict:
    out = out_dir / "algorithm_gs"
    ensure_dir(out)
    instance = Instance.from_iterable(
        h=4.0,
        L=24.0,
        segments=[(-8.0, -6.5), (-5.4, -4.2), (-1.8, -0.8), (1.5, 2.8), (4.2, 5.6), (6.7, 7.9)],
    )
    sol_log = solve_min_tours_gs(instance)
    sol_lin = solve_min_tours_gs_linear(instance)
    plot_instance_with_tours(instance, sol_log, "Algorithm Demo: GS MinTours", out / "algorithm_gs.png")

    payload = {
        "instance": {"h": instance.h, "L": instance.L, "segments": [(s.a, s.b) for s in instance.segments]},
        "gs_log": solution_payload(instance, sol_log),
        "gs_linear": solution_payload(instance, sol_lin),
        "same_count": sol_log.tour_count == sol_lin.tour_count,
        "same_total_length": abs(sol_log.total_length - sol_lin.total_length) <= 1e-6,
        "validation": {
            "log_valid": validate_solution(instance, sol_log).valid,
            "linear_valid": validate_solution(instance, sol_lin).valid,
        },
    }
    write_json(out / "algorithm_gs.json", payload)
    return payload


def algorithm_gsp_demo(out_dir: Path) -> dict:
    out = out_dir / "algorithm_gsp"
    ensure_dir(out)
    instance = Instance.from_iterable(h=4.0, L=22.0, segments=[(-8.0, 7.5)])
    sol = solve_min_length_one_segment_gsp(instance)
    plot_instance_with_tours(instance, sol, "Algorithm Demo: GSP One Segment", out / "algorithm_gsp.png")
    payload = solution_payload(instance, sol)
    payload["valid"] = validate_solution(instance, sol).valid
    write_json(out / "algorithm_gsp.json", payload)
    return payload


def algorithm_dpos_demo(out_dir: Path) -> dict:
    out = out_dir / "algorithm_dpos"
    ensure_dir(out)
    instance = Instance.from_iterable(
        h=4.0,
        L=22.0,
        segments=[(0.6, 1.5), (2.4, 3.2), (4.1, 5.1), (6.2, 7.1), (8.0, 8.9)],
    )
    sol, artifacts = solve_min_length_one_side_dpos_with_artifacts(instance)
    plot_instance_with_tours(instance, sol, "Algorithm Demo: DPOS (One Side)", out / "algorithm_dpos.png")
    payload = solution_payload(instance, sol)
    payload["candidate_count"] = len(artifacts.candidates)
    payload["candidates"] = list(artifacts.candidates)
    payload["decision_count"] = len(artifacts.decision)
    payload["valid"] = validate_solution(instance, sol).valid
    write_json(out / "algorithm_dpos.json", payload)
    return payload


def algorithm_two_side_demo(out_dir: Path) -> dict:
    out = out_dir / "algorithm_two_side"
    ensure_dir(out)
    instance = Instance.from_iterable(
        h=4.0,
        L=24.0,
        segments=[(-8.5, -7.0), (-5.9, -4.8), (-2.9, -2.0), (1.8, 2.9), (4.6, 5.7), (7.2, 8.3)],
    )
    sol = solve_min_length_two_side(instance)
    plot_instance_with_tours(instance, sol, "Algorithm Demo: Two-Side MinLength", out / "algorithm_two_side.png")
    payload = solution_payload(instance, sol)
    payload["valid"] = validate_solution(instance, sol).valid
    write_json(out / "algorithm_two_side.json", payload)
    return payload


ALGORITHM_DEMOS = {
    "algorithm-gs": algorithm_gs_demo,
    "algorithm-gsp": algorithm_gsp_demo,
    "algorithm-dpos": algorithm_dpos_demo,
    "algorithm-two-side": algorithm_two_side_demo,
}


def run_algorithm_demo(name: str, out_dir: Path) -> dict:
    if name not in ALGORITHM_DEMOS:
        raise ValueError(f"Unknown algorithm demo: {name}")
    return ALGORITHM_DEMOS[name](out_dir)


def run_all_algorithm_demos(out_dir: Path) -> dict:
    results = {}
    for name in ("algorithm-gs", "algorithm-gsp", "algorithm-dpos", "algorithm-two-side"):
        results[name] = run_algorithm_demo(name, out_dir)
    write_json(out_dir / "algorithms_summary.json", results)
    return results
