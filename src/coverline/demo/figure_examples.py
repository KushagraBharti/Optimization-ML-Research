from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..algorithms import solve_min_length_one_segment_gsp, solve_min_length_one_side_dpos, solve_min_tours_gs
from ..candidates import build_candidate_sets_one_side
from ..geometry import tour_length
from ..model import Instance, Solution, Tour
from .common import ensure_dir, plot_instance_with_tours, solution_payload, write_json


def _manual_solution(tours: list[Tour], metadata: dict[str, object] | None = None) -> Solution:
    return Solution.from_tours(tours, metadata={} if metadata is None else metadata)


def figure1_demo(out_dir: Path) -> dict:
    out = out_dir / "figure1"
    ensure_dir(out)

    instance = Instance.from_iterable(
        h=4.0,
        L=30.0,
        segments=[(-9.0, -7.0), (-4.0, -2.0), (1.0, 4.0), (6.0, 9.0)],
    )
    p = -3.5
    q = 7.5
    manual = _manual_solution(
        [
            Tour(
                left=p,
                right=q,
                length=tour_length(p, q, instance.h),
                maximal=False,
                tag="notation_tour",
            )
        ],
        metadata={"figure": 1, "description": "Notation example for S^j_pq coverage slice."},
    )
    plot_instance_with_tours(instance, manual, "Figure 1-style Notation Example", out / "figure1.png")
    payload = solution_payload(instance, manual)
    write_json(out / "figure1.json", payload)
    return payload


def _find_fig2_counterexample() -> tuple[Instance, Solution, Solution]:
    h = 4.0
    # Deterministic bounded scan.
    for a in (-9.0, -8.0, -7.0, -6.0):
        for b in (6.0, 7.0, 8.0, 9.0):
            if a >= b:
                continue
            for L in (20.0, 21.0, 22.0, 23.0, 24.0, 25.0):
                instance = Instance.from_iterable(h=h, L=L, segments=[(a, b)])
                try:
                    gs = solve_min_tours_gs(instance)
                    gsp = solve_min_length_one_segment_gsp(instance)
                except ValueError:
                    continue
                if gs.total_length > gsp.total_length + 1e-6 and gs.tour_count == gsp.tour_count:
                    return instance, gs, gsp
    raise RuntimeError("Unable to locate deterministic Fig.2-style counterexample.")


def figure2_demo(out_dir: Path) -> dict:
    out = out_dir / "figure2"
    ensure_dir(out)
    instance, gs_solution, gsp_solution = _find_fig2_counterexample()

    plot_instance_with_tours(
        instance,
        gs_solution,
        "Figure 2-style: GS-style Tours",
        out / "figure2_gs.png",
    )
    plot_instance_with_tours(
        instance,
        gsp_solution,
        "Figure 2-style: GSP Optimal Tours",
        out / "figure2_gsp.png",
    )
    payload = {
        "instance": {"h": instance.h, "L": instance.L, "segments": [(s.a, s.b) for s in instance.segments]},
        "gs": solution_payload(instance, gs_solution),
        "gsp": solution_payload(instance, gsp_solution),
        "improvement": gs_solution.total_length - gsp_solution.total_length,
    }
    write_json(out / "figure2.json", payload)
    return payload


def figure3_demo(out_dir: Path) -> dict:
    out = out_dir / "figure3"
    ensure_dir(out)

    h = 4.0
    a1, b1 = 1.0, 3.0
    z = 9.0
    a2, b2 = z, z + 1.0
    g = 2.0
    L = tour_length(g, b2, h)
    instance = Instance.from_iterable(h=h, L=L, segments=[(a1, b1), (a2, b2)])

    red_tours = _manual_solution(
        [
            Tour(left=g, right=b2, length=tour_length(g, b2, h), maximal=True, tag="red_greedy_1"),
            Tour(left=a1, right=g, length=tour_length(a1, g, h), maximal=False, tag="red_greedy_2"),
        ],
        metadata={"figure": 3, "scheme": "red"},
    )
    blue_tours = _manual_solution(
        [
            Tour(left=a2, right=b2, length=tour_length(a2, b2, h), maximal=False, tag="blue_opt_1"),
            Tour(left=a1, right=b1, length=tour_length(a1, b1, h), maximal=False, tag="blue_opt_2"),
        ],
        metadata={"figure": 3, "scheme": "blue"},
    )
    dpos = solve_min_length_one_side_dpos(instance)

    plot_instance_with_tours(instance, red_tours, "Figure 3 Red (Greedy Gap-Covering)", out / "figure3_red.png")
    plot_instance_with_tours(instance, blue_tours, "Figure 3 Blue (Gap-Skipping)", out / "figure3_blue.png")
    plot_instance_with_tours(instance, dpos, "Figure 3 DPOS Result", out / "figure3_dpos.png")

    payload = {
        "h": h,
        "z": z,
        "L": L,
        "L1_red": red_tours.total_length,
        "L2_blue": blue_tours.total_length,
        "blue_better": blue_tours.total_length < red_tours.total_length,
        "dpos_matches_blue": abs(dpos.total_length - blue_tours.total_length) <= 1e-6,
    }
    write_json(out / "figure3.json", payload)
    return payload


def figure4_demo(out_dir: Path) -> dict:
    out = out_dir / "figure4"
    ensure_dir(out)

    instance = Instance.from_iterable(
        h=4.0,
        L=20.0,
        segments=[(0.8, 1.6), (2.3, 3.2), (4.0, 5.4), (6.8, 8.2)],
    )
    csets = build_candidate_sets_one_side(instance)
    dpos = solve_min_length_one_side_dpos(instance)
    plot_instance_with_tours(instance, dpos, "Figure 4-style Candidate Set Construction Context", out / "figure4.png")

    payload = {
        "instance": {"h": instance.h, "L": instance.L, "segments": [(s.a, s.b) for s in instance.segments]},
        "candidate_union": list(csets.union),
        "candidate_sets_by_i": {str(k + 1): list(v) for k, v in csets.by_index.items()},
        "dpos_total_length": dpos.total_length,
    }
    write_json(out / "figure4.json", payload)
    return payload


FIGURE_DEMOS = {
    "figure1": figure1_demo,
    "figure2": figure2_demo,
    "figure3": figure3_demo,
    "figure4": figure4_demo,
}


def run_figure_demo(name: str, out_dir: Path) -> dict:
    if name not in FIGURE_DEMOS:
        raise ValueError(f"Unknown figure demo: {name}")
    return FIGURE_DEMOS[name](out_dir)


def run_all_figure_demos(out_dir: Path) -> dict:
    results = {}
    for name in ("figure1", "figure2", "figure3", "figure4"):
        results[name] = run_figure_demo(name, out_dir)
    write_json(out_dir / "figures_summary.json", results)
    return results
