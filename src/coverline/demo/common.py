from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from ..model import Instance, Solution


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def plot_instance_with_tours(
    instance: Instance,
    solution: Solution,
    title: str,
    out_path: Path,
    annotate: bool = True,
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)

    # Base and projection line.
    ax.scatter([0], [0], color="red", s=40, zorder=5, label="Base O")
    ax.axhline(instance.h, color="#999999", linestyle="--", linewidth=1, label=f"y = {instance.h}")

    for s in instance.segments:
        ax.plot([s.a, s.b], [instance.h, instance.h], color="black", linewidth=4, solid_capstyle="butt")

    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]
    for i, t in enumerate(solution.tours):
        color = palette[i % len(palette)]
        ax.plot([0, t.left, t.right, 0], [0, instance.h, instance.h, 0], color=color, linewidth=1.8)
        if annotate:
            ax.text(
                0.5 * (t.left + t.right),
                instance.h + 0.08 * max(1.0, instance.h),
                f"t{i+1}",
                color=color,
                fontsize=8,
                ha="center",
            )

    all_x = [0.0]
    for s in instance.segments:
        all_x.extend([s.a, s.b])
    for t in solution.tours:
        all_x.extend([t.left, t.right])
    lo = min(all_x) - 1.0
    hi = max(all_x) + 1.0
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.2 * instance.h, 1.6 * instance.h)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def solution_payload(instance: Instance, solution: Solution) -> dict:
    return {
        "h": instance.h,
        "L": instance.L,
        "segments": [{"a": s.a, "b": s.b} for s in instance.segments],
        "tour_count": solution.tour_count,
        "total_length": solution.total_length,
        "tours": [
            {
                "left": t.left,
                "right": t.right,
                "length": t.length,
                "maximal": t.maximal,
                "tag": t.tag,
            }
            for t in solution.tours
        ],
        "metadata": solution.metadata,
    }


def merge_payloads(payloads: Iterable[tuple[str, dict]]) -> dict:
    return {k: v for k, v in payloads}
