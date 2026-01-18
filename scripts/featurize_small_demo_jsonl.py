#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from coverage_planning.algs.reference import (
    dp_full_with_plan,
    dp_one_side_with_plan,
    reconstruct_one_side_plan,
)
from coverage_planning.common.constants import DEFAULT_SEED, EPS_GEOM, TOL_NUM, seed_everywhere
from coverage_planning.data.io_utils import sample_from_dict
from coverage_planning.data.schemas import Sample
from coverage_planning.learn.featurize import featurize_sample

RAW_INPUTS = {
    "min_length": Path("data/demo_raw_minlength.jsonl"),
    "min_tours": Path("data/demo_raw_mintours.jsonl"),
}

OUTPUTS = {
    "min_length": Path("data/demo_featurized_minlength.jsonl"),
    "min_tours": Path("data/demo_featurized_mintours.jsonl"),
}


def _load_samples(path: Path, limit: int) -> List[Sample]:
    samples: List[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            samples.append(sample_from_dict(payload))
            if len(samples) >= limit:
                break
    return samples


def _orientation_flags(segments: Sequence[Tuple[float, float]]) -> Tuple[bool, bool]:
    has_positive = any(b >= EPS_GEOM for _, b in segments)
    has_negative = any(a <= -EPS_GEOM for a, _ in segments)
    return has_negative, has_positive


def _min_length_tours(sample: Sample) -> List[List[float]]:
    segments = list(sample.instance.segments)
    h = float(sample.instance.h)
    L = float(sample.instance.L)

    has_negative, has_positive = _orientation_flags(segments)

    if has_negative and has_positive:
        _, tours, _ = dp_full_with_plan(segments, h, L)
        return [[float(p), float(q)] for p, q in tours]

    if has_positive:
        _, candidates, plan = dp_one_side_with_plan(list(segments), h, L, tol=TOL_NUM)
        tours = reconstruct_one_side_plan(candidates, plan)
        return [[float(p), float(q)] for p, q in tours]

    segments_ref = [(-b, -a) for a, b in segments]
    _, candidates, plan = dp_one_side_with_plan(segments_ref, h, L, tol=TOL_NUM)
    tours_ref = reconstruct_one_side_plan(candidates, plan)
    return [[-float(q), -float(p)] for p, q in tours_ref]


def _resolve_tours(sample: Sample) -> tuple[list[list[float]], str, str]:
    meta = sample.gold.meta if isinstance(sample.gold.meta, dict) else {}
    original_objective = meta.get("objective", "min_length")
    tours = [[float(p), float(q)] for p, q in (sample.gold.tours or ())]
    featurize_objective = original_objective

    if original_objective == "min_tours":
        tours = _min_length_tours(sample)
        featurize_objective = "min_length"

    return tours, original_objective, featurize_objective


def _sample_to_featurizer_payload(
    sample: Sample,
    *,
    tours: list[list[float]],
    objective: str,
) -> Dict[str, Any]:
    return {
        "segments": [[float(a), float(b)] for a, b in sample.instance.segments],
        "h": float(sample.instance.h),
        "L": float(sample.instance.L),
        "tours": tours,
        "objective": objective,
        "cost": float(sample.gold.cost),
    }


def summarize_featurized_sample(
    sample: Sample,
    features: Dict[str, Any],
    *,
    original_objective: str,
    featurize_objective: str,
    step_preview: int = 3,
) -> Dict[str, Any]:
    graph = features.get("graph", {})
    seg_nodes = graph.get("seg_nodes", [])
    cand_nodes = graph.get("cand_nodes", [])
    edge_info = graph.get("edges", {}).get("seg_to_cand", {})
    edges = edge_info.get("idx", [])

    steps = features.get("steps", [])
    preview_steps: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps[:step_preview]):
        mask = step.get("mask", {})
        mask_format = mask.get("format", "dense")
        if mask_format == "dense":
            mask_right = mask.get("mask_right", [])
            mask_left_given_right = mask.get("mask_left_given_right", [])
            legal_right_count = sum(int(v) for v in mask_right)
            left_shapes = []
            for right_idx, flag in enumerate(mask_right):
                if flag and right_idx < len(mask_left_given_right):
                    left_shapes.append(len(mask_left_given_right[right_idx]))
            mask_summary: Dict[str, Any] = {
                "format": "dense",
                "right_length": len(mask_right),
                "right_legal": legal_right_count,
                "left_given_right_lengths": left_shapes,
            }
        else:
            legal_right = mask.get("legal_right", [])
            legal_pairs = mask.get("legal_pairs", [])
            mask_summary = {
                "format": "sparse",
                "right_count": len(legal_right),
                "pair_count": len(legal_pairs),
            }
        preview_steps.append(
            {
                "index": idx,
                "case": step.get("case"),
                "mask": mask_summary,
                "chosen_indices": {
                    "left": int(step.get("y_left", -1)),
                    "right": int(step.get("y_right", -1)),
                },
            }
        )

    return {
        "hash_id": sample.hash_id,
        "objective": original_objective,
        "featurize_objective": featurize_objective,
        "graph": {
            "segment_nodes": len(seg_nodes),
            "candidate_nodes": len(cand_nodes),
            "edge_count": len(edges),
        },
        "total_steps": len(steps),
        "step_preview": preview_steps,
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Featurize raw demo JSONL samples for quick inspection."
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum samples per objective to featurize.")
    parser.add_argument("--step_preview", type=int, default=3, help="Number of steps to include in previews.")
    parser.add_argument(
        "--sparse_threshold",
        type=int,
        default=64,
        help="Threshold forwarded to featurize_sample for sparse mask selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed to stabilise any downstream RNG consumers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everywhere(args.seed)

    summaries: Dict[str, Dict[str, float]] = {}
    for objective, input_path in RAW_INPUTS.items():
        output_path = OUTPUTS[objective]
        samples = _load_samples(input_path, args.limit)
        if not samples:
            raise RuntimeError(f"No samples found in {input_path}. Run gen_small_demo_jsonl.py first.")

        featurized_records: List[Dict[str, Any]] = []
        step_counts: List[int] = []
        candidate_counts: List[int] = []
        skipped: List[Tuple[str, str]] = []

        for sample in samples:
            tours, original_objective, featurize_objective = _resolve_tours(sample)
            payload = _sample_to_featurizer_payload(
                sample,
                tours=tours,
                objective=featurize_objective,
            )
            try:
                features = featurize_sample(payload, sparse_threshold=args.sparse_threshold)
            except Exception as exc:  # pragma: no cover - defensive guard
                skipped.append((sample.hash_id, f"{type(exc).__name__}: {exc}"))
                continue

            record = summarize_featurized_sample(
                sample,
                features,
                original_objective=original_objective,
                featurize_objective=featurize_objective,
                step_preview=args.step_preview,
            )
            featurized_records.append(record)
            step_counts.append(record["total_steps"])
            candidate_counts.append(record["graph"]["candidate_nodes"])

        _write_jsonl(output_path, featurized_records)

        summaries[objective] = {
            "count": len(featurized_records),
            "avg_steps": mean(step_counts) if step_counts else 0.0,
            "avg_candidates": mean(candidate_counts) if candidate_counts else 0.0,
            "skipped": skipped,
        }

    print("Featurized demo datasets written:")
    for objective, stats in summaries.items():
        output_path = OUTPUTS[objective]
        skipped = stats.get("skipped", [])
        print(
            f"- {objective}: {int(stats['count'])} samples -> "
            f"avg_steps={stats['avg_steps']:.2f}, avg_candidates={stats['avg_candidates']:.2f} "
            f"(output: {output_path})"
        )
        if skipped:
            preview = ", ".join(f"{hid[:8]}... ({reason.split(':', 1)[0]})" for hid, reason in skipped[:3])
            print(f"  Skipped {len(skipped)} sample(s) due to featurizer errors: {preview}")


if __name__ == "__main__":
    main()
