from __future__ import annotations

from coverage_planning.common.constants import seed_everywhere
from coverage_planning.data.labelers import label_gold, label_near_optimal, make_sample
from coverage_planning.data.schemas import Instance, Sample
from coverage_planning.learn.featurize import featurize_sample
from scripts.featurize_small_demo_jsonl import summarize_featurized_sample


def _build_demo_sample() -> Sample:
    seed_everywhere(2025)
    instance = Instance(
        segments=((0.0, 1.5), (2.5, 4.0)),
        h=5.0,
        L=30.0,
    )
    gold = label_gold(instance, objective="min_length", family="smoke")
    near_opt = label_near_optimal(instance, gold, objective="min_length")
    return make_sample(instance, gold, near_opt, split_tag="demo_smoke", seed=2025)


def test_smoke_featurize_summary() -> None:
    sample = _build_demo_sample()
    payload = {
        "segments": [[float(a), float(b)] for a, b in sample.instance.segments],
        "h": float(sample.instance.h),
        "L": float(sample.instance.L),
        "tours": [[float(p), float(q)] for p, q in (sample.gold.tours or ())],
        "objective": "min_length",
        "cost": float(sample.gold.cost),
    }

    features = featurize_sample(payload)
    summary = summarize_featurized_sample(
        sample,
        features,
        original_objective="min_length",
        featurize_objective="min_length",
        step_preview=3,
    )

    assert summary["hash_id"] == sample.hash_id
    assert summary["graph"]["segment_nodes"] >= 1
    assert summary["graph"]["candidate_nodes"] >= 1
    assert summary["total_steps"] >= 1
    assert summary["step_preview"], "Expected at least one preview step"

    first_step = summary["step_preview"][0]
    mask = first_step["mask"]
    if mask["format"] == "dense":
        assert mask["right_length"] > 0
        assert mask["right_legal"] > 0
    else:
        assert mask["right_count"] > 0
        assert mask["pair_count"] > 0
    assert first_step["chosen_indices"]["right"] >= 0
    assert first_step["chosen_indices"]["left"] >= 0
