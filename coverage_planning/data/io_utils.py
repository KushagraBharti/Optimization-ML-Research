from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

try:  # Optional parquet support
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pa = None  # type: ignore
    pq = None  # type: ignore

from coverage_planning.data.schemas import Sample, compute_instance_hash, sample_to_dict


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_samples_jsonl(path: str | Path, samples: Sequence[Sample]) -> None:
    target = Path(path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = sample_to_dict(sample)
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def write_samples_parquet(path: str | Path, samples: Sequence[Sample]) -> None:
    if pa is None or pq is None:  # pragma: no cover - optional dependency
        raise RuntimeError("pyarrow is required for parquet output")
    target = Path(path)
    _ensure_parent(target)
    records = [sample_to_dict(sample) for sample in samples]
    table = pa.Table.from_pylist(records)  # type: ignore[arg-type]
    pq.write_table(table, target)  # type: ignore[arg-type]


def write_manifest(directory: str | Path, meta: Dict[str, Any]) -> None:
    target = Path(directory) / "manifest.json"
    _ensure_parent(target)
    payload = dict(meta)
    payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def canonicalize_and_hash(
    segments: Sequence[Sequence[float]],
    h: float,
    L: float,
) -> str:
    normalized = tuple((float(a), float(b)) for a, b in segments)
    return compute_instance_hash(normalized, float(h), float(L))
