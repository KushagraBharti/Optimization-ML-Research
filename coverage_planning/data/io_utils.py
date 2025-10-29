from __future__ import annotations

import gzip
import json
from typing import Iterable, Iterator

from .schemas import Sample, sample_to_dict


def write_jsonl_gz(path: str, records: Iterable[Sample]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(sample_to_dict(record), sort_keys=True) + "\n")


def read_jsonl_gz(path: str) -> Iterator[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)
