from __future__ import annotations
import json, gzip
from dataclasses import asdict
from typing import Iterable
from .schemas import LabeledExample

def write_jsonl_gz(path: str, records: Iterable[LabeledExample]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")

def read_jsonl_gz(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)  # reconstruct if needed
