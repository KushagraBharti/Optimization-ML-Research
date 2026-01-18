# scripts/export_one_shard_to_jsonl.py

from pathlib import Path
import json
import pyarrow.parquet as pq

from coverage_planning.data.schemas import Sample

def main():
    # pick one shard from your demo dataset
    shard = Path("data/demo_minlength/train_000.parquet")
    out = Path("data/demo_minlength/train_000.jsonl")

    table = pq.read_table(shard)
    # assuming each row is a dict-like sample payload
    with out.open("w", encoding="utf-8") as f:
        for row in table.to_pylist():
            # If the row is already in the same shape as Sample, just dump it
            json.dump(row, f)
            f.write("\n")

if __name__ == "__main__":
    main()
