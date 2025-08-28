#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, json, math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    n = 0
    total_runtime = 0.0
    with gzip.open(args.path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n += 1
            total_runtime += rec.get("notes",{}).get("runtime_s", 0.0)

    print(f"records: {n}")
    print(f"avg solver runtime: {total_runtime/max(n,1):.6f}s")

if __name__ == "__main__":
    main()
