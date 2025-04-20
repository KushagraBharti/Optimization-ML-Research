# src/train/prepare_dataset.py

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from src.solvers.dp import cover_min_length

# Constants
MAX_SEGMENTS  = 30
MAX_ENDPOINTS = 2 * MAX_SEGMENTS   # up to 60
MAX_SEQ_LEN   = MAX_ENDPOINTS + 1  # one STOP token

def encode_instance(
    segments: List[Tuple[float, float]],
    L: float,
) -> dict:
    """
    Encode one raw instance into fixed-size tensors.
    """
    # flatten endpoints
    endpoints = []
    for a, b in segments:
        endpoints.extend([a, b])
    E = len(endpoints)
    assert E <= MAX_ENDPOINTS

    # compute target sequence using DP tours
    dp_tours = cover_min_length(segments, L)
    seq = []
    # map tours to pointer indices
    for p, q in dp_tours:
        i = endpoints.index(p)
        j = endpoints.index(q)
        seq += [i, j]

    # STOP is always the last slot
    STOP = MAX_ENDPOINTS
    seq.append(STOP)

    # pad/truncate to MAX_SEQ_LEN
    if len(seq) < MAX_SEQ_LEN:
        seq += [STOP] * (MAX_SEQ_LEN - len(seq))
    else:
        seq = seq[:MAX_SEQ_LEN]

    # build tensors
    endpoints_tensor = torch.zeros(MAX_ENDPOINTS, dtype=torch.float32)
    mask = torch.zeros(MAX_ENDPOINTS, dtype=torch.bool)
    for idx, val in enumerate(endpoints):
        endpoints_tensor[idx] = val
        mask[idx] = True

    seq_tensor = torch.tensor(seq, dtype=torch.long)  # (MAX_SEQ_LEN,)
    L_tensor = torch.tensor([L], dtype=torch.float32) # (1,)

    return {
        "endpoints": endpoints_tensor,  # (MAX_ENDPOINTS,)
        "mask": mask,                   # (MAX_ENDPOINTS,)
        "L": L_tensor,                  # (1,)
        "target_seq": seq_tensor,       # (MAX_SEQ_LEN,)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=str,   default="data/raw/train.jsonl")
    parser.add_argument("--out-dir", type=str,   default="data/processed")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    lines = Path(args.input).read_text().strip().splitlines()
    random.shuffle(lines)
    n = len(lines)
    n_train = int(n * args.train_frac)
    splits = {
        "train": lines[:n_train],
        "val":   lines[n_train: n_train + (n - n_train)//2],
        "test":  lines[n_train + (n - n_train)//2:]
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, subset in splits.items():
        examples = []
        for line in subset:
            rec = json.loads(line)
            segs = [tuple(s) for s in rec["segments"]]
            L = rec["L"]
            encoded = encode_instance(segs, L)
            examples.append(encoded)
        torch.save(examples, out_dir / f"{split}.pt")
        print(f"Wrote {len(examples)} examples to {split}.pt")

if __name__ == "__main__":
    main()
