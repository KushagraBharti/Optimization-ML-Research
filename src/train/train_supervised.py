# src/train/train_supervised.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from src.models.pointer_net import PointerNet

# turn on anomaly detection for NaN/Infs
torch.autograd.set_detect_anomaly(True)

def collate_fn(batch):
    endpoints   = torch.stack([ex["endpoints"]    for ex in batch], dim=0)
    mask        = torch.stack([ex["mask"]         for ex in batch], dim=0)
    L           = torch.cat ([ex["L"]            for ex in batch], dim=0).unsqueeze(-1)
    target_seq  = torch.stack([ex["target_seq"]   for ex in batch], dim=0)
    return endpoints, mask, L, target_seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=str, default="data/processed")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)  # lowered LR
    parser.add_argument("--clip-norm",  type=float, default=1.0)
    parser.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    # load data
    train_data = torch.load(Path(args.data_dir) / "train.pt")
    val_data   = torch.load(Path(args.data_dir) / "val.pt")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model  = PointerNet(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    for epoch in range(1, args.epochs + 1):
        # TRAIN
        model.train()
        total_loss = 0.0
        for endpoints, mask, L, target_seq in train_loader:
            endpoints, mask, L, target_seq = endpoints.to(device), mask.to(device), L.to(device), target_seq.to(device)
            B, T = target_seq.size()

            optimizer.zero_grad()
            log_probs = model(endpoints, mask, L, target_seq)  # (B, T, E+1)

            loss = torch.tensor(0.0, device=device)
            for t in range(T):
                step_loss = criterion(log_probs[:, t, :], target_seq[:, t])
                if not torch.isfinite(step_loss):
                    raise RuntimeError(f"Non‑finite loss at epoch {epoch}, step {t}: {step_loss.item()}")
                loss = loss + step_loss
            loss = loss / T

            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            total_loss += loss.item() * B

        avg_train = total_loss / len(train_loader.dataset)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for endpoints, mask, L, target_seq in val_loader:
                endpoints, mask, L, target_seq = endpoints.to(device), mask.to(device), L.to(device), target_seq.to(device)
                log_probs = model(endpoints, mask, L, target_seq)
                l = torch.tensor(0.0, device=device)
                for t in range(target_seq.size(1)):
                    l_step = criterion(log_probs[:, t, :], target_seq[:, t])
                    if not torch.isfinite(l_step):
                        raise RuntimeError(f"Non‑finite val loss at epoch {epoch}, step {t}: {l_step.item()}")
                    l = l + l_step
                val_loss += (l / target_seq.size(1)).item() * endpoints.size(0)

        avg_val = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        torch.save(model.state_dict(), f"checkpoints/pointer_epoch{epoch}.pt")

if __name__ == "__main__":
    main()
