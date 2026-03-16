"""
Onset Detection Training
========================
Train a 1D-CNN onset detector on the preprocessed sliding-window dataset.

Features:
  - Session-level train/val/test split (no leakage)
  - Balanced sampling to handle pos/neg imbalance
  - Focal loss option for hard-example mining
  - Per-epoch AUC + P/R/F1 on validation set
  - Saves best checkpoint + scaler (means/stds) for downstream use

Run:
  python3 train_onset.py
  python3 train_onset.py --dataset data/processed/onset_dataset.npz --epochs 100
  python3 train_onset.py --model cnn_large --focal-loss
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from onset_dataset import (
    OnsetWindowDataset,
    load_onset_dataset,
    make_balanced_sampler,
    session_split,
)
from onset_model import build_onset_model


# ── Focal Loss ───────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    """
    Focal loss for binary classification.
    Down-weights easy examples, focuses on hard ones.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze(-1))
        targets = targets.float()

        # Standard BCE terms
        bce = -targets * torch.log(probs + 1e-8) * self.pos_weight \
              - (1 - targets) * torch.log(1 - probs + 1e-8)

        # Focal modulation
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        return (focal_weight * bce).mean()


# ── Metrics ──────────────────────────────────────────────────

def compute_binary_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute P/R/F1/AUC for binary onset classification."""
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0,
    )

    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(auc),
        "accuracy": float((preds == labels).mean()),
        "n_pos": int(labels.sum()),
        "n_neg": int(len(labels) - labels.sum()),
    }


# ── Resolve device ───────────────────────────────────────────

def resolve_device(device: str = "auto") -> torch.device:
    req = (device or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"
    return torch.device(req)


# ── Training loop ────────────────────────────────────────────

def train_onset_detector(
    dataset_path: str,
    checkpoint_path: str = "results/onset_detector.pt",
    scaler_path: str = "results/onset_scaler.npz",
    report_path: str = "results/onset_training_report.json",
    model_name: str = "cnn",
    device_str: str = "auto",
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    balanced: bool = True,
    focal_loss: bool = False,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_str)
    print(f"Device: {device}")

    # Load data
    data = load_onset_dataset(dataset_path)
    windows = data["windows"]
    labels = data["labels"]
    sessions = data["sessions"]

    print(f"Loaded: {len(labels)} windows, {int(labels.sum())} pos, "
          f"{len(labels) - int(labels.sum())} neg")

    # Session-level split
    train_idx, val_idx, test_idx = session_split(sessions, labels, seed=seed)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Compute normalisation on TRAINING data only
    train_windows = windows[train_idx]
    means = train_windows.mean(axis=(0, 1))
    stds = train_windows.std(axis=(0, 1))

    # Build datasets
    train_ds = OnsetWindowDataset(
        windows[train_idx], labels[train_idx],
        augment=True, normalize=True, means=means, stds=stds,
    )
    val_ds = OnsetWindowDataset(
        windows[val_idx], labels[val_idx],
        augment=False, normalize=True, means=means, stds=stds,
    )
    test_ds = OnsetWindowDataset(
        windows[test_idx], labels[test_idx],
        augment=False, normalize=True, means=means, stds=stds,
    )

    # DataLoaders
    if balanced:
        train_sampler = make_balanced_sampler(labels[train_idx])
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    # Model
    n_channels = data["n_channels"]
    model = build_onset_model(model_name, n_channels=n_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params:,} params)")

    # Loss
    if focal_loss:
        criterion = FocalBCELoss(gamma=2.0)
        print("Loss: Focal BCE (gamma=2.0)")
    else:
        pos_weight = torch.tensor(
            [len(labels[train_idx]) / max(1, int(labels[train_idx].sum())) / 2],
            dtype=torch.float32,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Loss: BCE with pos_weight={pos_weight.item():.2f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training
    best_val_f1 = -1.0
    best_state = None
    patience_ctr = 0
    history = []

    print(f"\n{'='*60}")
    print(f"  Training onset detector: {epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits.squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_n += len(yb)
        scheduler.step()

        # Validate
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits.squeeze(-1))
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yb.numpy())

        val_probs = np.concatenate(all_probs)
        val_labels = np.concatenate(all_labels)
        val_metrics = compute_binary_metrics(val_probs, val_labels)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_n, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_info)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}: loss={epoch_info['train_loss']:.4f}  "
                f"val_F1={val_metrics['f1']:.3f}  val_AUC={val_metrics['auc']:.3f}  "
                f"val_P={val_metrics['precision']:.3f}  val_R={val_metrics['recall']:.3f}"
            )

        if patience_ctr >= patience:
            print(f"  Early stop at epoch {epoch+1} (patience={patience})")
            break

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s  |  Best val F1: {best_val_f1:.3f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    test_probs = []
    test_labels_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits.squeeze(-1))
            test_probs.append(probs.cpu().numpy())
            test_labels_list.append(yb.numpy())

    test_probs = np.concatenate(test_probs)
    test_labels_arr = np.concatenate(test_labels_list)
    test_metrics = compute_binary_metrics(test_probs, test_labels_arr)

    print(f"\n  TEST METRICS:")
    print(f"    AUC={test_metrics['auc']:.3f}  F1={test_metrics['f1']:.3f}  "
          f"P={test_metrics['precision']:.3f}  R={test_metrics['recall']:.3f}  "
          f"Acc={test_metrics['accuracy']:.3f}")

    # Save checkpoint
    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_name": model_name,
        "n_channels": n_channels,
        "window_ms": data["window_ms"],
        "stride_ms": data["stride_ms"],
        "label_radius_ms": data["label_radius_ms"],
        "target_rate_hz": data["target_rate_hz"],
        "best_val_f1": best_val_f1,
    }, checkpoint_path)
    print(f"  Saved checkpoint → {checkpoint_path}")

    # Save scaler
    np.savez(scaler_path, means=means, stds=stds)
    print(f"  Saved scaler → {scaler_path}")

    # Save report
    report = {
        "model_name": model_name,
        "n_params": n_params,
        "device": str(device),
        "dataset_path": dataset_path,
        "epochs_run": len(history),
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
        "elapsed_s": elapsed,
    }
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report → {report_path}")


# ── CLI ──────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Train onset detector")
    p.add_argument("--project-root", default="",
                   help="Project root directory. Relative paths resolve from here.")
    p.add_argument("--dataset", default="data/processed/onset_dataset.npz")
    p.add_argument("--checkpoint", default="results/onset_detector.pt")
    p.add_argument("--scaler", default="results/onset_scaler.npz")
    p.add_argument("--report", default="results/onset_training_report.json")
    p.add_argument("--model", choices=["cnn", "cnn_large"], default="cnn")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument("--focal-loss", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Resolve relative paths from project root
    if args.project_root:
        root = os.path.abspath(args.project_root)
        for attr in ("dataset", "checkpoint", "scaler", "report"):
            val = getattr(args, attr)
            if not os.path.isabs(val):
                setattr(args, attr, os.path.join(root, val))

    train_onset_detector(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        scaler_path=args.scaler,
        report_path=args.report,
        model_name=args.model,
        device_str=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        balanced=not args.no_balanced,
        focal_loss=args.focal_loss,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
