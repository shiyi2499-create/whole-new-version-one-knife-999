#!/usr/bin/env python3
"""
Diagnostic: evaluate the current password-route Inception trainer on held-out
single-key data.

Purpose:
- verify whether the current Phase 3 Inception training recipe can still
  reproduce a strong isolated-key baseline on the same 57-step data space
- separate "trainer/recipe weakness" from "zero-shot password domain gap"
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from run_password_closure_inception import (
    InceptionTimeClassifier,
    augment_batch,
    load_merged_training_data,
    resolve_torch_device,
    set_global_seed,
)


def topk_accuracy(prob: np.ndarray, y_true: np.ndarray, k: int) -> float:
    topk = np.argsort(prob, axis=1)[:, ::-1][:, :k]
    hits = sum(int(y in row) for y, row in zip(y_true.tolist(), topk.tolist()))
    return hits / max(len(y_true), 1)


def train_eval_singlekey(
    X: np.ndarray,
    y_enc: np.ndarray,
    classes: np.ndarray,
    device: torch.device,
    epochs: int = 280,
    batch_size: int = 32,
    lr: float = 8e-4,
    patience: int = 60,
    test_size: float = 0.2,
    augment: bool = True,
):
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=42,
        stratify=y_enc,
    )

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]
    y_train = y_enc[train_idx]
    y_test = y_enc[test_idx]

    ch_means = X_train_raw.mean(axis=(0, 1))
    ch_stds = X_train_raw.std(axis=(0, 1))

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    for ch in range(X.shape[2]):
        X_train[:, :, ch] = (X_train[:, :, ch] - ch_means[ch]) / (ch_stds[ch] + 1e-10)
        X_test[:, :, ch] = (X_test[:, :, ch] - ch_means[ch]) / (ch_stds[ch] + 1e-10)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    model = InceptionTimeClassifier(
        n_timesteps=X.shape[1],
        n_channels=X.shape[2],
        n_classes=len(classes),
    ).to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    dl_gen = torch.Generator()
    dl_gen.manual_seed(42)
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=dl_gen,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = -1.0
    best_state = None
    patience_ctr = 0

    # carve out validation from train split for recipe-consistent early stopping
    tr_idx, val_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=max(0.1, 1 / len(X_train)),
        random_state=42,
        stratify=y_train,
    )
    X_tr = X_train_t[tr_idx]
    y_tr = y_train_t[tr_idx]
    X_val = X_train_t[val_idx].to(device)
    y_val = y_train_t[val_idx].to(device)
    tr_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=dl_gen,
    )

    print("Training diagnostic InceptionTime on held-out single_key split")
    print(f"  Samples: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)} | Classes: {len(classes)}")
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for xb, yb in tr_loader:
            if augment:
                xb = augment_batch(xb, p=0.5)
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(len(yb))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc = float((val_logits.argmax(1) == y_val).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_acc = correct / max(total, 1)
            print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if patience_ctr >= patience:
            print(f"  Early stop at epoch {epoch+1} (patience={patience})")
            break

    if best_state is None:
        raise RuntimeError("No best_state captured during diagnostic training.")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()

    metrics = {
        "n_samples_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(len(classes)),
        "val_best_accuracy": float(best_val),
        "test_top1_accuracy": float((pred == y_test).mean()),
        "test_top3_accuracy": float(topk_accuracy(probs, y_test, 3)),
        "test_top5_accuracy": float(topk_accuracy(probs, y_test, 5)),
        "classes": classes.tolist(),
    }
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Diagnostic single_key eval for the password Inception trainer")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--merged-path", default="data/processed/merged_dataset.npz")
    p.add_argument("--report-path", default="results/diagnose_singlekey_inception.json")
    p.add_argument("--epochs", type=int, default=280)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--train-max-samples", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(42)
    device = resolve_torch_device(args.device)
    X, y, y_enc, classes = load_merged_training_data(args.merged_path, max_samples=args.train_max_samples)
    print(f"Device: {device}")
    print(f"Loaded merged training data: X={X.shape}, classes={len(classes)}")
    metrics = train_eval_singlekey(
        X=X,
        y_enc=y_enc,
        classes=classes,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        test_size=args.test_size,
        augment=(not args.no_augment),
    )
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump({"device": str(device), "merged_path": args.merged_path, "metrics": metrics}, f, indent=2)
    print("\nDIAGNOSTIC METRICS")
    print(f"  val_best_accuracy:  {metrics['val_best_accuracy']:.1%}")
    print(f"  test_top1_accuracy: {metrics['test_top1_accuracy']:.1%}")
    print(f"  test_top3_accuracy: {metrics['test_top3_accuracy']:.1%}")
    print(f"  test_top5_accuracy: {metrics['test_top5_accuracy']:.1%}")
    print(f"Saved -> {args.report_path}")


if __name__ == "__main__":
    main()
