"""
Onset / Password-Boundary Training
==================================

Train one of:
  - onset              : binary keystroke onset detector
  - activity           : legacy binary keyboard-active detector
  - password_boundary  : 4-class password-boundary model

The new main task is `password_boundary`.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Optional

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


# ── Losses ───────────────────────────────────────────────────

class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze(-1))
        targets = targets.float()
        bce = -targets * torch.log(probs + 1e-8) * self.pos_weight - (1 - targets) * torch.log(1 - probs + 1e-8)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class FocalCrossEntropyLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        ce = nn.functional.nll_loss(log_probs, targets.long(), weight=self.class_weights, reduction="none")
        pt = probs.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt) ** self.gamma
        return (focal * ce).mean()


# ── Metrics ──────────────────────────────────────────────────

def compute_binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    labels = labels.astype(int)
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
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



def compute_multiclass_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    label_names: list[str] | np.ndarray,
) -> dict:
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

    labels = labels.astype(int)
    preds = np.argmax(probs, axis=1).astype(int)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        labels, preds, average=None, labels=np.arange(len(label_names)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=np.arange(len(label_names)))
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            str(label_names[i]): {
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "f1": float(per_f1[i]),
                "support": int(per_support[i]),
            }
            for i in range(len(label_names))
        },
    }


# ── Device ───────────────────────────────────────────────────

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


# ── Model/task helpers ───────────────────────────────────────

def default_paths_for_task(task: str) -> tuple[str, str, str]:
    if task == "password_boundary":
        return (
            "results/password_boundary_detector.pt",
            "results/password_boundary_scaler.npz",
            "results/password_boundary_training_report.json",
        )
    if task == "password_segment":
        return (
            "results/password_segment_detector.pt",
            "results/password_segment_scaler.npz",
            "results/password_segment_training_report.json",
        )
    if task == "activity":
        return (
            "results/activity_detector.pt",
            "results/activity_scaler.npz",
            "results/activity_training_report.json",
        )
    return (
        "results/onset_detector.pt",
        "results/onset_scaler.npz",
        "results/onset_training_report.json",
    )


# ── Training ─────────────────────────────────────────────────

def train_onset_detector(
    dataset_path: str,
    checkpoint_path: str,
    scaler_path: str,
    report_path: str,
    model_name: str = "cnn",
    device_str: str = "auto",
    epochs: int = 80,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 20,
    balanced: bool = True,
    focal_loss: bool = False,
    seed: int = 42,
    task: str = "onset",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device_str)
    print(f"Device: {device}")

    data = load_onset_dataset(dataset_path)
    windows = data["windows"]
    labels = data["labels"]
    sessions = data["sessions"]
    task = str(data.get("task", task) or task)
    label_names = data.get("label_names", np.array(["negative", "positive"])).tolist()
    n_classes = int(data.get("n_classes", len(label_names)))
    is_multiclass = n_classes > 2 or int(np.max(labels, initial=0)) > 1

    print(f"Task: {task}")
    print(f"Loaded dataset: {len(labels)} windows across {len(set(sessions.tolist()))} sessions")

    train_idx, val_idx, test_idx = session_split(sessions, labels, seed=seed)
    print(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_windows = windows[train_idx]
    means = train_windows.mean(axis=(0, 1)) if len(train_windows) else np.zeros(windows.shape[-1], dtype=np.float32)
    stds = train_windows.std(axis=(0, 1)) if len(train_windows) else np.ones(windows.shape[-1], dtype=np.float32)

    train_ds = OnsetWindowDataset(windows[train_idx], labels[train_idx], augment=True, normalize=True, means=means, stds=stds, n_classes=n_classes)
    val_ds = OnsetWindowDataset(windows[val_idx], labels[val_idx], augment=False, normalize=True, means=means, stds=stds, n_classes=n_classes)
    test_ds = OnsetWindowDataset(windows[test_idx], labels[test_idx], augment=False, normalize=True, means=means, stds=stds, n_classes=n_classes)

    if balanced and len(train_idx):
        train_sampler = make_balanced_sampler(labels[train_idx], n_classes=n_classes)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    if task == "activity" and model_name == "cnn":
        model_name = "activity_cnn"
    elif task == "password_boundary" and model_name == "cnn":
        model_name = "password_boundary_cnn"

    model = build_onset_model(model_name, n_channels=data["n_channels"], n_classes=(n_classes if is_multiclass else 1), task=task).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} ({n_params:,} params)")

    label_counts = np.bincount(labels[train_idx], minlength=n_classes)
    class_weights_np = len(train_idx) / np.maximum(label_counts, 1)
    class_weights_np = class_weights_np / max(class_weights_np.mean(), 1e-8)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    if is_multiclass:
        if focal_loss:
            criterion = FocalCrossEntropyLoss(gamma=2.0, class_weights=class_weights)
            print("Loss: focal cross entropy")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Loss: cross entropy with class weights {class_weights_np.round(3).tolist()}")
    else:
        if focal_loss:
            pos_weight = float(class_weights_np[1]) if len(class_weights_np) > 1 else 1.0
            criterion = FocalBCELoss(gamma=2.0, pos_weight=pos_weight)
            print(f"Loss: focal BCE (pos_weight={pos_weight:.3f})")
        else:
            pos_weight = torch.tensor([float(class_weights_np[1]) if len(class_weights_np) > 1 else 1.0], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight.item():.3f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_metric = -1.0
    best_state = None
    history = []
    patience_ctr = 0

    metric_name = "macro_f1" if is_multiclass else "f1"

    print(f"\n{'='*60}")
    print(f"  Training {task}: epochs={epochs} batch={batch_size} lr={lr}")
    print(f"{'='*60}\n")

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            if is_multiclass:
                loss = criterion(logits, yb.long())
            else:
                loss = criterion(logits.squeeze(-1), yb.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
            train_n += len(yb)
        scheduler.step()

        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                if is_multiclass:
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(yb.numpy())
                else:
                    probs = torch.sigmoid(logits.squeeze(-1))
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(yb.numpy())

        val_probs = np.concatenate(all_probs) if all_probs else np.array([])
        val_labels = np.concatenate(all_labels) if all_labels else np.array([])
        if is_multiclass:
            val_metrics = compute_multiclass_metrics(val_probs, val_labels, label_names)
        else:
            val_metrics = compute_binary_metrics(val_probs, val_labels)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(train_n, 1),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"},
        }
        history.append(epoch_info)

        score = float(val_metrics[metric_name])
        if score > best_metric:
            best_metric = score
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            if is_multiclass:
                print(
                    f"  Epoch {epoch+1:3d}: loss={epoch_info['train_loss']:.4f}  "
                    f"val_macroF1={val_metrics['macro_f1']:.3f}  val_acc={val_metrics['accuracy']:.3f}"
                )
            else:
                print(
                    f"  Epoch {epoch+1:3d}: loss={epoch_info['train_loss']:.4f}  "
                    f"val_F1={val_metrics['f1']:.3f}  val_AUC={val_metrics['auc']:.3f}  "
                    f"val_P={val_metrics['precision']:.3f}  val_R={val_metrics['recall']:.3f}"
                )
        if patience_ctr >= patience:
            print(f"  Early stop at epoch {epoch+1} (patience={patience})")
            break

    elapsed = time.time() - t0
    print(f"\n  Training complete in {elapsed:.1f}s  |  Best val {metric_name}: {best_metric:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_probs = []
    test_labels_list = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            if is_multiclass:
                probs = torch.softmax(logits, dim=-1)
                test_probs.append(probs.cpu().numpy())
                test_labels_list.append(yb.numpy())
            else:
                probs = torch.sigmoid(logits.squeeze(-1))
                test_probs.append(probs.cpu().numpy())
                test_labels_list.append(yb.numpy())

    test_probs_arr = np.concatenate(test_probs) if test_probs else np.array([])
    test_labels_arr = np.concatenate(test_labels_list) if test_labels_list else np.array([])
    if is_multiclass:
        test_metrics = compute_multiclass_metrics(test_probs_arr, test_labels_arr, label_names)
        print(f"\n  TEST METRICS: macroF1={test_metrics['macro_f1']:.3f}  weightedF1={test_metrics['weighted_f1']:.3f}  Acc={test_metrics['accuracy']:.3f}")
        for cls_name, cls_m in test_metrics["per_class"].items():
            print(f"    {cls_name:16s} P={cls_m['precision']:.3f}  R={cls_m['recall']:.3f}  F1={cls_m['f1']:.3f}  n={cls_m['support']}")
    else:
        test_metrics = compute_binary_metrics(test_probs_arr, test_labels_arr)
        print(
            f"\n  TEST METRICS: AUC={test_metrics['auc']:.3f}  F1={test_metrics['f1']:.3f}  "
            f"P={test_metrics['precision']:.3f}  R={test_metrics['recall']:.3f}  Acc={test_metrics['accuracy']:.3f}"
        )

    ckpt_dir = os.path.dirname(checkpoint_path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_name": model_name,
        "n_channels": int(data["n_channels"]),
        "window_ms": int(data["window_ms"]),
        "stride_ms": int(data["stride_ms"]),
        "label_radius_ms": int(data["label_radius_ms"]),
        "target_rate_hz": int(data["target_rate_hz"]),
        "best_val_metric": float(best_metric),
        "task": task,
        "n_classes": int(n_classes if is_multiclass else 1 if n_classes <= 2 else n_classes),
        "label_names": label_names,
    }, checkpoint_path)
    np.savez(scaler_path, means=means, stds=stds)

    print(f"  Saved checkpoint → {checkpoint_path}")
    print(f"  Saved scaler → {scaler_path}")

    report = {
        "task": task,
        "model_name": model_name,
        "n_params": n_params,
        "device": str(device),
        "dataset_path": dataset_path,
        "epochs_run": len(history),
        "best_val_metric": float(best_metric),
        "metric_name": metric_name,
        "test_metrics": test_metrics,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "elapsed_s": float(elapsed),
        "label_names": label_names,
        "class_weights": class_weights_np.tolist(),
    }
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report → {report_path}")


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train onset / password-boundary detector")
    parser.add_argument("--dataset", default="data/processed/onset_dataset.npz")
    parser.add_argument("--task", choices=["onset", "activity", "password_boundary", "password_segment"], default="onset")
    parser.add_argument("--model", default="cnn", help="cnn | cnn_large | activity_cnn | password_boundary_cnn")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--scaler", default="")
    parser.add_argument("--report", default="")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-balanced", action="store_true")
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ckpt_default, scaler_default, report_default = default_paths_for_task(args.task)
    checkpoint = args.checkpoint or ckpt_default
    scaler = args.scaler or scaler_default
    report = args.report or report_default

    if args.task == "password_boundary" and args.dataset == "data/processed/onset_dataset.npz":
        args.dataset = "data/processed/password_boundary_dataset.npz"
    elif args.task == "password_segment" and args.dataset == "data/processed/onset_dataset.npz":
        args.dataset = "data/processed/password_segment_dataset.npz"
    elif args.task == "activity" and args.dataset == "data/processed/onset_dataset.npz":
        args.dataset = "data/processed/activity_dataset.npz"

    train_onset_detector(
        dataset_path=args.dataset,
        checkpoint_path=checkpoint,
        scaler_path=scaler,
        report_path=report,
        model_name=args.model,
        device_str=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        balanced=not args.no_balanced,
        focal_loss=args.focal_loss,
        seed=args.seed,
        task=args.task,
    )


if __name__ == "__main__":
    main()
