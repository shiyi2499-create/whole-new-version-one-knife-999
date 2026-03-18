#!/usr/bin/env python3
"""
Password len=8 adaptation experiment.

Protocol:
- baseline training source: single_key + boost merged dataset
- password adaptation source: password parts 1-16 (160 strings) by default
- password held-out test source: password parts 17-20 (40 strings) by default

Outputs:
- zero-shot metrics on held-out password test split
- adapted metrics on the same held-out split
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.join(ROOT, "phase3_password_inception")
if PHASE3_DIR not in sys.path:
    sys.path.insert(0, PHASE3_DIR)

from run_password_closure_inception import (  # noqa: E402
    InceptionTimeClassifier,
    build_no_space_sequences,
    discover_freetype_sessions,
    evaluate_sequences,
    load_final_inception,
    load_merged_training_data,
    resolve_torch_device,
    set_global_seed,
    train_final_inception,
)

try:  # noqa: E402
    from run_password_closure_inception import augment_batch  # type: ignore
except ImportError:  # Backward-compatible fallback for older server copies.
    def augment_batch(X_batch: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        B, T, C = X_batch.shape
        X_aug = X_batch.clone()
        for i in range(B):
            if np.random.random() > p:
                continue
            aug_type = np.random.choice(["shift", "noise", "scale", "ch_drop"])
            if aug_type == "shift":
                shift = np.random.randint(-T // 10, T // 10 + 1)
                X_aug[i] = torch.roll(X_aug[i], shifts=shift, dims=0)
            elif aug_type == "noise":
                std = X_aug[i].std() * 0.01
                X_aug[i] += torch.randn_like(X_aug[i]) * std
            elif aug_type == "scale":
                scale = 0.8 + 0.4 * np.random.random()
                X_aug[i] *= scale
            elif aug_type == "ch_drop":
                ch = np.random.randint(0, C)
                X_aug[i][:, ch] = 0.0
        return X_aug


PART_RE = re.compile(r"_part(\d+)_")


def parse_part(session_prefix: str) -> int:
    m = PART_RE.search(os.path.basename(session_prefix))
    if not m:
        raise ValueError(f"Could not parse part number from session: {session_prefix}")
    return int(m.group(1))


def split_sequences_by_parts(sequences: list[dict], adapt_parts: set[int], test_parts: set[int]):
    adapt = []
    test = []
    for seq in sequences:
        part = parse_part(seq["session"])
        if part in adapt_parts:
            adapt.append(seq)
        elif part in test_parts:
            test.append(seq)
    return adapt, test


def session_is_complete(session_prefix: str) -> bool:
    required = [
        f"{session_prefix}_sensor.csv",
        f"{session_prefix}_events.csv",
        f"{session_prefix}_prompts.csv",
    ]
    return all(os.path.exists(path) for path in required)


def select_latest_complete_sessions(sessions: list[str]) -> list[str]:
    by_part: dict[int, list[str]] = {}
    for sess in sessions:
        try:
            part = parse_part(sess)
        except ValueError:
            continue
        by_part.setdefault(part, []).append(sess)

    selected = []
    for part in sorted(by_part):
        candidates = sorted(by_part[part])
        complete = [sess for sess in candidates if session_is_complete(sess)]
        if complete:
            selected.append(complete[-1])
        else:
            selected.append(candidates[-1])
    return selected


def flatten_items(sequences: list[dict], class_to_idx: dict[str, int]):
    X = []
    y = []
    for seq in sequences:
        for item in seq["items"]:
            key = str(item["key"])
            if key not in class_to_idx:
                continue
            X.append(item["window"].astype(np.float32))
            y.append(class_to_idx[key])
    if not X:
        raise RuntimeError("No adaptation items collected from password sequences.")
    return np.stack(X), np.asarray(y, dtype=np.int64)


def normalize_windows(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    Xn = X.copy()
    for ch in range(Xn.shape[2]):
        Xn[:, :, ch] = (Xn[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)
    return Xn


def fine_tune_on_password(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
    head_epochs: int = 20,
    full_epochs: int = 60,
    head_lr: float = 5e-4,
    full_lr: float = 2e-4,
    patience: int = 15,
):
    counts = Counter(y.tolist())
    stratify_ok = min(counts.values()) >= 2 and len(X) >= len(counts) * 2
    idx = np.arange(len(X))
    tr_idx, val_idx = train_test_split(
        idx,
        test_size=max(0.15, 1 / len(idx)),
        random_state=42,
        stratify=y if stratify_ok else None,
    )

    X_tr = torch.tensor(X[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.long)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_idx], dtype=torch.long).to(device)

    dl_gen = torch.Generator()
    dl_gen.manual_seed(42)
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        generator=dl_gen,
    )

    criterion = torch.nn.CrossEntropyLoss()

    def run_stage(stage_name: str, epochs: int, lr: float, trainable_selector):
        for p in model.parameters():
            p.requires_grad = False
        for p in trainable_selector():
            p.requires_grad = True

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_val = -1.0
        best_state = None
        patience_ctr = 0

        print(f"  {stage_name}")
        for epoch in range(epochs):
            model.train()
            correct = 0
            total = 0
            for xb, yb in train_loader:
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
                best_state = copy.deepcopy(model.state_dict())
                patience_ctr = 0
            else:
                patience_ctr += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                train_acc = correct / max(total, 1)
                print(f"    Epoch {epoch+1:3d}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
            if patience_ctr >= patience:
                print(f"    Early stop at epoch {epoch+1} (patience={patience})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

    run_stage("Stage 1/2: head warm-up", head_epochs, head_lr, lambda: model.head.parameters())
    run_stage("Stage 2/2: full-network fine-tune", full_epochs, full_lr, lambda: model.parameters())
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Password len=8 adaptation experiment")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--merged-path", default="data/processed/merged_dataset.npz")
    p.add_argument("--password-dir", default="data/raw/password/len_8")
    p.add_argument("--checkpoint-path", default="results/inception_password_final.pt")
    p.add_argument("--scaler-path", default="results/inception_password_scaler.npz")
    p.add_argument("--report-path", default="results/password_len8_adaptation.json")
    p.add_argument("--force-train-baseline", action="store_true")
    p.add_argument("--baseline-epochs", type=int, default=280)
    p.add_argument("--baseline-batch-size", type=int, default=32)
    p.add_argument("--baseline-lr", type=float, default=8e-4)
    p.add_argument("--baseline-patience", type=int, default=60)
    p.add_argument("--head-epochs", type=int, default=20)
    p.add_argument("--full-epochs", type=int, default=60)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--full-lr", type=float, default=2e-4)
    p.add_argument("--adapt-batch-size", type=int, default=32)
    p.add_argument("--adapt-part-end", type=int, default=16, help="Use parts 1..N for password adaptation")
    p.add_argument("--test-part-start", type=int, default=17, help="Use parts N..end for held-out password test")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(42)
    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    X, y, y_enc, classes = load_merged_training_data(args.merged_path)
    print(f"Loaded baseline training data: X={X.shape}, classes={len(classes)}")
    train_kwargs = dict(
        X=X,
        y_enc=y_enc,
        classes=classes,
        checkpoint_path=args.checkpoint_path,
        scaler_path=args.scaler_path,
        device=device,
        force=args.force_train_baseline,
        epochs=args.baseline_epochs,
        batch_size=args.baseline_batch_size,
        lr=args.baseline_lr,
        patience=args.baseline_patience,
    )
    # Keep compatibility with both the new password-route trainer (which
    # accepts augment=...) and older server copies that do not.
    if "augment" in inspect.signature(train_final_inception).parameters:
        train_kwargs["augment"] = True
    train_final_inception(**train_kwargs)

    model, classes, means, stds = load_final_inception(args.checkpoint_path, args.scaler_path, device)
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}

    sessions = discover_freetype_sessions([args.password_dir])
    if not sessions:
        raise RuntimeError("No password sessions found.")
    sessions = select_latest_complete_sessions(sessions)
    print(f"Found {len(sessions)} selected password sessions")

    sequences = []
    for sess in sessions:
        part = parse_part(sess)
        seqs = build_no_space_sequences(sess, yes_only=True, eval_max_sequences=0)
        print(f"  part {part}: {len(seqs)} sequences")
        sequences.extend(seqs)

    adapt_parts = set(range(1, args.adapt_part_end + 1))
    test_parts = set(range(args.test_part_start, max(parse_part(s) for s in sessions) + 1))
    adapt_sequences, test_sequences = split_sequences_by_parts(sequences, adapt_parts, test_parts)
    print(f"Adapt sequences: {len(adapt_sequences)} | Test sequences: {len(test_sequences)}")

    zero_shot_metrics, _ = evaluate_sequences(
        test_sequences,
        model,
        classes,
        means,
        stds,
        device,
        char_topk=(1, 3, 5),
        seq_branch_topk=5,
        seq_beam_width=100,
        seq_hit_cutoffs=(10, 50, 100),
    )

    X_adapt, y_adapt = flatten_items(adapt_sequences, class_to_idx)
    X_adapt = normalize_windows(X_adapt, means, stds)
    print(f"Adapt windows: {X_adapt.shape}")
    model = fine_tune_on_password(
        model=model,
        X=X_adapt,
        y=y_adapt,
        device=device,
        batch_size=args.adapt_batch_size,
        head_epochs=args.head_epochs,
        full_epochs=args.full_epochs,
        head_lr=args.head_lr,
        full_lr=args.full_lr,
    )

    adapted_metrics, examples = evaluate_sequences(
        test_sequences,
        model,
        classes,
        means,
        stds,
        device,
        char_topk=(1, 3, 5),
        seq_branch_topk=5,
        seq_beam_width=100,
        seq_hit_cutoffs=(10, 50, 100),
    )

    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_timesteps": int(X_adapt.shape[1]),
        "n_channels": int(X_adapt.shape[2]),
        "n_classes": int(len(classes)),
        "classes": classes,
        "model_name": "InceptionTime",
        "adapted_from_password_len8": True,
        "adapt_parts": sorted(adapt_parts),
        "test_parts": sorted(test_parts),
    }, args.checkpoint_path)
    if not os.path.exists(args.scaler_path):
        np.savez(args.scaler_path, means=means, stds=stds)

    report = {
        "device": str(device),
        "merged_path": args.merged_path,
        "password_dir": args.password_dir,
        "checkpoint_path": args.checkpoint_path,
        "adapt_parts": sorted(adapt_parts),
        "test_parts": sorted(test_parts),
        "zero_shot_metrics": zero_shot_metrics,
        "adapted_metrics": adapted_metrics,
        "examples": examples[:20],
    }
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nZERO-SHOT METRICS")
    print(f"  char_top1: {zero_shot_metrics['char_top1_accuracy']:.1%}")
    print(f"  char_top3: {zero_shot_metrics['char_top3_accuracy']:.1%}")
    print(f"  char_top5: {zero_shot_metrics['char_top5_accuracy']:.1%}")
    print(f"  seq_top100: {zero_shot_metrics['sequence_top100_hit_rate']:.1%}")
    print(f"  CER: {zero_shot_metrics['cer_top1']:.1%}")

    print("\nADAPTED METRICS")
    print(f"  char_top1: {adapted_metrics['char_top1_accuracy']:.1%}")
    print(f"  char_top3: {adapted_metrics['char_top3_accuracy']:.1%}")
    print(f"  char_top5: {adapted_metrics['char_top5_accuracy']:.1%}")
    print(f"  seq_top100: {adapted_metrics['sequence_top100_hit_rate']:.1%}")
    print(f"  CER: {adapted_metrics['cer_top1']:.1%}")
    print(f"Saved adapted checkpoint -> {args.checkpoint_path}")
    print(f"Saved -> {args.report_path}")


if __name__ == "__main__":
    main()
