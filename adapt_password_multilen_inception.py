#!/usr/bin/env python3
"""
Incremental multi-length password adaptation for the Inception classifier.

Typical use:

python adapt_password_multilen_inception.py \
  --base-checkpoint results/inception_password_final.pt \
  --base-scaler results/inception_password_scaler.npz \
  --password-dir data/raw/password/len_8 \
  --adapt-part-end 16 \
  --test-part-start 17 \
  --password-dir data/raw/password/len9 \
  --adapt-part-end 4 \
  --test-part-start 5 \
  --output-checkpoint results/inception_password_len8_len9.pt \
  --output-scaler results/inception_password_len8_len9_scaler.npz \
  --report-path results/password_len8_len9_adaptation.json
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
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

from adapt_password_len8_inception import (  # noqa: E402
    flatten_items,
    normalize_windows,
    parse_part,
    select_latest_complete_sessions,
    split_sequences_by_parts,
)
from phase3_password_inception.run_password_closure_inception import (  # noqa: E402
    discover_freetype_sessions,
    evaluate_sequences,
    load_final_inception,
    resolve_torch_device,
    set_global_seed,
)

try:  # noqa: E402
    from run_password_closure_inception import augment_batch  # type: ignore
except ImportError:  # pragma: no cover
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
    p = argparse.ArgumentParser(description="Incremental multi-length password adaptation")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--base-checkpoint", default="results/inception_password_final.pt")
    p.add_argument("--base-scaler", default="results/inception_password_scaler.npz")
    p.add_argument("--output-checkpoint", required=True)
    p.add_argument("--output-scaler", required=True)
    p.add_argument("--report-path", required=True)
    p.add_argument("--password-dir", action="append", required=True)
    p.add_argument("--adapt-part-end", action="append", type=int, required=True)
    p.add_argument("--test-part-start", action="append", type=int, required=True)
    p.add_argument("--head-epochs", type=int, default=20)
    p.add_argument("--full-epochs", type=int, default=60)
    p.add_argument("--head-lr", type=float, default=5e-4)
    p.add_argument("--full-lr", type=float, default=2e-4)
    p.add_argument("--adapt-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_sequences(password_dir: str):
    sessions = discover_freetype_sessions([password_dir])
    if not sessions:
        raise RuntimeError(f"No password sessions found in {password_dir}")
    sessions = select_latest_complete_sessions(sessions)
    seqs = []
    from phase3_password_inception.run_password_closure_inception import build_no_space_sequences  # noqa: E402
    for sess in sessions:
        seqs.extend(build_no_space_sequences(sess, yes_only=True, eval_max_sequences=0))
    return sessions, seqs


def metrics_subset(metrics: dict) -> dict:
    keep = [
        "char_top1_accuracy",
        "char_top3_accuracy",
        "char_top5_accuracy",
        "sequence_top100_hit_rate",
        "cer_top1",
    ]
    return {k: metrics[k] for k in keep}


def main():
    args = parse_args()
    if not (len(args.password_dir) == len(args.adapt_part_end) == len(args.test_part_start)):
        raise ValueError("--password-dir / --adapt-part-end / --test-part-start must have equal length")

    set_global_seed(args.seed)
    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    model, classes, means, stds = load_final_inception(args.base_checkpoint, args.base_scaler, device)
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}
    print(f"Loaded base classifier: classes={len(classes)}")

    dataset_rows = []
    combined_adapt = []
    combined_tests = []
    test_sets = []

    for password_dir, adapt_end, test_start in zip(args.password_dir, args.adapt_part_end, args.test_part_start):
        sessions, sequences = load_sequences(password_dir)
        max_part = max(parse_part(s) for s in sessions)
        adapt_parts = set(range(1, adapt_end + 1))
        test_parts = set(range(test_start, max_part + 1))
        adapt_sequences, test_sequences = split_sequences_by_parts(sequences, adapt_parts, test_parts)
        combined_adapt.extend(adapt_sequences)
        combined_tests.extend(test_sequences)
        test_sets.append((password_dir, test_sequences))
        dataset_rows.append({
            "password_dir": password_dir,
            "num_sessions": len(sessions),
            "num_sequences": len(sequences),
            "adapt_parts": sorted(adapt_parts),
            "test_parts": sorted(test_parts),
            "adapt_sequences": len(adapt_sequences),
            "test_sequences": len(test_sequences),
        })
        print(f"[dataset] {password_dir}")
        print(f"  sessions={len(sessions)} sequences={len(sequences)} adapt={len(adapt_sequences)} test={len(test_sequences)}")

    zero_shot = {"per_dir": {}, "combined": None}
    for password_dir, test_sequences in test_sets:
        metrics, _ = evaluate_sequences(
            test_sequences, model, classes, means, stds, device,
            char_topk=(1, 3, 5), seq_branch_topk=5, seq_beam_width=100, seq_hit_cutoffs=(10, 50, 100),
        )
        zero_shot["per_dir"][password_dir] = metrics_subset(metrics)
    combined_metrics, _ = evaluate_sequences(
        combined_tests, model, classes, means, stds, device,
        char_topk=(1, 3, 5), seq_branch_topk=5, seq_beam_width=100, seq_hit_cutoffs=(10, 50, 100),
    )
    zero_shot["combined"] = metrics_subset(combined_metrics)

    X_adapt, y_adapt = flatten_items(combined_adapt, class_to_idx)
    X_adapt = normalize_windows(X_adapt, means, stds)
    print(f"Combined adaptation windows: {X_adapt.shape}")

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

    adapted = {"per_dir": {}, "combined": None}
    for password_dir, test_sequences in test_sets:
        metrics, _ = evaluate_sequences(
            test_sequences, model, classes, means, stds, device,
            char_topk=(1, 3, 5), seq_branch_topk=5, seq_beam_width=100, seq_hit_cutoffs=(10, 50, 100),
        )
        adapted["per_dir"][password_dir] = metrics_subset(metrics)
    combined_metrics, _ = evaluate_sequences(
        combined_tests, model, classes, means, stds, device,
        char_topk=(1, 3, 5), seq_branch_topk=5, seq_beam_width=100, seq_hit_cutoffs=(10, 50, 100),
    )
    adapted["combined"] = metrics_subset(combined_metrics)

    os.makedirs(os.path.dirname(args.output_checkpoint), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_timesteps": int(X_adapt.shape[1]),
        "n_channels": int(X_adapt.shape[2]),
        "n_classes": int(len(classes)),
        "classes": classes,
        "model_name": "InceptionTime",
        "adapted_from_multilen": True,
        "datasets": dataset_rows,
    }, args.output_checkpoint)
    np.savez(args.output_scaler, means=means, stds=stds)

    report = {
        "device": str(device),
        "base_checkpoint": args.base_checkpoint,
        "base_scaler": args.base_scaler,
        "output_checkpoint": args.output_checkpoint,
        "output_scaler": args.output_scaler,
        "datasets": dataset_rows,
        "zero_shot": zero_shot,
        "adapted": adapted,
    }
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nZERO-SHOT COMBINED")
    print(json.dumps(zero_shot["combined"], indent=2))
    print("\nADAPTED COMBINED")
    print(json.dumps(adapted["combined"], indent=2))
    print(f"Saved checkpoint -> {args.output_checkpoint}")
    print(f"Saved report -> {args.report_path}")


if __name__ == "__main__":
    main()
