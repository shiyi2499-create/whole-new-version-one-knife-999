#!/usr/bin/env python3
"""
Repeated group-split evaluation for the len=8 password route.

This script keeps InceptionTime fixed and evaluates:
1. zero-shot transfer from single_key + boost
2. password-style adaptation from the same baseline

It uses group-level splits over password parts, rather than a single fixed
1-16 / 17-20 split, so we can estimate result stability.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from statistics import mean, pstdev

from adapt_password_len8_inception import (
    fine_tune_on_password,
    flatten_items,
    normalize_windows,
    parse_part,
    select_latest_complete_sessions,
    split_sequences_by_parts,
)
from phase3_password_inception.run_password_closure_inception import (
    build_no_space_sequences,
    discover_freetype_sessions,
    evaluate_sequences,
    load_final_inception,
    load_merged_training_data,
    resolve_torch_device,
    set_global_seed,
    train_final_inception,
)


def parse_args():
    p = argparse.ArgumentParser(description="Repeated group-split password adaptation evaluation")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--merged-path", default="data/processed/merged_dataset.npz")
    p.add_argument("--password-dir", default="data/raw/password/len_8")
    p.add_argument("--checkpoint-path", default="results/inception_password_final.pt")
    p.add_argument("--scaler-path", default="results/inception_password_scaler.npz")
    p.add_argument("--report-path", default="results/password_len8_multisplit.json")
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
    p.add_argument("--num-splits", type=int, default=5)
    p.add_argument("--train-parts", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def aggregate_metrics(split_rows: list[dict], prefix: str) -> dict:
    keys = [
        "char_top1_accuracy",
        "char_top3_accuracy",
        "char_top5_accuracy",
        "sequence_top100_hit_rate",
        "cer_top1",
    ]
    out = {}
    for key in keys:
        vals = [row[f"{prefix}_{key}"] for row in split_rows]
        out[key] = {
            "mean": mean(vals),
            "std": pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    return out


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_torch_device(args.device)
    print(f"Device: {device}")

    X, _, y_enc, classes = load_merged_training_data(args.merged_path)
    print(f"Loaded baseline training data: X={X.shape}, classes={len(classes)}")
    train_final_inception(
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
        augment=True,
    )

    base_model, classes, means, stds = load_final_inception(
        args.checkpoint_path, args.scaler_path, device
    )
    base_state = copy.deepcopy(base_model.state_dict())
    class_to_idx = {c: i for i, c in enumerate(classes.tolist())}

    sessions = select_latest_complete_sessions(discover_freetype_sessions([args.password_dir]))
    if not sessions:
        raise RuntimeError("No password sessions found.")
    print(f"Found {len(sessions)} selected password sessions")

    sequences = []
    parts = set()
    for sess in sessions:
        part = parse_part(sess)
        seqs = build_no_space_sequences(sess, yes_only=True, eval_max_sequences=0)
        print(f"  part {part}: {len(seqs)} sequences")
        sequences.extend(seqs)
        parts.add(part)

    parts = sorted(parts)
    if len(parts) < args.train_parts + 1:
        raise RuntimeError("Not enough password parts for the requested split.")

    import random
    rng = random.Random(args.seed)
    seen = set()
    split_rows = []

    while len(split_rows) < args.num_splits:
        test_parts = tuple(sorted(rng.sample(parts, len(parts) - args.train_parts)))
        if test_parts in seen:
            continue
        seen.add(test_parts)
        adapt_parts = tuple(p for p in parts if p not in test_parts)

        adapt_sequences, test_sequences = split_sequences_by_parts(
            sequences, set(adapt_parts), set(test_parts)
        )
        print(
            f"\nSplit {len(split_rows)+1}/{args.num_splits}: "
            f"adapt_parts={adapt_parts} test_parts={test_parts}"
        )
        print(f"  Adapt sequences: {len(adapt_sequences)} | Test sequences: {len(test_sequences)}")

        zero_shot_metrics, _ = evaluate_sequences(
            test_sequences,
            base_model,
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
        model = copy.deepcopy(base_model)
        model.load_state_dict(base_state)
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
        adapted_metrics, _ = evaluate_sequences(
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

        split_rows.append({
            "adapt_parts": list(adapt_parts),
            "test_parts": list(test_parts),
            "zero_shot_char_top1_accuracy": zero_shot_metrics["char_top1_accuracy"],
            "zero_shot_char_top3_accuracy": zero_shot_metrics["char_top3_accuracy"],
            "zero_shot_char_top5_accuracy": zero_shot_metrics["char_top5_accuracy"],
            "zero_shot_sequence_top100_hit_rate": zero_shot_metrics["sequence_top100_hit_rate"],
            "zero_shot_cer_top1": zero_shot_metrics["cer_top1"],
            "adapted_char_top1_accuracy": adapted_metrics["char_top1_accuracy"],
            "adapted_char_top3_accuracy": adapted_metrics["char_top3_accuracy"],
            "adapted_char_top5_accuracy": adapted_metrics["char_top5_accuracy"],
            "adapted_sequence_top100_hit_rate": adapted_metrics["sequence_top100_hit_rate"],
            "adapted_cer_top1": adapted_metrics["cer_top1"],
        })

    report = {
        "device": str(device),
        "num_splits": args.num_splits,
        "train_parts": args.train_parts,
        "test_parts_per_split": len(parts) - args.train_parts,
        "splits": split_rows,
        "zero_shot_summary": aggregate_metrics(split_rows, "zero_shot"),
        "adapted_summary": aggregate_metrics(split_rows, "adapted"),
    }
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nADAPTED SUMMARY")
    s = report["adapted_summary"]
    print(f"  top1 mean/std:   {s['char_top1_accuracy']['mean']:.1%} / {s['char_top1_accuracy']['std']:.1%}")
    print(f"  top3 mean/std:   {s['char_top3_accuracy']['mean']:.1%} / {s['char_top3_accuracy']['std']:.1%}")
    print(f"  top5 mean/std:   {s['char_top5_accuracy']['mean']:.1%} / {s['char_top5_accuracy']['std']:.1%}")
    print(f"  seq@100 mean/std:{s['sequence_top100_hit_rate']['mean']:.1%} / {s['sequence_top100_hit_rate']['std']:.1%}")
    print(f"  CER mean/std:    {s['cer_top1']['mean']:.1%} / {s['cer_top1']['std']:.1%}")
    print(f"Saved -> {args.report_path}")


if __name__ == "__main__":
    main()
