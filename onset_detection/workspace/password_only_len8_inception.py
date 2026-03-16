#!/usr/bin/env python3
"""
Password-only InceptionTime experiment for len=8 strings.

This script intentionally does not use single_key + boost. It trains directly
on password windows from selected password parts and evaluates on held-out
password parts.
"""

from __future__ import annotations

import argparse
import json
import os
import inspect
from statistics import mean, pstdev

import numpy as np

from adapt_password_len8_inception import (
    flatten_items,
    parse_part,
    select_latest_complete_sessions,
    split_sequences_by_parts,
)
from phase3_password_inception.run_password_closure_inception import (
    SUPPORTED_RE,
    build_no_space_sequences,
    discover_freetype_sessions,
    evaluate_sequences,
    load_final_inception,
    resolve_torch_device,
    set_global_seed,
    train_final_inception,
)


FULL_CLASSES = np.array(sorted(set("abcdefghijklmnopqrstuvwxyz0123456789")))


def parse_args():
    p = argparse.ArgumentParser(description="Password-only repeated group-split experiment")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--password-dir", default="data/raw/password/len_8")
    p.add_argument("--report-path", default="results/password_only_len8_multisplit.json")
    p.add_argument("--checkpoint-dir", default="results/password_only_len8")
    p.add_argument("--epochs", type=int, default=280)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--num-splits", type=int, default=5)
    p.add_argument("--train-parts", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def aggregate_metrics(split_rows: list[dict]) -> dict:
    keys = [
        "char_top1_accuracy",
        "char_top3_accuracy",
        "char_top5_accuracy",
        "sequence_top100_hit_rate",
        "cer_top1",
    ]
    out = {}
    for key in keys:
        vals = [row[key] for row in split_rows]
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
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    class_to_idx = {c: i for i, c in enumerate(FULL_CLASSES.tolist())}

    while len(split_rows) < args.num_splits:
        test_parts = tuple(sorted(rng.sample(parts, len(parts) - args.train_parts)))
        if test_parts in seen:
            continue
        seen.add(test_parts)
        train_parts = tuple(p for p in parts if p not in test_parts)

        train_sequences, test_sequences = split_sequences_by_parts(
            sequences, set(train_parts), set(test_parts)
        )
        print(
            f"\nSplit {len(split_rows)+1}/{args.num_splits}: "
            f"train_parts={train_parts} test_parts={test_parts}"
        )
        print(f"  Train sequences: {len(train_sequences)} | Test sequences: {len(test_sequences)}")

        X_train, y_train = flatten_items(train_sequences, class_to_idx)
        ckpt_path = os.path.join(args.checkpoint_dir, f"split_{len(split_rows)+1}.pt")
        scaler_path = os.path.join(args.checkpoint_dir, f"split_{len(split_rows)+1}_scaler.npz")
        train_kwargs = dict(
            X=X_train,
            y_enc=y_train,
            classes=FULL_CLASSES,
            checkpoint_path=ckpt_path,
            scaler_path=scaler_path,
            device=device,
            force=True,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
        if "augment" in inspect.signature(train_final_inception).parameters:
            train_kwargs["augment"] = True
        train_final_inception(**train_kwargs)
        model, classes, means, stds = load_final_inception(ckpt_path, scaler_path, device)
        metrics, _ = evaluate_sequences(
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
            "train_parts": list(train_parts),
            "test_parts": list(test_parts),
            "char_top1_accuracy": metrics["char_top1_accuracy"],
            "char_top3_accuracy": metrics["char_top3_accuracy"],
            "char_top5_accuracy": metrics["char_top5_accuracy"],
            "sequence_top100_hit_rate": metrics["sequence_top100_hit_rate"],
            "cer_top1": metrics["cer_top1"],
            "unsupported_ref_char_rate": metrics["unsupported_ref_char_rate"],
        })

    report = {
        "device": str(device),
        "num_splits": args.num_splits,
        "train_parts": args.train_parts,
        "test_parts_per_split": len(parts) - args.train_parts,
        "splits": split_rows,
        "summary": aggregate_metrics(split_rows),
    }
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    s = report["summary"]
    print("\nPASSWORD-ONLY SUMMARY")
    print(f"  top1 mean/std:   {s['char_top1_accuracy']['mean']:.1%} / {s['char_top1_accuracy']['std']:.1%}")
    print(f"  top3 mean/std:   {s['char_top3_accuracy']['mean']:.1%} / {s['char_top3_accuracy']['std']:.1%}")
    print(f"  top5 mean/std:   {s['char_top5_accuracy']['mean']:.1%} / {s['char_top5_accuracy']['std']:.1%}")
    print(f"  seq@100 mean/std:{s['sequence_top100_hit_rate']['mean']:.1%} / {s['sequence_top100_hit_rate']['std']:.1%}")
    print(f"  CER mean/std:    {s['cer_top1']['mean']:.1%} / {s['cer_top1']['std']:.1%}")
    print(f"Saved -> {args.report_path}")


if __name__ == "__main__":
    main()
