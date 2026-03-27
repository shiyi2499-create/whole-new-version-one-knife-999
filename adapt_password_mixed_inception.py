#!/usr/bin/env python3
"""
Adapt the Stage3 Inception classifier with mixed-scene password windows.

This script keeps the existing standalone password adaptation path, but adds
mixed-session password episodes extracted from onset_detection/stage2_segmental.
It is intended to reduce the mixed-scene distribution shift observed in the
fair 6-session / 10-episode closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE3_DIR = os.path.join(ROOT, "phase3_password_inception")
if PHASE3_DIR not in sys.path:
    sys.path.insert(0, PHASE3_DIR)

from adapt_password_multilen_inception import (  # noqa: E402
    fine_tune_on_password,
    flatten_items,
    load_sequences,
    normalize_windows,
)
from onset_detection.stage2_segmental.data import (  # noqa: E402
    PasswordEpisode,
    build_password_episodes,
    extract_fixed_window,
)
from phase3_password_inception.run_password_closure_inception import (  # noqa: E402
    WindowConfig,
    build_no_space_sequences,
    discover_freetype_sessions,
    evaluate_sequences,
    load_final_inception,
    load_inception_norm_mode,
    resolve_torch_device,
    set_global_seed,
)


def _episode_to_sequence(
    ep: PasswordEpisode,
    class_to_idx: dict[str, int],
    target_len: int,
    pre_ms: float,
    post_ms: float,
) -> dict | None:
    items = []
    for frame, ch in zip(ep.key_frames.tolist(), ep.chars):
        if ch not in class_to_idx:
            continue
        win = extract_fixed_window(ep, int(frame), pre_ms=pre_ms, post_ms=post_ms, target_len=target_len)
        if win is None:
            return None
        items.append({"key": ch, "timestamp_ns": None, "window": win.astype(np.float32)})
    if not items:
        return None
    return {
        "session": ep.session_id,
        "sequence_idx": ep.episode_index,
        "reference": ep.password,
        "items": items,
    }


def _load_mixed_sequences(
    mixed_dir: str,
    class_to_idx: dict[str, int],
    target_len: int,
    holdout_sessions: set[str],
    pre_ms: float,
    post_ms: float,
) -> tuple[list[dict], list[dict], dict]:
    episodes = build_password_episodes(mixed_dir)
    train_seqs: list[dict] = []
    holdout_seqs: list[dict] = []
    for ep in episodes:
        seq = _episode_to_sequence(ep, class_to_idx, target_len, pre_ms, post_ms)
        if seq is None:
            continue
        if ep.session_id in holdout_sessions:
            holdout_seqs.append(seq)
        else:
            train_seqs.append(seq)
    info = {
        "mixed_dir": mixed_dir,
        "episodes_total": len(episodes),
        "train_sequences": len(train_seqs),
        "holdout_sequences": len(holdout_seqs),
    }
    return train_seqs, holdout_seqs, info


def _load_password_sequences_with_window(
    password_dir: str,
    pre_ms: float,
    post_ms: float,
) -> tuple[list[dict], dict]:
    sessions = discover_freetype_sessions([password_dir])
    if not sessions:
        raise RuntimeError(f"No password sessions found in {password_dir}")
    window_cfg = WindowConfig(
        pre_trigger_ms=int(round(pre_ms)),
        post_trigger_ms=int(round(post_ms)),
        min_window_samples=2,
    )
    seqs: list[dict] = []
    for sess in sessions:
        seqs.extend(
            build_no_space_sequences(
                sess,
                yes_only=True,
                eval_max_sequences=0,
                window_cfg=window_cfg,
            )
        )
    info = {
        "password_dir": password_dir,
        "sessions": len(sessions),
        "sequences": len(seqs),
        "pre_ms": float(pre_ms),
        "post_ms": float(post_ms),
    }
    return seqs, info


def _metrics_subset(metrics: dict) -> dict:
    keep = [
        "total_sequences",
        "exact_match_rate_top1",
        "char_top1_accuracy",
        "char_top3_accuracy",
        "char_top5_accuracy",
        "sequence_top100_hit_rate",
        "cer_top1",
    ]
    return {k: metrics[k] for k in keep if k in metrics}


def _expand_hard_chars(groups: list[str], class_to_idx: dict[str, int]) -> list[str]:
    chars: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for ch in str(group):
            if ch in class_to_idx and ch not in seen:
                seen.add(ch)
                chars.append(ch)
    return chars


def _oversample_target_windows(
    X: np.ndarray,
    y: np.ndarray,
    target_indices: set[int],
    factor: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    info = {
        "enabled": bool(target_indices and factor > 1 and len(X) > 0),
        "factor": int(factor),
        "num_target_before": 0,
        "num_target_after": 0,
        "num_total_before": int(len(X)),
        "num_total_after": int(len(X)),
    }
    if not info["enabled"]:
        return X, y, info

    mask = np.asarray([int(lbl) in target_indices for lbl in y.tolist()], dtype=bool)
    target_count = int(mask.sum())
    info["num_target_before"] = target_count
    if target_count == 0:
        return X, y, info

    X_dup = np.repeat(X[mask], factor - 1, axis=0)
    y_dup = np.repeat(y[mask], factor - 1, axis=0)
    X_new = np.concatenate([X, X_dup], axis=0)
    y_new = np.concatenate([y, y_dup], axis=0)
    info["num_target_after"] = int(target_count * factor)
    info["num_total_after"] = int(len(X_new))
    return X_new, y_new, info


def parse_args():
    ap = argparse.ArgumentParser(description="Stage3 mixed-scene adaptation")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    ap.add_argument("--base-checkpoint", default="results/inception_password_len8_len9_len10_quick.pt")
    ap.add_argument("--base-scaler", default="results/inception_password_len8_len9_len10_quick_scaler.npz")
    ap.add_argument("--output-checkpoint", required=True)
    ap.add_argument("--output-scaler", required=True)
    ap.add_argument("--report-path", required=True)
    ap.add_argument("--password-dir", action="append", default=[])
    ap.add_argument("--mixed-dir", action="append", default=[])
    ap.add_argument("--holdout-session-id", action="append", default=[])
    ap.add_argument("--pre-ms", type=float, default=100.0)
    ap.add_argument("--post-ms", type=float, default=200.0)
    ap.add_argument("--head-epochs", type=int, default=12)
    ap.add_argument("--full-epochs", type=int, default=30)
    ap.add_argument("--head-lr", type=float, default=3e-4)
    ap.add_argument("--full-lr", type=float, default=1.5e-4)
    ap.add_argument("--adapt-batch-size", type=int, default=32)
    ap.add_argument("--hard-char-group", action="append", default=[])
    ap.add_argument("--hard-oversample-factor", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = resolve_torch_device(args.device)
    holdout_sessions = {str(x) for x in args.holdout_session_id}

    model, classes, means, stds = load_final_inception(args.base_checkpoint, args.base_scaler, device)
    model.eval()
    norm_mode = load_inception_norm_mode(args.base_checkpoint)
    target_len = int((args.pre_ms + args.post_ms) / 1000.0 * 190.0)
    class_to_idx = {str(c): i for i, c in enumerate(classes.tolist())}

    standalone_train = []
    standalone_info = []
    for password_dir in args.password_dir:
        seqs, info = _load_password_sequences_with_window(password_dir, args.pre_ms, args.post_ms)
        standalone_train.extend(seqs)
        standalone_info.append(info)

    mixed_train = []
    mixed_holdout = []
    mixed_info = []
    for mixed_dir in args.mixed_dir:
        train_seqs, holdout_seqs, info = _load_mixed_sequences(
            mixed_dir,
            class_to_idx,
            target_len,
            holdout_sessions,
            args.pre_ms,
            args.post_ms,
        )
        mixed_train.extend(train_seqs)
        mixed_holdout.extend(holdout_seqs)
        mixed_info.append(info)

    if not standalone_train and not mixed_train:
        raise RuntimeError("No adaptation sequences found.")
    if not mixed_holdout:
        raise RuntimeError("No mixed holdout sequences found. Check --holdout-session-id / --mixed-dir.")

    zero_shot_metrics, _ = evaluate_sequences(
        mixed_holdout,
        model,
        classes,
        means,
        stds,
        device,
        norm_mode=norm_mode,
    )

    standalone_chars_before = {}
    mixed_chars_before = {}
    standalone_chars_after = {}
    mixed_chars_after = {}

    X_parts = []
    y_parts = []

    if standalone_train:
        X_standalone, y_standalone = flatten_items(standalone_train, class_to_idx)
        standalone_chars_before = {str(classes[int(k)]): int(v) for k, v in Counter(y_standalone.tolist()).items()}
        X_standalone = normalize_windows(X_standalone, means, stds, norm_mode=norm_mode)
        standalone_chars_after = dict(standalone_chars_before)
        X_parts.append(X_standalone)
        y_parts.append(y_standalone)

    hard_chars = _expand_hard_chars(args.hard_char_group, class_to_idx)
    hard_target_indices = {int(class_to_idx[ch]) for ch in hard_chars}
    hard_oversample_info = {
        "enabled": False,
        "factor": int(args.hard_oversample_factor),
        "target_chars": hard_chars,
        "target_indices": sorted(int(x) for x in hard_target_indices),
    }

    if mixed_train:
        X_mixed, y_mixed = flatten_items(mixed_train, class_to_idx)
        mixed_chars_before = {str(classes[int(k)]): int(v) for k, v in Counter(y_mixed.tolist()).items()}
        X_mixed = normalize_windows(X_mixed, means, stds, norm_mode=norm_mode)
        X_mixed, y_mixed, over_info = _oversample_target_windows(
            X_mixed,
            y_mixed,
            hard_target_indices,
            max(int(args.hard_oversample_factor), 1),
        )
        mixed_chars_after = {str(classes[int(k)]): int(v) for k, v in Counter(y_mixed.tolist()).items()}
        hard_oversample_info.update(over_info)
        X_parts.append(X_mixed)
        y_parts.append(y_mixed)

    X_train = np.concatenate(X_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)

    model = fine_tune_on_password(
        model,
        X_train,
        y_train,
        device,
        batch_size=args.adapt_batch_size,
        head_epochs=args.head_epochs,
        full_epochs=args.full_epochs,
        head_lr=args.head_lr,
        full_lr=args.full_lr,
    )
    model.eval()

    adapted_metrics, _ = evaluate_sequences(
        mixed_holdout,
        model,
        classes,
        means,
        stds,
        device,
        norm_mode=norm_mode,
    )

    output_ckpt = Path(args.output_checkpoint)
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    ckpt["model_state"] = copy.deepcopy(model.state_dict())
    ckpt["norm_mode"] = norm_mode
    ckpt["pre_ms"] = float(args.pre_ms)
    ckpt["post_ms"] = float(args.post_ms)
    ckpt["n_timesteps"] = int(target_len)
    ckpt["target_rate_hz"] = 190.0
    torch.save(ckpt, output_ckpt)

    output_scaler = Path(args.output_scaler)
    output_scaler.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_scaler, means=means, stds=stds, classes=classes)

    report = {
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "base_scaler": os.path.abspath(args.base_scaler),
        "output_checkpoint": str(output_ckpt.resolve()),
        "output_scaler": str(output_scaler.resolve()),
        "device": str(device),
        "target_len": int(target_len),
        "norm_mode": norm_mode,
        "pre_ms": float(args.pre_ms),
        "post_ms": float(args.post_ms),
        "holdout_sessions": sorted(holdout_sessions),
        "standalone_info": standalone_info,
        "mixed_info": mixed_info,
        "num_standalone_train_sequences": len(standalone_train),
        "num_mixed_train_sequences": len(mixed_train),
        "num_mixed_holdout_sequences": len(mixed_holdout),
        "num_train_items": int(len(X_train)),
        "hard_oversample": hard_oversample_info,
        "standalone_char_counts_before": standalone_chars_before,
        "standalone_char_counts_after": standalone_chars_after,
        "mixed_char_counts_before": mixed_chars_before,
        "mixed_char_counts_after": mixed_chars_after,
        "zero_shot_mixed_holdout": _metrics_subset(zero_shot_metrics),
        "adapted_mixed_holdout": _metrics_subset(adapted_metrics),
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
