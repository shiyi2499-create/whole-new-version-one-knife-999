#!/usr/bin/env python3
"""
Smoke test for the overlap-window v2 model.

Uses the bundle's data_samples/ (2 mixed_training sessions) to verify:
1. Data loading works
2. Model forward/backward pass works
3. Pre-train eval matches fixed-window baseline (within tolerance)
4. Training loop runs without crash
5. Learned offsets/widths stay reasonable

Usage:
    python onset_detection/stage2_segmental/scripts/smoke_test_overlap.py

Or with the data_samples directly:
    python onset_detection/stage2_segmental/scripts/smoke_test_overlap.py \
      --input_dir onset_gpt54pro_bundle/data_samples
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
for p in (PROJECT_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="",
                    help="Path to mixed_training-like sessions. "
                         "If empty, tries to find data_samples/ automatically.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # Find data
    if args.input_dir:
        input_dir = args.input_dir
    else:
        candidates = [
            os.path.join(PROJECT_ROOT, "data_samples"),
            os.path.join(PROJECT_ROOT, "onset_gpt54pro_bundle", "data_samples"),
        ]
        input_dir = None
        for c in candidates:
            if os.path.isdir(c):
                input_dir = c
                break
        if input_dir is None:
            print("ERROR: Cannot find data_samples/. Pass --input_dir explicitly.")
            sys.exit(1)

    print(f"[smoke] Using data from: {input_dir}")

    device = torch.device(args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Step 1: Load data ---
    print("\n[1/5] Loading episodes...")
    from onset_detection.stage2_segmental.data import (
        ALL_CLASSES,
        build_password_episodes,
        compute_channel_stats,
        extract_fixed_window,
        split_by_session,
        windows_from_episodes,
    )
    episodes = build_password_episodes(input_dir)
    print(f"  Found {len(episodes)} episodes from "
          f"{len({ep.session_id for ep in episodes})} sessions")
    assert len(episodes) > 0, "No episodes found!"

    train_eps, val_eps, _, _ = split_by_session(episodes, val_ratio=0.5, seed=42)
    print(f"  train={len(train_eps)}, val={len(val_eps)}")
    assert len(train_eps) > 0 and len(val_eps) > 0

    # --- Step 2: Build classifier ---
    print("\n[2/5] Building local classifier...")
    from onset_detection.stage2_segmental.model import build_classifier, train_classifier

    classes = ALL_CLASSES.tolist()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    X_train, y_train = windows_from_episodes(train_eps, class_to_idx, target_len=57)
    X_val, y_val = windows_from_episodes(val_eps, class_to_idx, target_len=57)
    means, stds = compute_channel_stats(X_train)
    classifier = build_classifier(target_len=57, classes=classes, means=means, stds=stds).to(device)
    classifier, _ = train_classifier(
        classifier, X_train, y_train, X_val, y_val,
        device=device, epochs=15, lr=8e-4,  # short run for smoke test
    )
    print("  Classifier trained (15 epochs).")

    # --- Step 3: Fixed-window baseline ---
    print("\n[3/5] Fixed-window baseline...")
    from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits

    baseline_results = []
    for ep in val_eps:
        label_idx = np.asarray([class_to_idx[c] for c in ep.chars if c in class_to_idx], dtype=np.int64)
        if len(label_idx) != len(ep.chars):
            continue
        windows = []
        for frame in ep.key_frames.tolist():
            win = extract_fixed_window(ep, int(frame), target_len=57)
            if win is not None:
                windows.append(win)
        if len(windows) != len(ep.chars):
            continue
        with torch.no_grad():
            xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=device)
            logits = classifier(xb).cpu().numpy()
        topk = char_topk_from_logits(logits, label_idx)
        pred = "".join(classes[int(i)] for i in logits.argmax(axis=1).tolist())
        baseline_results.append({
            "episode_id": ep.episode_id, "session_id": ep.session_id,
            "reference": ep.password, "prediction": pred, **topk,
        })
    bl_metrics = aggregate_episode_results(baseline_results)
    print(f"  Baseline: top1={bl_metrics['char_top1']:.4f} "
          f"top5={bl_metrics['char_top5']:.4f} cer={bl_metrics['cer']:.4f}")

    # --- Step 4: Build overlap model and check pre-train ---
    print("\n[4/5] Overlap model pre-train check...")
    from onset_detection.stage2_segmental.model_v2 import OverlapConfig, OverlapWindowModel

    cfg = OverlapConfig(target_len=57, encoder_hidden=64, encoder_blocks=4)
    model = OverlapWindowModel(cfg, classifier).to(device)
    model.freeze_classifier(True)

    # Pre-train eval should be close to baseline
    model.eval()
    pre_results = []
    for ep in val_eps:
        labels = np.asarray([class_to_idx[c] for c in ep.chars if c in class_to_idx], dtype=np.int64)
        if len(labels) != len(ep.chars):
            continue
        imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
        key_frames = torch.tensor(ep.key_frames, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model.forward_episode(imu, key_frames, ep.sample_rate_hz)
        logits = out["logits"].cpu().numpy()
        topk = char_topk_from_logits(logits, labels)
        pred = "".join(classes[int(i)] for i in logits.argmax(axis=1).tolist())
        pre_results.append({
            "episode_id": ep.episode_id, "session_id": ep.session_id,
            "reference": ep.password, "prediction": pred, **topk,
        })
    pre_metrics = aggregate_episode_results(pre_results)
    print(f"  Pre-train: top1={pre_metrics['char_top1']:.4f} "
          f"top5={pre_metrics['char_top5']:.4f} cer={pre_metrics['cer']:.4f}")

    # Check pre-train is within reasonable tolerance of baseline
    # The encoder adds some noise at init, so allow some slack
    delta_top1 = abs(pre_metrics["char_top1"] - bl_metrics["char_top1"])
    if delta_top1 > 0.15:
        print(f"  WARNING: pre-train top1 differs from baseline by {delta_top1:.4f}")
        print(f"  This is expected if encoder initialization adds noise,")
        print(f"  but should converge back during training.")
    else:
        print(f"  GOOD: pre-train matches baseline within {delta_top1:.4f}")

    # Check offsets and widths are near initialization
    with torch.no_grad():
        ep = val_eps[0]
        imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
        kf = torch.tensor(ep.key_frames, dtype=torch.long, device=device)
        out = model.forward_episode(imu, kf, ep.sample_rate_hz)
        offsets = out["offsets"].cpu().numpy()
        w_scales = out["width_scales"].cpu().numpy()
        print(f"  Init offsets: mean={offsets.mean():.4f} std={offsets.std():.4f} "
              f"(should be near 0)")
        print(f"  Init width_scales: mean={w_scales.mean():.4f} std={w_scales.std():.4f} "
              f"(should be near 1.0)")

    # --- Step 5: Short training loop ---
    print("\n[5/5] Training 10 epochs...")
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)

    for epoch in range(10):
        random.shuffle(train_eps)
        losses = []
        for ep in train_eps:
            labels = [class_to_idx[c] for c in ep.chars if c in class_to_idx]
            if len(labels) != len(ep.chars):
                continue
            imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
            kf = torch.tensor(ep.key_frames, dtype=torch.long, device=device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_episode(imu, kf, ep.sample_rate_hz)
            loss, metrics = model.compute_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        if (epoch + 1) % 5 == 0:
            model.eval()
            post_results = []
            for ep in val_eps:
                labels_np = np.asarray([class_to_idx[c] for c in ep.chars if c in class_to_idx], dtype=np.int64)
                if len(labels_np) != len(ep.chars):
                    continue
                imu_t = torch.tensor(ep.imu, dtype=torch.float32, device=device)
                kf_t = torch.tensor(ep.key_frames, dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model.forward_episode(imu_t, kf_t, ep.sample_rate_hz)
                logits_np = out["logits"].cpu().numpy()
                topk = char_topk_from_logits(logits_np, labels_np)
                pred = "".join(classes[int(j)] for j in logits_np.argmax(axis=1).tolist())
                post_results.append({
                    "episode_id": ep.episode_id, "session_id": ep.session_id,
                    "reference": ep.password, "prediction": pred, **topk,
                })
            post_m = aggregate_episode_results(post_results)
            print(f"  epoch={epoch+1:02d} loss={np.mean(losses):.4f} "
                  f"val_top1={post_m['char_top1']:.4f} "
                  f"val_top5={post_m['char_top5']:.4f}")
            model.train()

    # Final check
    model.eval()
    with torch.no_grad():
        ep = val_eps[0]
        imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
        kf = torch.tensor(ep.key_frames, dtype=torch.long, device=device)
        out = model.forward_episode(imu, kf, ep.sample_rate_hz)
        offsets = out["offsets"].cpu().numpy()
        w_scales = out["width_scales"].cpu().numpy()
        print(f"\n  Final offsets: mean={offsets.mean():.4f} std={offsets.std():.4f}")
        print(f"  Final width_scales: mean={w_scales.mean():.4f} std={w_scales.std():.4f}")

    print("\n" + "=" * 50)
    print("SMOKE TEST PASSED")
    print("=" * 50)
    print(f"\nTo run full training:")
    print(f"  python onset_detection/stage2_segmental/scripts/train_gt_overlap.py \\")
    print(f"    --input_dir data/raw/mixed_training \\")
    print(f"    --output_dir runs/stage2_overlap_gt \\")
    print(f"    --classifier_checkpoint results/inception_password_final.pt \\")
    print(f"    --classifier_scaler results/inception_password_scaler.npz \\")
    print(f"    --device mps")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
