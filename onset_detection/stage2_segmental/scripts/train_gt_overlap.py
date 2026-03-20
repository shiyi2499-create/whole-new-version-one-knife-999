#!/usr/bin/env python3
"""
Train GT-episode overlapping-window prototype (v2).

This replaces train_gt_segmental.py with the overlap model that gives each key
an independent window (can overlap neighbors) instead of non-overlapping partitions.

Usage (identical CLI to v1, same classifier path):

    python onset_detection/stage2_segmental/scripts/train_gt_overlap.py \
      --input_dir data/raw/mixed_training \
      --output_dir runs/stage2_overlap_gt \
      --classifier_checkpoint results/inception_password_final.pt \
      --classifier_scaler results/inception_password_scaler.npz \
      --device mps

Or without external classifier (will train a local one first):

    python onset_detection/stage2_segmental/scripts/train_gt_overlap.py \
      --input_dir data/raw/mixed_training \
      --output_dir runs/stage2_overlap_gt \
      --device mps
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
for p in (PROJECT_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_segmental.data import (
    ALL_CLASSES,
    build_password_episodes,
    compute_channel_stats,
    describe_episodes,
    extract_fixed_window,
    save_split_manifest,
    split_by_session,
    windows_from_episodes,
)
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.model import (
    build_classifier,
    load_external_inception,
    save_classifier_checkpoint,
    train_classifier,
)
from onset_detection.stage2_segmental.model_v2 import (
    OverlapConfig,
    OverlapWindowModel,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    req = device.lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"
    dev = torch.device(req)
    if dev.type == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    return dev


def _jitter_frames_np(
    key_frames: np.ndarray,
    num_frames: int,
    jitter_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply bounded anchor jitter while preserving temporal order."""
    if jitter_frames <= 0 or len(key_frames) == 0:
        return np.asarray(key_frames, dtype=np.int64)
    noise = rng.integers(-jitter_frames, jitter_frames + 1, size=len(key_frames))
    out = np.asarray(key_frames, dtype=np.int64) + noise
    out = np.clip(out, 0, max(num_frames - 1, 0))
    # Preserve nondecreasing order so labels still align to a valid sequence.
    out = np.maximum.accumulate(out)
    return out.astype(np.int64)


def _deterministic_jittered_frames(ep, jitter_ms: float) -> np.ndarray:
    if jitter_ms <= 0:
        return np.asarray(ep.key_frames, dtype=np.int64)
    jitter_frames = int(round(jitter_ms / 1000.0 * ep.sample_rate_hz))
    seed = zlib.crc32(ep.episode_id.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    return _jitter_frames_np(ep.key_frames, len(ep.imu), jitter_frames, rng)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_fixed_window(classifier, episodes, eval_anchor_jitter_ms: float = 0.0):
    """Evaluate fixed-window baseline (same as v1)."""
    results = []
    for ep in episodes:
        label_idx = np.asarray(
            [classifier.class_to_idx[c] for c in ep.chars if c in classifier.class_to_idx],
            dtype=np.int64,
        )
        if len(label_idx) != len(ep.chars):
            continue
        windows = []
        eval_frames = _deterministic_jittered_frames(ep, eval_anchor_jitter_ms)
        for frame in eval_frames.tolist():
            win = extract_fixed_window(ep, int(frame), target_len=classifier.target_len)
            if win is None:
                break
            windows.append(win)
        if len(windows) != len(ep.chars):
            continue
        with torch.no_grad():
            xb = torch.tensor(
                np.stack(windows), dtype=torch.float32,
                device=next(classifier.parameters()).device,
            )
            logits = classifier(xb).cpu().numpy()
        topk = char_topk_from_logits(logits, label_idx)
        pred = "".join(classifier.classes[int(i)] for i in logits.argmax(axis=1).tolist())
        results.append({
            "episode_id": ep.episode_id,
            "session_id": ep.session_id,
            "reference": ep.password,
            "prediction": pred,
            **topk,
        })
    return aggregate_episode_results(results), results


def evaluate_overlap(model, episodes, device, eval_anchor_jitter_ms: float = 0.0):
    """Evaluate the overlap model on episodes."""
    model.eval()
    results = []
    debug = []
    for ep in episodes:
        imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
        eval_frames = _deterministic_jittered_frames(ep, eval_anchor_jitter_ms)
        key_frames = torch.tensor(eval_frames, dtype=torch.long, device=device)
        labels = np.asarray(
            [model.classifier.class_to_idx[c] for c in ep.chars if c in model.classifier.class_to_idx],
            dtype=np.int64,
        )
        if len(labels) != len(ep.chars):
            continue
        with torch.no_grad():
            out = model.forward_episode(imu, key_frames, ep.sample_rate_hz)
        logits = out["logits"].detach().cpu().numpy()
        pred = "".join(model.classifier.classes[int(i)] for i in logits.argmax(axis=1).tolist())
        topk = char_topk_from_logits(logits, labels)
        results.append({
            "episode_id": ep.episode_id,
            "session_id": ep.session_id,
            "reference": ep.password,
            "prediction": pred,
            **topk,
        })
        debug.append({
            "episode_id": ep.episode_id,
            "starts": out["starts"].detach().cpu().tolist(),
            "ends": out["ends"].detach().cpu().tolist(),
            "offsets": out["offsets"].detach().cpu().tolist(),
            "widths": out["widths"].detach().cpu().tolist(),
            "width_scales": out["width_scales"].detach().cpu().tolist(),
            "reference": ep.password,
            "prediction": pred,
        })
    return aggregate_episode_results(results), results, debug


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_overlap(
    model: OverlapWindowModel,
    train_eps,
    val_eps,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: str,
    warmup_epochs: int = 5,
    patience: int = 25,
    train_anchor_jitter_ms: float = 0.0,
):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1),
    )

    best_metric = -1e18
    best_payload = None
    history = []
    no_improve = 0

    for epoch in range(epochs):
        random.shuffle(train_eps)
        model.train()
        train_losses = []
        train_metrics_agg = {}

        for ep in train_eps:
            labels = [
                model.classifier.class_to_idx[c]
                for c in ep.chars
                if c in model.classifier.class_to_idx
            ]
            if len(labels) != len(ep.chars):
                continue
            imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
            jitter_frames = int(round(train_anchor_jitter_ms / 1000.0 * ep.sample_rate_hz))
            if jitter_frames > 0:
                rng = np.random.default_rng((epoch + 1) * 1000003 + zlib.crc32(ep.episode_id.encode("utf-8")))
                train_frames = _jitter_frames_np(ep.key_frames, len(ep.imu), jitter_frames, rng)
            else:
                train_frames = np.asarray(ep.key_frames, dtype=np.int64)
            key_frames = torch.tensor(train_frames, dtype=torch.long, device=device)
            y = torch.tensor(labels, dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            out = model.forward_episode(imu, key_frames, ep.sample_rate_hz)
            loss, metrics = model.compute_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))
            for k, v in metrics.items():
                train_metrics_agg.setdefault(k, []).append(v)

        # LR scheduling: warmup then cosine
        if epoch >= warmup_epochs:
            scheduler.step()

        # Validation
        val_metrics, _, _ = evaluate_overlap(model, val_eps, device)
        score = float(val_metrics["char_top1"] - 0.25 * val_metrics["cer"])

        avg_train = {k: float(np.mean(v)) for k, v in train_metrics_agg.items()}
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "train_char_loss": avg_train.get("char_loss"),
            "train_mean_offset": avg_train.get("mean_offset_frames"),
            "train_mean_width_scale": avg_train.get("mean_width_scale"),
            "val_metrics": val_metrics,
            "selection_score": score,
        }
        history.append(epoch_row)

        tag = ""
        if score > best_metric:
            best_metric = score
            no_improve = 0
            best_payload = {
                **model.checkpoint_payload(),
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "history": history,
            }
            torch.save(best_payload, os.path.join(output_dir, "best_overlap.pt"))
            tag = " *"
        else:
            no_improve += 1

        print(
            f"[overlap] epoch={epoch+1:03d} "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"char_loss={avg_train.get('char_loss', 0):.4f} "
            f"offset={avg_train.get('mean_offset_frames', 0):.2f} "
            f"w_scale={avg_train.get('mean_width_scale', 0):.3f} "
            f"val_top1={val_metrics['char_top1']:.4f} "
            f"val_top5={val_metrics['char_top5']:.4f} "
            f"val_cer={val_metrics['cer']:.4f}"
            f"{tag}",
            flush=True,
        )

        if no_improve >= patience and epoch >= warmup_epochs + 10:
            print(
                f"[overlap] early stop at epoch {epoch+1} "
                f"(no improve for {patience})",
                flush=True,
            )
            break

    if best_payload is None:
        raise RuntimeError("No overlap checkpoint produced.")
    return best_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="GT episode overlap-window prototype (v2)",
    )
    ap.add_argument("--input_dir", required=True,
                    help="Directory with mixed_training sessions")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--holdout_sessions", default="",
                    help="Comma-separated session_ids for validation")
    ap.add_argument("--target_len", type=int, default=57)
    # Classifier
    ap.add_argument("--classifier_checkpoint", default="")
    ap.add_argument("--classifier_scaler", default="")
    ap.add_argument("--classifier_epochs", type=int, default=120)
    ap.add_argument("--classifier_lr", type=float, default=8e-4)
    # Overlap model
    ap.add_argument("--overlap_epochs", type=int, default=120)
    ap.add_argument("--overlap_lr", type=float, default=2e-4)
    ap.add_argument("--warmup_epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--encoder_hidden", type=int, default=96)
    ap.add_argument("--encoder_blocks", type=int, default=6)
    ap.add_argument("--max_offset_ms", type=float, default=60.0)
    ap.add_argument("--max_width_scale", type=float, default=2.0)
    ap.add_argument("--loss_offset", type=float, default=0.10)
    ap.add_argument("--loss_width", type=float, default=0.08)
    ap.add_argument("--loss_consistency", type=float, default=0.05)
    ap.add_argument("--unfreeze_classifier", action="store_true")
    ap.add_argument("--train_anchor_jitter_ms", type=float, default=0.0,
                    help="Uniform jitter magnitude applied to key anchors during training")
    ap.add_argument("--eval_anchor_jitter_ms", type=float, default=0.0,
                    help="Deterministic jitter magnitude applied to validation anchors")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load episodes
    # -----------------------------------------------------------------------
    episodes = build_password_episodes(args.input_dir)
    if not episodes:
        raise RuntimeError(f"No password episodes found under {args.input_dir}")
    print(
        f"[data] {len(episodes)} episodes from "
        f"{len({ep.session_id for ep in episodes})} sessions",
        flush=True,
    )

    holdouts = [s.strip() for s in args.holdout_sessions.split(",") if s.strip()]
    train_eps, val_eps, train_sessions, val_sessions = split_by_session(
        episodes,
        val_ratio=args.val_ratio,
        seed=args.seed,
        holdout_session_ids=holdouts or None,
    )
    print(
        f"[split] train={len(train_eps)} episodes ({len(train_sessions)} sessions), "
        f"val={len(val_eps)} episodes ({len(val_sessions)} sessions)",
        flush=True,
    )

    save_split_manifest(
        str(out_dir / "split_manifest.json"),
        train_sessions, val_sessions, episodes,
    )
    summary = {
        "all": describe_episodes(episodes),
        "train": describe_episodes(train_eps),
        "val": describe_episodes(val_eps),
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -----------------------------------------------------------------------
    # 2. Classifier (reuse or train local)
    # -----------------------------------------------------------------------
    if args.classifier_checkpoint and args.classifier_scaler:
        print("[info] loading external classifier checkpoint", flush=True)
        classifier = load_external_inception(
            args.classifier_checkpoint, args.classifier_scaler, device,
        )
        classifier_source = "external_checkpoint"
        classifier_stats = {"source": classifier_source}
    else:
        print("[info] training local fixed-window classifier", flush=True)
        classes = ALL_CLASSES.tolist()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        X_train, y_train = windows_from_episodes(
            train_eps, class_to_idx, target_len=args.target_len,
        )
        X_val, y_val = windows_from_episodes(
            val_eps, class_to_idx, target_len=args.target_len,
        )
        means, stds = compute_channel_stats(X_train)
        classifier = build_classifier(
            target_len=args.target_len, classes=classes, means=means, stds=stds,
        ).to(device)
        classifier, classifier_stats = train_classifier(
            classifier, X_train, y_train, X_val, y_val,
            device=device, epochs=args.classifier_epochs, lr=args.classifier_lr,
        )
        save_classifier_checkpoint(classifier, str(out_dir / "local_classifier.pt"))
        classifier_source = "trained_local_fixed_window"
        print(
            f"[classifier] best_val_acc={classifier_stats['best_val_acc']:.4f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # 3. Fixed-window baseline
    # -----------------------------------------------------------------------
    baseline_metrics, baseline_results = evaluate_fixed_window(
        classifier, val_eps, eval_anchor_jitter_ms=args.eval_anchor_jitter_ms,
    )
    print(
        f"[baseline] top1={baseline_metrics['char_top1']:.4f} "
        f"top3={baseline_metrics['char_top3']:.4f} "
        f"top5={baseline_metrics['char_top5']:.4f} "
        f"cer={baseline_metrics['cer']:.4f}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # 4. Build and train overlap model
    # -----------------------------------------------------------------------
    cfg = OverlapConfig(
        target_len=classifier.target_len,
        encoder_hidden=args.encoder_hidden,
        encoder_blocks=args.encoder_blocks,
        max_offset_ms=args.max_offset_ms,
        max_width_scale=args.max_width_scale,
        loss_offset=args.loss_offset,
        loss_width=args.loss_width,
        loss_consistency=args.loss_consistency,
    )
    model = OverlapWindowModel(cfg, classifier).to(device)
    model.freeze_classifier(not args.unfreeze_classifier)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"[model] trainable={trainable:,} / total={total_p:,} params", flush=True)

    # Sanity check: evaluate before any training (should match baseline closely)
    pre_metrics, _, _ = evaluate_overlap(
        model, val_eps, device, eval_anchor_jitter_ms=args.eval_anchor_jitter_ms,
    )
    print(
        f"[pre-train] top1={pre_metrics['char_top1']:.4f} "
        f"top5={pre_metrics['char_top5']:.4f} "
        f"cer={pre_metrics['cer']:.4f} "
        f"(should be close to baseline)",
        flush=True,
    )

    best_payload = train_overlap(
        model,
        train_eps,
        val_eps,
        device=device,
        epochs=args.overlap_epochs,
        lr=args.overlap_lr,
        output_dir=str(out_dir),
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        train_anchor_jitter_ms=args.train_anchor_jitter_ms,
    )

    # -----------------------------------------------------------------------
    # 5. Final evaluation with best checkpoint
    # -----------------------------------------------------------------------
    model.load_state_dict(best_payload["model_state"])
    ov_metrics, ov_results, ov_debug = evaluate_overlap(
        model, val_eps, device, eval_anchor_jitter_ms=args.eval_anchor_jitter_ms,
    )

    report = {
        "classifier_source": classifier_source,
        "classifier_stats": classifier_stats,
        "baseline_fixed_window": baseline_metrics,
        "pre_train_overlap": pre_metrics,
        "overlap": ov_metrics,
        "delta_top1": ov_metrics["char_top1"] - baseline_metrics["char_top1"],
        "delta_top3": ov_metrics["char_top3"] - baseline_metrics["char_top3"],
        "delta_top5": ov_metrics["char_top5"] - baseline_metrics["char_top5"],
        "delta_cer": ov_metrics["cer"] - baseline_metrics["cer"],
        "best_epoch": best_payload.get("epoch"),
        "config": cfg.__dict__,
        "train_anchor_jitter_ms": args.train_anchor_jitter_ms,
        "eval_anchor_jitter_ms": args.eval_anchor_jitter_ms,
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
    }

    print("\n" + "=" * 60, flush=True)
    print("FINAL COMPARISON", flush=True)
    print("=" * 60, flush=True)
    print(f"  Fixed-window baseline:", flush=True)
    print(
        f"    top1={baseline_metrics['char_top1']:.4f}  "
        f"top3={baseline_metrics['char_top3']:.4f}  "
        f"top5={baseline_metrics['char_top5']:.4f}  "
        f"cer={baseline_metrics['cer']:.4f}",
        flush=True,
    )
    print(
        f"  Overlap learned-window (best epoch {best_payload.get('epoch')}):",
        flush=True,
    )
    print(
        f"    top1={ov_metrics['char_top1']:.4f}  "
        f"top3={ov_metrics['char_top3']:.4f}  "
        f"top5={ov_metrics['char_top5']:.4f}  "
        f"cer={ov_metrics['cer']:.4f}",
        flush=True,
    )
    print(f"  Delta:", flush=True)
    print(
        f"    top1={report['delta_top1']:+.4f}  "
        f"top3={report['delta_top3']:+.4f}  "
        f"top5={report['delta_top5']:+.4f}  "
        f"cer={report['delta_cer']:+.4f}",
        flush=True,
    )
    print("=" * 60, flush=True)

    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "overlap_debug.json", "w", encoding="utf-8") as f:
        json.dump(ov_debug, f, ensure_ascii=False, indent=2)
    with open(out_dir / "overlap_results.json", "w", encoding="utf-8") as f:
        json.dump(ov_results, f, ensure_ascii=False, indent=2)
    with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
