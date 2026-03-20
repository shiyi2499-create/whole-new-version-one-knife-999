#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
    save_split_manifest,
    split_by_session,
    windows_from_episodes,
)
from onset_detection.stage2_segmental.metrics import aggregate_episode_results, char_topk_from_logits
from onset_detection.stage2_segmental.model import (
    SegmentalConfig,
    SegmentalSequenceModel,
    build_classifier,
    load_external_inception,
    save_classifier_checkpoint,
    train_classifier,
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


def evaluate_fixed_window(classifier, episodes):
    results = []
    for ep in episodes:
        label_idx = np.asarray([classifier.class_to_idx[c] for c in ep.chars if c in classifier.class_to_idx], dtype=np.int64)
        if len(label_idx) != len(ep.chars):
            continue
        windows = []
        for frame in ep.key_frames.tolist():
            from onset_detection.stage2_segmental.data import extract_fixed_window
            win = extract_fixed_window(ep, int(frame), target_len=classifier.target_len)
            if win is None:
                break
            windows.append(win)
        if len(windows) != len(ep.chars):
            continue
        with torch.no_grad():
            xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=next(classifier.parameters()).device)
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


def evaluate_segmental(model, episodes, device):
    model.eval()
    results = []
    debug = []
    for ep in episodes:
        imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
        key_frames = torch.tensor(ep.key_frames, dtype=torch.long, device=device)
        labels = np.asarray([model.classifier.class_to_idx[c] for c in ep.chars if c in model.classifier.class_to_idx], dtype=np.int64)
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
            "boundaries": out["boundaries"].detach().cpu().tolist(),
            "durations": out["durations"].detach().cpu().tolist(),
            "rel_pos": out["rel_pos"].detach().cpu().tolist(),
            "reference": ep.password,
            "prediction": pred,
        })
    return aggregate_episode_results(results), results, debug


def train_segmental(model, train_eps, val_eps, device, epochs: int, lr: float, output_dir: str):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    best_metric = -1e18
    best_payload = None
    history = []

    for epoch in range(epochs):
        random.shuffle(train_eps)
        model.train()
        train_losses = []
        for ep in train_eps:
            labels = [model.classifier.class_to_idx[c] for c in ep.chars if c in model.classifier.class_to_idx]
            if len(labels) != len(ep.chars):
                continue
            imu = torch.tensor(ep.imu, dtype=torch.float32, device=device)
            key_frames = torch.tensor(ep.key_frames, dtype=torch.long, device=device)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_episode(imu, key_frames, ep.sample_rate_hz)
            loss, _ = model.compute_loss(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics, _, _ = evaluate_segmental(model, val_eps, device)
        score = float(val_metrics["char_top1"] - 0.25 * val_metrics["cer"])
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(train_losses)) if train_losses else None,
            "val_metrics": val_metrics,
            "selection_score": score,
        }
        history.append(epoch_row)
        print(f"[segmental] epoch={epoch+1:03d} train_loss={epoch_row['train_loss']:.4f} val_top1={val_metrics['char_top1']:.4f} val_top5={val_metrics['char_top5']:.4f} val_cer={val_metrics['cer']:.4f}", flush=True)
        if score > best_metric:
            best_metric = score
            best_payload = {
                **model.checkpoint_payload(),
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "history": history,
            }
            torch.save(best_payload, os.path.join(output_dir, "best_segmental.pt"))

    if best_payload is None:
        raise RuntimeError("No segmental checkpoint produced.")
    return best_payload


def parse_args():
    ap = argparse.ArgumentParser(description="GT episode monotonic segmental prototype")
    ap.add_argument("--input_dir", required=True, help="Directory containing mixed_training-like sessions")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--holdout_sessions", default="", help="Comma-separated session_ids for validation")
    ap.add_argument("--target_len", type=int, default=57)
    ap.add_argument("--classifier_checkpoint", default="")
    ap.add_argument("--classifier_scaler", default="")
    ap.add_argument("--classifier_epochs", type=int, default=120)
    ap.add_argument("--segmental_epochs", type=int, default=80)
    ap.add_argument("--classifier_lr", type=float, default=8e-4)
    ap.add_argument("--segmental_lr", type=float, default=3e-4)
    ap.add_argument("--encoder_hidden", type=int, default=96)
    ap.add_argument("--encoder_blocks", type=int, default=6)
    ap.add_argument("--unfreeze_classifier", action="store_true", help="Fine-tune classifier together with the cutter instead of freezing it.")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = build_password_episodes(args.input_dir)
    if not episodes:
        raise RuntimeError(f"No password episodes found under {args.input_dir}")

    holdouts = [s.strip() for s in args.holdout_sessions.split(",") if s.strip()]
    train_eps, val_eps, train_sessions, val_sessions = split_by_session(
        episodes,
        val_ratio=args.val_ratio,
        seed=args.seed,
        holdout_session_ids=holdouts or None,
    )

    save_split_manifest(str(out_dir / "split_manifest.json"), train_sessions, val_sessions, episodes)
    summary = {
        "all": describe_episodes(episodes),
        "train": describe_episodes(train_eps),
        "val": describe_episodes(val_eps),
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
    }
    with open(out_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.classifier_checkpoint and args.classifier_scaler:
        print("[info] loading external classifier checkpoint", flush=True)
        classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
        classifier_source = "external_checkpoint"
        classifier_stats = {"source": classifier_source}
    else:
        print("[info] training local fixed-window classifier", flush=True)
        classes = ALL_CLASSES.tolist()
        class_to_idx = {c: i for i, c in enumerate(classes)}
        X_train, y_train = windows_from_episodes(train_eps, class_to_idx, target_len=args.target_len)
        X_val, y_val = windows_from_episodes(val_eps, class_to_idx, target_len=args.target_len)
        means, stds = compute_channel_stats(X_train)
        classifier = build_classifier(target_len=args.target_len, classes=classes, means=means, stds=stds).to(device)
        classifier, classifier_stats = train_classifier(
            classifier,
            X_train,
            y_train,
            X_val,
            y_val,
            device=device,
            epochs=args.classifier_epochs,
            lr=args.classifier_lr,
        )
        save_classifier_checkpoint(classifier, str(out_dir / "local_classifier.pt"))
        classifier_source = "trained_local_fixed_window"
        print(f"[classifier] best_val_acc={classifier_stats['best_val_acc']:.4f}", flush=True)

    baseline_metrics, baseline_results = evaluate_fixed_window(classifier, val_eps)
    print(f"[baseline] top1={baseline_metrics['char_top1']:.4f} top5={baseline_metrics['char_top5']:.4f} cer={baseline_metrics['cer']:.4f}", flush=True)

    cfg = SegmentalConfig(
        target_len=classifier.target_len,
        encoder_hidden=args.encoder_hidden,
        encoder_blocks=args.encoder_blocks,
    )
    model = SegmentalSequenceModel(cfg, classifier).to(device)
    model.freeze_classifier(not args.unfreeze_classifier)
    best_payload = train_segmental(
        model,
        train_eps,
        val_eps,
        device=device,
        epochs=args.segmental_epochs,
        lr=args.segmental_lr,
        output_dir=str(out_dir),
    )

    model.load_state_dict(best_payload["model_state"])
    seg_metrics, seg_results, seg_debug = evaluate_segmental(model, val_eps, device)
    report = {
        "classifier_source": classifier_source,
        "classifier_stats": classifier_stats,
        "baseline_fixed_window": baseline_metrics,
        "segmental": seg_metrics,
        "delta_top1": seg_metrics["char_top1"] - baseline_metrics["char_top1"],
        "delta_top5": seg_metrics["char_top5"] - baseline_metrics["char_top5"],
        "delta_cer": seg_metrics["cer"] - baseline_metrics["cer"],
        "best_epoch": best_payload.get("epoch"),
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
    }
    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "segmental_debug.json", "w", encoding="utf-8") as f:
        json.dump(seg_debug, f, ensure_ascii=False, indent=2)
    with open(out_dir / "segmental_results.json", "w", encoding="utf-8") as f:
        json.dump(seg_results, f, ensure_ascii=False, indent=2)
    with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
