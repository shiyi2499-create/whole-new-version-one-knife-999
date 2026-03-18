"""
Stage 2 Dense Structured Pipeline
=================================

Minimal training + inference module for the GPT-5.4 Stage 2 line.

This module keeps:
- Stage 1 coarse localization unchanged
- Stage 3 classifier unchanged

And introduces:
- dense patch-level training data from password/len_8 sessions
- lightweight temporal model with key/boundary/inside heads
- structured decode into exactly N passwords x K key slots
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_ONSET_DIR = os.path.dirname(HERE)
if PARENT_ONSET_DIR not in sys.path:
    sys.path.insert(0, PARENT_ONSET_DIR)

from onset_preprocessor import load_sensor_csv
from password_stage2_dataset import (
    PasswordStage2Dataset,
    PasswordStage2Sequence,
    pad_collate_password_stage2,
)
from password_stage2_model import build_password_stage2_model, temporal_smoothing_loss
from password_stage2_preprocessor import PatchConfig, build_patch_views, dense_targets_from_groups
from stage2_decoder import Stage2DecodeConfig, Stage2ProtocolPrior, decode_stage2_dense_topk

SUPPORTED_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789")


@dataclass
class DenseStage2Meta:
    patch_width_ms: int = 160
    patch_stride_ms: int = 20
    key_radius_ms: int = 60
    boundary_radius_ms: int = 120
    gap_expand_ms: int = 80
    hidden_dim: int = 64
    trunk_depth: int = 8
    refine_stages: int = 2
    refine_depth: int = 4
    dropout: float = 0.1


def supported_key(key: str) -> bool:
    return (key or "").lower() in SUPPORTED_CHARS


def _session_prefixes(password_dirs: list[str]) -> list[str]:
    out = []
    seen = set()
    for d in password_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith("_sensor.csv"):
                continue
            prefix = os.path.join(d, fn[:-11])
            if prefix not in seen:
                seen.add(prefix)
                out.append(prefix)
    return sorted(out)


def _part_idx(prefix: str) -> int:
    m = re.search(r"_part(\d+)_", os.path.basename(prefix))
    return int(m.group(1)) if m else -1


def _load_attempt_rows(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["prompt_index"] = int(row["prompt_index"])
                row["attempt_index"] = int(row["attempt_index"])
                row["attempt_start_ns"] = int(row["attempt_start_ns"])
                row["submit_ns"] = int(row["submit_ns"])
            except Exception:
                continue
            rows.append(row)
    return rows


def _load_press_rows(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") != "press":
                continue
            try:
                ts = int(row["timestamp_ns"])
            except Exception:
                continue
            rows.append({"timestamp_ns": ts, "key": (row.get("key") or "").lower()})
    return rows


def _extract_success_groups(prefix: str) -> list[tuple[str, list[int], int, int]]:
    attempts = _load_attempt_rows(prefix + "_attempts.csv")
    press_rows = _load_press_rows(prefix + "_events.csv")
    groups = []
    for row in attempts:
        prompt = (row.get("prompt_text") or "").strip().lower()
        typed = (row.get("typed_text") or "").strip().lower()
        if row.get("match") != "YES":
            continue
        if len(prompt) != 8 or typed != prompt:
            continue
        start_ns = row["attempt_start_ns"]
        end_ns = row["submit_ns"]
        keys = [
            r["timestamp_ns"]
            for r in press_rows
            if start_ns <= r["timestamp_ns"] <= end_ns and supported_key(r["key"])
        ]
        if len(keys) != len(prompt):
            continue
        groups.append((prompt, keys, start_ns, end_ns))
    return groups


def _crop_sensor(sensor: np.ndarray, start_ns: int, end_ns: int) -> np.ndarray:
    mask = (sensor[:, 0] >= start_ns) & (sensor[:, 0] <= end_ns)
    return sensor[mask]


def build_sequences_from_password_sessions(
    password_dirs: list[str],
    split_parts: set[int],
    patch_cfg: PatchConfig,
    window_passwords: int = 5,
    pre_margin_ms: int = 300,
    post_margin_ms: int = 500,
    context_variants_ms: Optional[list[tuple[int, int]]] = None,
) -> list[PasswordStage2Sequence]:
    seqs: list[PasswordStage2Sequence] = []
    context_variants_ms = context_variants_ms or [
        (pre_margin_ms, post_margin_ms),
        (max(120, pre_margin_ms - 120), post_margin_ms + 120),
        (pre_margin_ms + 120, max(180, post_margin_ms - 120)),
        (pre_margin_ms + 220, post_margin_ms + 220),
    ]
    for prefix in _session_prefixes(password_dirs):
        part = _part_idx(prefix)
        if part not in split_parts:
            continue
        groups = _extract_success_groups(prefix)
        if len(groups) < window_passwords:
            continue
        sensor = load_sensor_csv(prefix + "_sensor.csv")
        for start in range(0, len(groups) - window_passwords + 1):
            chunk = groups[start:start + window_passwords]
            for variant_idx, (pre_ms, post_ms) in enumerate(context_variants_ms):
                start_ns = chunk[0][2] - int(pre_ms * 1e6)
                end_ns = chunk[-1][3] + int(post_ms * 1e6)
                cropped = _crop_sensor(sensor, start_ns, end_ns)
                if len(cropped) < 20:
                    continue
                features, patch_times = build_patch_views(
                    cropped,
                    patch_width_ms=patch_cfg.patch_width_ms,
                    patch_stride_ms=patch_cfg.patch_stride_ms,
                )
                if len(patch_times) == 0:
                    continue
                rel_groups = [[int(t - start_ns) for t in g[1]] for g in chunk]
                rel_patch_times = patch_times - start_ns
                key_t, boundary_t, inside_t = dense_targets_from_groups(rel_patch_times, rel_groups, cfg=patch_cfg)
                seqs.append(
                    PasswordStage2Sequence(
                        features=features,
                        patch_times_ns=patch_times,
                        key_target=key_t,
                        boundary_target=boundary_t,
                        inside_target=inside_t,
                        session_id=os.path.basename(prefix),
                        source=f"part{part}_win{start}_ctx{variant_idx}",
                    )
                )
    return seqs


def _masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask_f = mask.float()
    return (loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def train_dense_stage2(
    train_sequences: list[PasswordStage2Sequence],
    val_sequences: list[PasswordStage2Sequence],
    checkpoint_path: str,
    scaler_path: str,
    report_path: str,
    meta: DenseStage2Meta,
    device: torch.device,
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> dict:
    train_ds = PasswordStage2Dataset(train_sequences, normalize=True)
    val_ds = PasswordStage2Dataset(val_sequences, normalize=True, means=train_ds.means, stds=train_ds.stds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_password_stage2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_password_stage2)

    model = build_password_stage2_model(
        n_channels=int(train_ds.means.shape[0]),
        hidden_dim=meta.hidden_dim,
        trunk_depth=meta.trunk_depth,
        refine_stages=meta.refine_stages,
        refine_depth=meta.refine_depth,
        dropout=meta.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best = {"val_loss": float("inf"), "epoch": 0}
    patience = 20
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for batch in train_loader:
            x = batch["features"].to(device)
            mask = batch["mask"].to(device)
            out = model(x)
            loss = (
                _masked_bce(out["key_logits"], batch["key_target"].to(device), mask)
                + 1.5 * _masked_bce(out["boundary_logits"], batch["boundary_target"].to(device), mask)
                + 0.5 * _masked_bce(out["inside_logits"], batch["inside_target"].to(device), mask)
                + 0.15 * temporal_smoothing_loss(out["key_logits"], mask)
                + 0.20 * temporal_smoothing_loss(out["boundary_logits"], mask)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_loss_sum += float(loss.item()) * x.size(0)
            train_items += x.size(0)
        sched.step()

        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                mask = batch["mask"].to(device)
                out = model(x)
                loss = (
                    _masked_bce(out["key_logits"], batch["key_target"].to(device), mask)
                    + 1.5 * _masked_bce(out["boundary_logits"], batch["boundary_target"].to(device), mask)
                    + 0.5 * _masked_bce(out["inside_logits"], batch["inside_target"].to(device), mask)
                    + 0.15 * temporal_smoothing_loss(out["key_logits"], mask)
                    + 0.20 * temporal_smoothing_loss(out["boundary_logits"], mask)
                )
                val_loss_sum += float(loss.item()) * x.size(0)
                val_items += x.size(0)

        train_loss = train_loss_sum / max(train_items, 1)
        val_loss = val_loss_sum / max(val_items, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": sched.get_last_lr()[0]})
        if epoch == 1 or epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}")

        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch}
            bad_epochs = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "n_channels": int(train_ds.means.shape[0]),
                "meta": meta.__dict__,
                "best_val_loss": val_loss,
                "epoch": epoch,
            }, checkpoint_path)
            np.savez(scaler_path, means=train_ds.means, stds=train_ds.stds)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    report = {
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "best": best,
        "history": history,
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    import json
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Best val_loss: {best['val_loss']:.4f}")
    print(f"  Checkpoint -> {checkpoint_path}")
    return report


def load_dense_stage2_model(checkpoint_path: str, scaler_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler = np.load(scaler_path)
    means = scaler["means"].astype(np.float32)
    stds = np.maximum(scaler["stds"].astype(np.float32), 1e-6)
    meta_dict = ckpt.get("meta", {})
    meta = DenseStage2Meta(**{k: v for k, v in meta_dict.items() if k in DenseStage2Meta.__annotations__})
    model = build_password_stage2_model(
        n_channels=int(ckpt["n_channels"]),
        hidden_dim=meta.hidden_dim,
        trunk_depth=meta.trunk_depth,
        refine_stages=meta.refine_stages,
        refine_depth=meta.refine_depth,
        dropout=meta.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    model._dense_stage2_meta = meta
    return model, means, stds, meta, ckpt


def infer_dense_stage2_on_region(
    model,
    sensor: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    region_start_s: float,
    region_end_s: float,
    meta: DenseStage2Meta,
):
    mask = (sensor[:, 0] >= region_start_s * 1e9) & (sensor[:, 0] <= region_end_s * 1e9)
    region = sensor[mask]
    if len(region) < 20:
        return None
    features, patch_times = build_patch_views(region, meta.patch_width_ms, meta.patch_stride_ms)
    if len(patch_times) == 0:
        return None
    x = (features.astype(np.float32) - means) / stds
    with torch.no_grad():
        out = model(torch.from_numpy(x).unsqueeze(0).to(device))
    key_score = torch.sigmoid(out["key_logits"]).squeeze(0).cpu().numpy()
    boundary_score = torch.sigmoid(out["boundary_logits"]).squeeze(0).cpu().numpy()
    inside_score = torch.sigmoid(out["inside_logits"]).squeeze(0).cpu().numpy()
    return {
        "patch_times_ns": patch_times,
        "key_score": key_score,
        "boundary_score": boundary_score,
        "inside_score": inside_score,
    }


def run_stage2_dense(
    sensor: np.ndarray,
    coarse_regions,
    model,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    expected_password_count: int = 5,
    expected_password_len: int = 8,
    min_segment_s: float = 0.8,
    max_segment_s: float = 8.0,
    n_best_hypotheses: int = 12,
):
    if not coarse_regions:
        return [], {"method": "dense_structured", "regions": []}

    region = coarse_regions[0]
    ckpt_meta = getattr(model, "_dense_stage2_meta", None)
    meta = ckpt_meta if ckpt_meta is not None else DenseStage2Meta()
    pred = infer_dense_stage2_on_region(model, sensor, means, stds, device, region.start_s, region.end_s, meta)
    if pred is None:
        return [], {"method": "dense_structured", "regions": []}

    patch_times_s = pred["patch_times_ns"].astype(np.float64) / 1e9
    prior = Stage2ProtocolPrior(
        expected_password_count=expected_password_count,
        expected_password_len=expected_password_len,
        min_segment_s=min_segment_s,
        max_segment_s=max_segment_s,
    )
    cfg = Stage2DecodeConfig(n_best_hypotheses=max(1, int(n_best_hypotheses)))
    decoded_hyps = decode_stage2_dense_topk(
        patch_times_s,
        pred["key_score"],
        pred["boundary_score"],
        pred["inside_score"],
        prior=prior,
        cfg=cfg,
    )
    decoded = decoded_hyps[0] if decoded_hyps else None
    groups_s = decoded.password_groups_s if decoded is not None else []

    hypotheses_debug = []
    for hyp in decoded_hyps:
        hypotheses_debug.append({
            "hypothesis_rank": int(hyp.hypothesis_rank),
            "stage2_score": float(hyp.total_score),
            "boundary_score": float(hyp.boundary_score),
            "boundary_indices": [int(i) for i in hyp.boundary_indices],
            "boundary_times_s": [float(patch_times_s[i]) for i in hyp.boundary_indices],
            "segment_scores": [float(seg.score) for seg in hyp.segments],
            "group_lengths": [len(seg.key_indices) for seg in hyp.segments],
            "segment_patch_ranges": [[int(seg.start_idx), int(seg.end_idx)] for seg in hyp.segments],
            "key_indices_by_group": [[int(i) for i in seg.key_indices] for seg in hyp.segments],
            "key_times_s_by_group": hyp.password_groups_s,
        })

    debug = {
        "method": "dense_structured",
        "regions": [{
            "start_s": region.start_s,
            "end_s": region.end_s,
            "n_patches": int(len(patch_times_s)),
            "patch_times_s": patch_times_s.tolist(),
            "key_score": pred["key_score"].astype(float).tolist(),
            "boundary_score": pred["boundary_score"].astype(float).tolist(),
            "inside_score": pred["inside_score"].astype(float).tolist(),
            "n_hypotheses": int(len(hypotheses_debug)),
            "selected_stage2_hypothesis_rank": 0,
            "hypotheses": hypotheses_debug,
            "boundary_indices": hypotheses_debug[0]["boundary_indices"] if hypotheses_debug else [],
            "boundary_times_s": hypotheses_debug[0]["boundary_times_s"] if hypotheses_debug else [],
            "segment_scores": hypotheses_debug[0]["segment_scores"] if hypotheses_debug else [],
            "group_lengths": hypotheses_debug[0]["group_lengths"] if hypotheses_debug else [],
            "segment_patch_ranges": hypotheses_debug[0]["segment_patch_ranges"] if hypotheses_debug else [],
            "key_indices_by_group": hypotheses_debug[0]["key_indices_by_group"] if hypotheses_debug else [],
            "key_times_s_by_group": hypotheses_debug[0]["key_times_s_by_group"] if hypotheses_debug else [],
        }],
    }
    return groups_s, debug


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--password-dirs", nargs="+", default=["data/raw/password/len_8"])
    p.add_argument("--checkpoint", default="results/gpt54_dense_stage2.pt")
    p.add_argument("--scaler", default="results/gpt54_dense_stage2_scaler.npz")
    p.add_argument("--report", default="results/gpt54_dense_stage2_report.json")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-parts", default="1-16")
    p.add_argument("--val-parts", default="17-20")
    args = p.parse_args()

    root = os.path.abspath(args.project_root)
    password_dirs = [d if os.path.isabs(d) else os.path.join(root, d) for d in args.password_dirs]
    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(root, args.checkpoint)
    scaler = args.scaler if os.path.isabs(args.scaler) else os.path.join(root, args.scaler)
    report = args.report if os.path.isabs(args.report) else os.path.join(root, args.report)

    req = args.device.lower()
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    device = torch.device(req)
    print(f"Device: {device}")

    def parse_parts(spec: str) -> set[int]:
        out = set()
        for chunk in spec.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            if '-' in chunk:
                a, b = chunk.split('-', 1)
                out.update(range(int(a), int(b) + 1))
            else:
                out.add(int(chunk))
        return out

    meta = DenseStage2Meta()
    patch_cfg = PatchConfig(
        patch_width_ms=meta.patch_width_ms,
        patch_stride_ms=meta.patch_stride_ms,
        key_radius_ms=meta.key_radius_ms,
        boundary_radius_ms=meta.boundary_radius_ms,
        gap_expand_ms=meta.gap_expand_ms,
    )
    train_sequences = build_sequences_from_password_sessions(password_dirs, parse_parts(args.train_parts), patch_cfg)
    val_sequences = build_sequences_from_password_sessions(password_dirs, parse_parts(args.val_parts), patch_cfg)
    print(f"Train sequences: {len(train_sequences)}  Val sequences: {len(val_sequences)}")
    if not train_sequences or not val_sequences:
        raise SystemExit("Need non-empty train/val sequences for dense Stage 2 training")
    train_dense_stage2(train_sequences, val_sequences, checkpoint, scaler, report, meta, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == '__main__':
    main()
