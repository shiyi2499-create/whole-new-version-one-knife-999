#!/usr/bin/env python3
"""
Password-like / continuous-string closure using an InceptionTime baseline.

This script is intentionally self-contained so it can live outside the main
training workspace and still be copied into another machine/server.

Design goals:
- train the final baseline on merged single_key + boost data
- exclude space / enter / backspace from the classifier target space
- evaluate free_type as no-space continuous character strings
- report character-level and exact-string metrics without relying on LM
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from scipy.signal import resample
except ImportError as e:
    raise ImportError("scipy is required: pip install scipy") from e

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    raise ImportError("scikit-learn is required: pip install scikit-learn") from e

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as e:
    raise ImportError("torch is required: pip install torch") from e


SUPPORTED_RE = re.compile(r"^[a-z0-9]$")


def supported_key(key: str) -> bool:
    return bool(SUPPORTED_RE.match((key or "").lower()))


def normalize_sequence(text: str) -> str:
    text = (text or "").lower()
    return "".join(ch for ch in text if supported_key(ch))


def resolve_torch_device(device: str = "auto") -> torch.device:
    req = (device or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"

    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested device=cuda but CUDA is not available.")
    if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested device=mps but MPS is not available.")
    if req not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unsupported device. Use one of auto/cpu/mps/cuda.")
    return torch.device(req)


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class WindowConfig:
    pre_trigger_ms: int = 100
    post_trigger_ms: int = 200
    target_rate_hz: int = 190
    min_window_samples: int = 2

    @property
    def target_window_len(self) -> int:
        return int((self.pre_trigger_ms + self.post_trigger_ms) / 1000 * self.target_rate_hz)


class SessionWindowExtractor:
    def __init__(self, session_prefix: str, window_cfg: Optional[WindowConfig] = None):
        self.session_prefix = session_prefix
        self.wcfg = window_cfg or WindowConfig()
        self.sensor_path = session_prefix + "_sensor.csv"
        self.events_path = session_prefix + "_events.csv"
        self.sensor_data: Optional[np.ndarray] = None

    def load_sensor(self):
        rows = []
        with open(self.sensor_path, "r") as f:
            for row in csv.DictReader(f):
                rows.append([
                    int(row["timestamp_ns"]),
                    float(row["accel_x"]),
                    float(row["accel_y"]),
                    float(row["accel_z"]),
                    float(row["gyro_x"]),
                    float(row["gyro_y"]),
                    float(row["gyro_z"]),
                ])
        self.sensor_data = np.asarray(rows, dtype=np.float64)

    @staticmethod
    def resample_window(values: np.ndarray, target_len: int) -> np.ndarray:
        out = resample(values, target_len, axis=0)
        if np.iscomplexobj(out):
            out = np.real(out)
        return np.asarray(out, dtype=np.float32)

    def extract_window(self, evt_ts: int) -> Optional[np.ndarray]:
        if self.sensor_data is None:
            self.load_sensor()
        ts = self.sensor_data[:, 0]
        vals = self.sensor_data[:, 1:]
        pre_ns = self.wcfg.pre_trigger_ms * 1_000_000
        post_ns = self.wcfg.post_trigger_ms * 1_000_000
        idx_start = np.searchsorted(ts, evt_ts - pre_ns, side="left")
        idx_end = np.searchsorted(ts, evt_ts + post_ns, side="right")
        if idx_end - idx_start < self.wcfg.min_window_samples:
            return None
        window = vals[idx_start:idx_end]
        return self.resample_window(window, self.wcfg.target_window_len)


def _pick_inception_kernels(n_timesteps: int) -> tuple[int, int, int]:
    cap = max(7, int(n_timesteps))
    if cap % 2 == 0:
        cap -= 1
    kernels = []
    for k in (9, 19, 39):
        kk = min(k, cap)
        if kk % 2 == 0:
            kk -= 1
        kk = max(3, kk)
        if kk not in kernels:
            kernels.append(kk)
    cur = kernels[-1] if kernels else 7
    while len(kernels) < 3:
        cur = max(3, cur - 2)
        if cur not in kernels:
            kernels.append(cur)
    return tuple(kernels[:3])


class InceptionModule1D(nn.Module):
    def __init__(self, in_channels: int, n_filters: int,
                 kernel_sizes: tuple[int, int, int], bottleneck: int = 32):
        super().__init__()
        self.use_bottleneck = in_channels > 1 and bottleneck > 0
        if self.use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck, kernel_size=1, bias=False)
            branch_in = bottleneck
        else:
            self.bottleneck = nn.Identity()
            branch_in = in_channels
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(branch_in, n_filters, kernel_size=k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, n_filters, kernel_size=1, bias=False),
        )
        self.bn = nn.BatchNorm1d(n_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_b = self.bottleneck(x)
        outs = [conv(x_b) for conv in self.conv_branches]
        outs.append(self.pool_branch(x))
        return self.relu(self.bn(torch.cat(outs, dim=1)))


class InceptionResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, n_filters: int, kernel_sizes: tuple[int, int, int]):
        super().__init__()
        self.m1 = InceptionModule1D(in_channels, n_filters, kernel_sizes)
        mid_channels = n_filters * 4
        self.m2 = InceptionModule1D(mid_channels, n_filters, kernel_sizes)
        self.m3 = InceptionModule1D(mid_channels, n_filters, kernel_sizes)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.m1(x)
        out = self.m2(out)
        out = self.m3(out)
        return self.relu(out + self.shortcut(x))


class InceptionTimeClassifier(nn.Module):
    def __init__(self, n_timesteps=57, n_channels=6, n_classes=36,
                 n_filters=32, n_blocks=2):
        super().__init__()
        kernels = _pick_inception_kernels(n_timesteps)
        blocks = []
        in_ch = n_channels
        for _ in range(n_blocks):
            blocks.append(InceptionResidualBlock1D(in_ch, n_filters, kernels))
            in_ch = n_filters * 4
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(in_ch, n_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.backbone(x)
        return self.head(x)


def load_merged_training_data(merged_path: str, max_samples: int = 0):
    raw = np.load(merged_path, allow_pickle=True)
    X = raw["X"].astype(np.float32)
    y = raw["y"].astype(str)

    mask = np.array([supported_key(k) for k in y], dtype=bool)
    X = X[mask]
    y = y[mask]

    if max_samples and len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), size=max_samples, replace=False)
        idx = np.sort(idx)
        X = X[idx]
        y = y[idx]

    classes = np.array(sorted(set(y.tolist())))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([class_to_idx[k] for k in y], dtype=np.int64)
    return X, y, y_enc, classes


def train_final_inception(
    X: np.ndarray,
    y_enc: np.ndarray,
    classes: np.ndarray,
    checkpoint_path: str,
    scaler_path: str,
    device: torch.device,
    force: bool = False,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 8e-4,
    patience: int = 20,
):
    if (not force) and os.path.exists(checkpoint_path) and os.path.exists(scaler_path):
        print(f"  Found saved model: {checkpoint_path}")
        return

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    n_timesteps = X.shape[1]
    n_channels = X.shape[2]
    n_classes = len(classes)

    ch_means = X.mean(axis=(0, 1))
    ch_stds = X.std(axis=(0, 1))
    X_norm = X.copy()
    for ch in range(n_channels):
        X_norm[:, :, ch] = (X_norm[:, :, ch] - ch_means[ch]) / (ch_stds[ch] + 1e-10)

    indices = np.arange(len(X_norm))
    test_size = max(0.1, 1 / len(indices))
    counts = Counter(y_enc.tolist())
    use_stratify = min(counts.values()) >= 2 and len(indices) >= len(counts) * 2
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=y_enc if use_stratify else None,
    )
    X_train = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_enc[train_idx], dtype=torch.long)
    X_val = torch.tensor(X_norm[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y_enc[val_idx], dtype=torch.long).to(device)

    model = InceptionTimeClassifier(
        n_timesteps=n_timesteps,
        n_channels=n_channels,
        n_classes=n_classes,
    ).to(device)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = -1.0
    best_state = None
    patience_ctr = 0
    t0 = time.time()
    print(f"  Training InceptionTime final model")
    print(f"  Samples: {len(X_norm)} | Classes: {n_classes} | Timesteps: {n_timesteps} | Device: {device}")
    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for xb, yb in loader:
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
            val_acc = float((model(X_val).argmax(1) == y_val).float().mean().item())
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_acc = correct / max(total, 1)
            print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
        if patience_ctr >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best state captured.")
    model.load_state_dict(best_state)
    torch.save({
        "model_state": model.state_dict(),
        "n_timesteps": n_timesteps,
        "n_channels": n_channels,
        "n_classes": n_classes,
        "classes": classes,
        "model_name": "InceptionTime",
    }, checkpoint_path)
    np.savez(scaler_path, means=ch_means, stds=ch_stds)
    print(f"  Best val acc: {best_val:.3f} | Total time: {time.time()-t0:.1f}s")
    print(f"  Saved -> {checkpoint_path}")
    print(f"  Saved -> {scaler_path}")


def load_final_inception(checkpoint_path: str, scaler_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = InceptionTimeClassifier(
        n_timesteps=int(ckpt["n_timesteps"]),
        n_channels=int(ckpt["n_channels"]),
        n_classes=int(ckpt["n_classes"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = np.load(scaler_path)
    return model, np.array(ckpt["classes"]).astype(str), scaler["means"], scaler["stds"]


def discover_freetype_sessions(freetype_dirs: list[str]) -> list[str]:
    sessions = []
    for rd in freetype_dirs:
        if not os.path.isdir(rd):
            print(f"  ⚠ Not found: {rd}")
            continue
        for f in sorted(os.listdir(rd)):
            if "_free_type_" in f and f.endswith("_sensor.csv"):
                prefix = os.path.join(rd, f.replace("_sensor.csv", ""))
                if os.path.exists(prefix + "_events.csv"):
                    sessions.append(prefix)
    return sessions


def read_attempt_rows(session_prefix: str) -> list[dict]:
    path = session_prefix + "_attempts.csv"
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_no_space_sequences(
    session_prefix: str,
    yes_only: bool = True,
    eval_max_sequences: int = 0,
) -> list[dict]:
    extractor = SessionWindowExtractor(session_prefix, window_cfg=WindowConfig(min_window_samples=2))
    attempts = read_attempt_rows(session_prefix)
    events_path = session_prefix + "_events.csv"
    sequences = []
    cur_events = []
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["event_type"] != "press":
                continue
            key = row["key"].lower()
            ts = int(row["timestamp_ns"])
            if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
                       "left", "right", "up", "down", "delete"}:
                continue
            if key in {"enter", "return"}:
                sequences.append({"events": cur_events.copy(), "submit_ns": ts})
                cur_events = []
                if eval_max_sequences and len(sequences) >= eval_max_sequences:
                    break
                continue
            cur_events.append({"key": key, "timestamp_ns": ts})

    out = []
    for idx, seq in enumerate(sequences):
        att = attempts[idx] if idx < len(attempts) else {}
        match = (att.get("match") or "").upper()
        if yes_only and match and match != "YES":
            continue
        ref = normalize_sequence(att.get("typed_text", ""))
        items = []
        for evt in seq["events"]:
            key = evt["key"]
            if key in {"space", "backspace"}:
                continue
            if not supported_key(key):
                continue
            window = extractor.extract_window(evt["timestamp_ns"])
            if window is None:
                continue
            items.append({"key": key, "timestamp_ns": evt["timestamp_ns"], "window": window})
        if not ref or not items:
            continue
        out.append({
            "session": os.path.basename(session_prefix),
            "sequence_idx": idx,
            "reference": ref,
            "items": items,
        })
    return out


def infer_one(model, window: np.ndarray, means: np.ndarray, stds: np.ndarray, device: torch.device) -> np.ndarray:
    w = window.copy().astype(np.float32)
    for ch in range(w.shape[1]):
        w[:, ch] = (w[:, ch] - means[ch]) / (stds[ch] + 1e-10)
    with torch.no_grad():
        xb = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def topk_strings_from_prob_vectors(
    prob_vectors: list[np.ndarray],
    classes: np.ndarray,
    branch_topk: int = 5,
    beam_width: int = 100,
) -> list[dict]:
    """
    Beam-search over per-position class probabilities to generate top-N
    candidate strings.

    Each position is expanded only by its local top-k classes to keep the
    combinatorics manageable for long password-like strings.
    """
    if not prob_vectors:
        return []

    beam = [("", 0.0)]  # (string, cumulative_log_prob)
    for probs in prob_vectors:
        top_idx = np.argsort(probs)[::-1][:branch_topk]
        expanded = []
        for prefix, score in beam:
            for idx in top_idx:
                p = float(probs[idx])
                expanded.append((prefix + str(classes[int(idx)]), score + math.log(max(p, 1e-12))))
        expanded.sort(key=lambda x: x[1], reverse=True)
        beam = expanded[:beam_width]

    return [
        {
            "candidate": cand,
            "log_prob": score,
        }
        for cand, score in beam
    ]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def evaluate_sequences(
    sequences: list[dict],
    model,
    classes: np.ndarray,
    means,
    stds,
    device: torch.device,
    char_topk: tuple[int, ...] = (1, 3, 5),
    seq_branch_topk: int = 5,
    seq_beam_width: int = 100,
    seq_hit_cutoffs: tuple[int, ...] = (10, 50, 100),
):
    reports = []
    total_chars = 0
    correct_chars = 0
    exact = 0
    total_edits = 0
    topk_correct = {k: 0 for k in char_topk}
    seq_hits = {k: 0 for k in seq_hit_cutoffs}
    class_set = set(map(str, classes.tolist()))
    supported_ref_chars = 0
    unsupported_ref_chars = 0
    topk_correct_supported = {k: 0 for k in char_topk}
    sequences_supported_only = 0
    seq_hits_supported_only = {k: 0 for k in seq_hit_cutoffs}

    for seq in sequences:
        hyp_chars = []
        prob_vectors = []
        topk_per_pos = []
        for item in seq["items"]:
            probs = infer_one(model, item["window"], means, stds, device)
            prob_vectors.append(probs)
            ranked_idx = np.argsort(probs)[::-1]
            topk_per_pos.append([str(classes[int(i)]) for i in ranked_idx[: max(char_topk)]])
            hyp_chars.append(classes[int(ranked_idx[0])])
        hyp = "".join(hyp_chars)
        ref = seq["reference"]
        ref_supported_flags = [ch in class_set for ch in ref]
        all_ref_supported = all(ref_supported_flags)
        matches = sum(a == b for a, b in zip(ref, hyp))
        total_chars += len(ref)
        correct_chars += matches
        supported_ref_chars += sum(ref_supported_flags)
        unsupported_ref_chars += len(ref) - sum(ref_supported_flags)
        for ref_ch, preds, is_supported in zip(ref, topk_per_pos, ref_supported_flags):
            for k in char_topk:
                if ref_ch in preds[:k]:
                    topk_correct[k] += 1
                    if is_supported:
                        topk_correct_supported[k] += 1
        edits = levenshtein(ref, hyp)
        total_edits += edits
        if ref == hyp:
            exact += 1

        candidates = topk_strings_from_prob_vectors(
            prob_vectors,
            classes,
            branch_topk=seq_branch_topk,
            beam_width=max(seq_beam_width, max(seq_hit_cutoffs)),
        )
        candidate_strings = [c["candidate"] for c in candidates]
        for cutoff in seq_hit_cutoffs:
            if ref in candidate_strings[:cutoff]:
                seq_hits[cutoff] += 1
        if all_ref_supported:
            sequences_supported_only += 1
            for cutoff in seq_hit_cutoffs:
                if ref in candidate_strings[:cutoff]:
                    seq_hits_supported_only[cutoff] += 1
        reports.append({
            "session": seq["session"],
            "sequence_idx": seq["sequence_idx"],
            "reference": ref,
            "decoded_top1": hyp,
            "ref_len": len(ref),
            "hyp_len": len(hyp),
            "char_positional_acc": matches / max(len(ref), 1),
            "edit_distance": edits,
            "reference_has_unseen_char": not all_ref_supported,
            "char_topk_preview": topk_per_pos[:10],
            "top_sequence_candidates": candidates[:10],
        })

    metrics = {
        "total_sequences": len(sequences),
        "exact_matches_top1": exact,
        "exact_match_rate_top1": exact / max(len(sequences), 1),
        "total_ref_chars": total_chars,
        "char_top1_accuracy": correct_chars / max(total_chars, 1),
        "cer_top1": total_edits / max(total_chars, 1),
        "supported_ref_chars": supported_ref_chars,
        "unsupported_ref_chars": unsupported_ref_chars,
        "unsupported_ref_char_rate": unsupported_ref_chars / max(total_chars, 1),
        "supported_only_sequences": sequences_supported_only,
    }
    for k in char_topk:
        metrics[f"char_top{k}_accuracy"] = topk_correct[k] / max(total_chars, 1)
        metrics[f"char_top{k}_accuracy_supported_vocab"] = (
            topk_correct_supported[k] / max(supported_ref_chars, 1)
        )
    for cutoff in seq_hit_cutoffs:
        metrics[f"sequence_top{cutoff}_hit_rate"] = seq_hits[cutoff] / max(len(sequences), 1)
        metrics[f"sequence_top{cutoff}_hit_rate_supported_vocab"] = (
            seq_hits_supported_only[cutoff] / max(sequences_supported_only, 1)
        )
    return metrics, reports


def parse_args():
    parser = argparse.ArgumentParser(description="Inception no-space continuous-string closure")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--merged-path", default="data/processed/merged_dataset.npz")
    parser.add_argument(
        "--free-type-dirs",
        nargs="+",
        default=["data/raw/trial_nonroot_free_type_refill"],
        help="One or more directories containing *_free_type_* raw files.",
    )
    parser.add_argument("--checkpoint-path", default="results/inception_password_final.pt")
    parser.add_argument("--scaler-path", default="results/inception_password_scaler.npz")
    parser.add_argument("--report-path", default="results/password_closure_inception.json")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--train-max-samples", type=int, default=0,
                        help="Optional small cap for smoke tests.")
    parser.add_argument("--eval-max-sequences", type=int, default=0,
                        help="Optional small cap for smoke tests.")
    parser.add_argument("--yes-only", action="store_true", default=True)
    parser.add_argument("--seq-branch-topk", type=int, default=5,
                        help="Per-position top-k used when expanding sequence candidates.")
    parser.add_argument("--seq-beam-width", type=int, default=100,
                        help="Beam width for no-space sequence candidate generation.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_torch_device(args.device)
    set_global_seed(42)
    print(f"Device: {device}")

    if not args.no_train:
        X, y, y_enc, classes = load_merged_training_data(args.merged_path, max_samples=args.train_max_samples)
        print(f"Training data after no-space filtering: X={X.shape}, classes={len(classes)}")
        train_final_inception(
            X, y_enc, classes,
            checkpoint_path=args.checkpoint_path,
            scaler_path=args.scaler_path,
            device=device,
            force=args.force_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )
    else:
        if not (os.path.exists(args.checkpoint_path) and os.path.exists(args.scaler_path)):
            raise FileNotFoundError("Checkpoint/scaler missing but --no-train was given.")

    model, classes, means, stds = load_final_inception(args.checkpoint_path, args.scaler_path, device)
    n_timesteps = int(torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)["n_timesteps"])
    print(f"Loaded Inception model: n_classes={len(classes)}, n_timesteps={n_timesteps}")
    print(f"Classifier classes: {' '.join(classes.tolist())}")

    sessions = discover_freetype_sessions(args.free_type_dirs)
    if not sessions:
        raise RuntimeError("No free_type sessions found.")
    print(f"Found {len(sessions)} free_type sessions")

    sequences = []
    for sess in sessions:
        seqs = build_no_space_sequences(sess, yes_only=args.yes_only, eval_max_sequences=args.eval_max_sequences)
        print(f"  {os.path.basename(sess)} -> {len(seqs)} no-space sequences")
        sequences.extend(seqs)
        if args.eval_max_sequences and len(sequences) >= args.eval_max_sequences:
            sequences = sequences[:args.eval_max_sequences]
            break

    if not sequences:
        raise RuntimeError("No valid no-space sequences found.")

    metrics, reports = evaluate_sequences(
        sequences,
        model,
        classes,
        means,
        stds,
        device,
        char_topk=(1, 3, 5),
        seq_branch_topk=args.seq_branch_topk,
        seq_beam_width=args.seq_beam_width,
        seq_hit_cutoffs=(10, 50, 100),
    )
    print("\nFINAL METRICS")
    print(f"  total_sequences:         {metrics['total_sequences']}")
    print(f"  exact_matches_top1:     {metrics['exact_matches_top1']} ({metrics['exact_match_rate_top1']:.1%})")
    print(f"  total_ref_chars:        {metrics['total_ref_chars']}")
    print(f"  unsupported_ref_chars: {metrics['unsupported_ref_chars']} ({metrics['unsupported_ref_char_rate']:.1%})")
    print(f"  char_top1_acc:          {metrics['char_top1_accuracy']:.1%}")
    print(f"  char_top3_acc:          {metrics['char_top3_accuracy']:.1%}")
    print(f"  char_top5_acc:          {metrics['char_top5_accuracy']:.1%}")
    print(f"  char_top1_acc_seen:     {metrics['char_top1_accuracy_supported_vocab']:.1%}")
    print(f"  char_top3_acc_seen:     {metrics['char_top3_accuracy_supported_vocab']:.1%}")
    print(f"  char_top5_acc_seen:     {metrics['char_top5_accuracy_supported_vocab']:.1%}")
    print(f"  sequence_top10_hit:     {metrics['sequence_top10_hit_rate']:.1%}")
    print(f"  sequence_top50_hit:     {metrics['sequence_top50_hit_rate']:.1%}")
    print(f"  sequence_top100_hit:    {metrics['sequence_top100_hit_rate']:.1%}")
    print(f"  supported_only_seqs:    {metrics['supported_only_sequences']}")
    print(f"  CER_top1:               {metrics['cer_top1']:.1%}")

    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump({
            "device": str(device),
            "merged_path": args.merged_path,
            "free_type_dirs": args.free_type_dirs,
            "checkpoint_path": args.checkpoint_path,
            "metrics": metrics,
            "examples": reports[:20],
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {args.report_path}")


if __name__ == "__main__":
    main()
