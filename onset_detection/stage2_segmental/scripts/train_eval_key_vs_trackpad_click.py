#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader, discover_sessions
from onset_detection.stage2_segmental.data import estimate_sample_rate_hz
from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import (
    _load_mixed_episodes,
    _load_password_attempt_episodes,
    _propose_peaks,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class WindowSample:
    session_id: str
    source: str
    label: int
    window: np.ndarray  # [T, 6]


def resolve_device(name: str) -> torch.device:
    req = str(name).lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"
    return torch.device(req)


def _resample_channelwise(window: np.ndarray, target_len: int) -> np.ndarray:
    src_len = int(window.shape[0])
    if src_len <= 1:
        return np.repeat(window[:1], target_len, axis=0).astype(np.float32)
    src_grid = np.linspace(0.0, 1.0, num=src_len, dtype=np.float64)
    dst_grid = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    out = np.stack([
        np.interp(dst_grid, src_grid, window[:, ch].astype(np.float64)).astype(np.float32)
        for ch in range(window.shape[1])
    ], axis=1)
    return out.astype(np.float32)


def _extract_window(imu: np.ndarray, center_frame: int, sample_rate_hz: float, pre_ms: float, post_ms: float, target_len: int) -> np.ndarray | None:
    pre_frames = int(round(float(pre_ms) / 1000.0 * float(sample_rate_hz)))
    post_frames = int(round(float(post_ms) / 1000.0 * float(sample_rate_hz)))
    lo = max(0, int(center_frame) - pre_frames)
    hi = min(len(imu), int(center_frame) + post_frames)
    if hi - lo < 3:
        return None
    return _resample_channelwise(np.asarray(imu[lo:hi], dtype=np.float32), target_len)


def _collect_positive_samples(
    password_dirs: list[str],
    mixed_dirs: list[str],
    pre_ms: float,
    post_ms: float,
    target_len: int,
) -> list[WindowSample]:
    out: list[WindowSample] = []
    for d in password_dirs:
        for ep in _load_password_attempt_episodes(d):
            for frame in ep["key_frames"]:
                win = _extract_window(ep["imu"], int(frame), float(ep["sample_rate_hz"]), pre_ms, post_ms, target_len)
                if win is None:
                    continue
                out.append(
                    WindowSample(
                        session_id=str(ep["session_id"]),
                        source="password",
                        label=1,
                        window=win,
                    )
                )
    for d in mixed_dirs:
        for ep in _load_mixed_episodes(d):
            for frame in ep["key_frames"]:
                win = _extract_window(ep["imu"], int(frame), float(ep["sample_rate_hz"]), pre_ms, post_ms, target_len)
                if win is None:
                    continue
                out.append(
                    WindowSample(
                        session_id=str(ep["session_id"]),
                        source="mixed_password",
                        label=1,
                        window=win,
                    )
                )
    return out


def _collect_trackpad_click_samples(
    trackpad_dirs: list[str],
    pre_ms: float,
    post_ms: float,
    target_len: int,
    max_peaks_per_session: int,
    min_rel_energy: float,
) -> list[WindowSample]:
    out: list[WindowSample] = []
    for root in trackpad_dirs:
        for sess in sorted(discover_sessions(root)):
            loader = SessionLoader(sess)
            ts_ns, imu = loader.get_imu()
            if len(ts_ns) < 8:
                continue
            sr = estimate_sample_rate_hz(ts_ns)
            peaks, smoothed, _ = _propose_peaks({"imu": imu, "timestamps_ns": ts_ns, "sample_rate_hz": sr})
            if len(peaks) == 0:
                continue
            peak_energy = smoothed[np.clip(peaks, 0, len(smoothed) - 1)].astype(np.float64)
            rel = peak_energy / max(float(np.max(peak_energy)), 1e-8)
            keep = np.where(rel >= float(min_rel_energy))[0]
            if len(keep) == 0:
                order = np.argsort(-peak_energy)[: max_peaks_per_session]
            else:
                order = keep[np.argsort(-peak_energy[keep])]
            if max_peaks_per_session > 0:
                order = order[: max_peaks_per_session]
            session_id = Path(sess).name
            for idx in order.tolist():
                frame = int(peaks[int(idx)])
                win = _extract_window(imu, frame, float(sr), pre_ms, post_ms, target_len)
                if win is None:
                    continue
                out.append(
                    WindowSample(
                        session_id=session_id,
                        source="trackpad_click",
                        label=0,
                        window=win,
                    )
                )
    return out


def _segments_overlap(seg_start: int, seg_end: int, gt_start: int, gt_end: int) -> bool:
    return max(0, min(int(seg_end), int(gt_end)) - max(int(seg_start), int(gt_start))) > 0


def _collect_hard_negative_samples(
    result_json: str,
    session_dirs: list[str],
    pre_ms: float,
    post_ms: float,
    target_len: int,
    min_segment_confidence: float,
    min_peak_probability: float,
    max_peaks_per_segment: int,
) -> list[WindowSample]:
    if not result_json:
        return []
    path = Path(result_json)
    if not path.exists():
        raise FileNotFoundError(f"hard negative result json not found: {result_json}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    details = blob.get("details", [])
    if not isinstance(details, list):
        return []

    session_lookup: dict[str, str] = {}
    for root in session_dirs:
        for sess in sorted(discover_sessions(root)):
            session_lookup[Path(sess).name] = sess

    out: list[WindowSample] = []
    for row in details:
        session_id = str(row.get("session_id", ""))
        session_path = session_lookup.get(session_id)
        if not session_path:
            continue
        gt_rows = row.get("gt_rows", []) or []
        loader = SessionLoader(session_path)
        ts_ns, imu = loader.get_imu()
        if len(ts_ns) < 8:
            continue
        sr = estimate_sample_rate_hz(ts_ns)

        candidates = row.get("pred_segments_top5_after_filter", []) or []
        for seg in candidates:
            seg_start = int(seg.get("start_frame", 0))
            seg_end = int(seg.get("end_frame", 0))
            seg_conf = float(seg.get("confidence", 0.0))
            if seg_conf < float(min_segment_confidence):
                continue
            if any(
                _segments_overlap(seg_start, seg_end, int(gt.get("gt_start_frame", 0)), int(gt.get("gt_end_frame", 0)))
                for gt in gt_rows
            ):
                continue

            peak_frames = np.asarray(seg.get("selected_peak_frames", []) or [], dtype=np.int64)
            peak_probs = np.asarray(seg.get("selected_peak_probs", []) or [], dtype=np.float64)
            if len(peak_frames) == 0 or len(peak_frames) != len(peak_probs):
                crop_imu = np.asarray(imu[seg_start:seg_end], dtype=np.float32)
                crop_ts = np.asarray(ts_ns[seg_start:seg_end], dtype=np.int64)
                if len(crop_ts) < 8:
                    continue
                peaks, smoothed, _ = _propose_peaks(
                    {
                        "imu": crop_imu,
                        "timestamps_ns": crop_ts,
                        "sample_rate_hz": float(sr),
                    }
                )
                if len(peaks) == 0:
                    continue
                peak_frames = np.asarray(peaks, dtype=np.int64)
                peak_energy = smoothed[np.clip(peak_frames, 0, len(smoothed) - 1)].astype(np.float64)
                order = np.argsort(-peak_energy)
            else:
                keep = np.where(peak_probs >= float(min_peak_probability))[0]
                if len(keep) == 0:
                    continue
                order = keep[np.argsort(-peak_probs[keep])]
            if max_peaks_per_segment > 0:
                order = order[: int(max_peaks_per_segment)]
            for idx in order.tolist():
                local_frame = int(peak_frames[int(idx)])
                global_frame = seg_start + local_frame
                win = _extract_window(imu, global_frame, float(sr), pre_ms, post_ms, target_len)
                if win is None:
                    continue
                out.append(
                    WindowSample(
                        session_id=session_id,
                        source="hard_negative_false_segment",
                        label=0,
                        window=win,
                    )
                )
    return out


def _split_sessions(samples: list[WindowSample], val_ratio: float) -> tuple[list[WindowSample], list[WindowSample], list[str], list[str]]:
    by_session: dict[str, list[WindowSample]] = defaultdict(list)
    for s in samples:
        by_session[s.session_id].append(s)
    sessions = sorted(by_session.keys())
    if len(sessions) <= 1:
        return samples[:], samples[:], sessions, sessions
    n_val = max(1, int(round(len(sessions) * float(val_ratio))))
    val_sessions = sessions[-n_val:]
    train_sessions = sessions[:-n_val]
    if not train_sessions:
        train_sessions = sessions[:-1]
        val_sessions = sessions[-1:]
    train = [s for s in samples if s.session_id in set(train_sessions)]
    val = [s for s in samples if s.session_id in set(val_sessions)]
    return train, val, train_sessions, val_sessions


def _compute_norm_stats(samples: list[WindowSample]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.concatenate([s.window for s in samples], axis=0).astype(np.float32)
    mean = arr.mean(axis=0).astype(np.float32)
    std = np.maximum(arr.std(axis=0).astype(np.float32), 1e-6)
    return mean, std


class WindowDataset(Dataset):
    def __init__(self, samples: list[WindowSample], mean: np.ndarray, std: np.ndarray):
        self.samples = samples
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        x = ((s.window.astype(np.float32) - self.mean[None, :]) / self.std[None, :]).T.astype(np.float32)
        y = np.int64(s.label)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class BinaryWaveformNet(nn.Module):
    def __init__(self, in_ch: int = 6, width: int = 32, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(width, width * 2, 5, padding=2),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(width * 2, width * 4, 5, padding=2),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, width * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _run_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device, class_weight: torch.Tensor | None):
    train = optimizer is not None
    model.train(train)
    losses = []
    logits_all = []
    y_all = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=class_weight)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
        logits_all.append(logits.detach().cpu())
        y_all.append(y.detach().cpu())
    logits_np = torch.cat(logits_all, dim=0).numpy() if logits_all else np.zeros((0, 2), dtype=np.float32)
    y_np = torch.cat(y_all, dim=0).numpy() if y_all else np.zeros((0,), dtype=np.int64)
    prob = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()[:, 1] if len(logits_np) else np.zeros((0,), dtype=np.float32)
    pred = (prob >= 0.5).astype(np.int64)
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_np, pred)) if len(y_np) else 0.0,
        "auc": float(roc_auc_score(y_np, prob)) if len(np.unique(y_np)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_np, prob)) if len(np.unique(y_np)) > 1 else 0.0,
    }
    if len(y_np):
        prec, rec, f1, _ = precision_recall_fscore_support(y_np, pred, average="binary", zero_division=0)
        metrics["precision"] = float(prec)
        metrics["recall"] = float(rec)
        metrics["f1"] = float(f1)
    else:
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0
    return metrics, prob, y_np


def _count_by_source(samples: list[WindowSample]) -> dict:
    out: dict[str, int] = defaultdict(int)
    for s in samples:
        out[s.source] += 1
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser(description="Train a waveform classifier: real password key vs trackpad click peak")
    ap.add_argument("--password_dirs", nargs="+", required=True)
    ap.add_argument("--mixed_dirs", nargs="+", required=True)
    ap.add_argument("--trackpad_click_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pre_ms", type=float, default=100.0)
    ap.add_argument("--post_ms", type=float, default=200.0)
    ap.add_argument("--target_len", type=int, default=57)
    ap.add_argument("--max_trackpad_peaks_per_session", type=int, default=200)
    ap.add_argument("--trackpad_min_rel_energy", type=float, default=0.10)
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--hard_negative_result_json", default="")
    ap.add_argument("--hard_negative_session_dirs", nargs="*", default=[])
    ap.add_argument("--hard_negative_min_segment_confidence", type=float, default=0.70)
    ap.add_argument("--hard_negative_min_peak_probability", type=float, default=0.70)
    ap.add_argument("--hard_negative_max_peaks_per_segment", type=int, default=24)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    pos_samples = _collect_positive_samples(
        password_dirs=args.password_dirs,
        mixed_dirs=args.mixed_dirs,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        target_len=args.target_len,
    )
    neg_samples = _collect_trackpad_click_samples(
        trackpad_dirs=args.trackpad_click_dirs,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        target_len=args.target_len,
        max_peaks_per_session=args.max_trackpad_peaks_per_session,
        min_rel_energy=args.trackpad_min_rel_energy,
    )
    hard_neg_samples = _collect_hard_negative_samples(
        result_json=args.hard_negative_result_json,
        session_dirs=[str(x) for x in args.hard_negative_session_dirs],
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        target_len=args.target_len,
        min_segment_confidence=args.hard_negative_min_segment_confidence,
        min_peak_probability=args.hard_negative_min_peak_probability,
        max_peaks_per_segment=args.hard_negative_max_peaks_per_segment,
    )
    neg_samples.extend(hard_neg_samples)
    if not pos_samples or not neg_samples:
        raise RuntimeError("Positive or negative samples are empty.")

    pos_train, pos_val, pos_train_sessions, pos_val_sessions = _split_sessions(pos_samples, args.val_ratio)
    neg_train, neg_val, neg_train_sessions, neg_val_sessions = _split_sessions(neg_samples, max(args.val_ratio, 0.5 if len({s.session_id for s in neg_samples}) == 2 else args.val_ratio))
    train_samples = pos_train + neg_train
    val_samples = pos_val + neg_val

    mean, std = _compute_norm_stats(train_samples)
    train_ds = WindowDataset(train_samples, mean, std)
    val_ds = WindowDataset(val_samples, mean, std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BinaryWaveformNet(in_ch=6, width=args.width, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    n_pos = sum(s.label == 1 for s in train_samples)
    n_neg = sum(s.label == 0 for s in train_samples)
    class_weight = torch.tensor(
        [
            max(1.0, n_pos / max(n_neg, 1)),
            max(1.0, n_neg / max(n_pos, 1)),
        ],
        dtype=torch.float32,
        device=device,
    )

    history = []
    best = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = _run_epoch(model, train_loader, optimizer, device, class_weight)
        val_metrics, _, _ = _run_epoch(model, val_loader, None, device, class_weight)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        score = float(val_metrics["f1"]) + float(val_metrics["auc"])
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch, "val": val_metrics, "train": train_metrics}
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    assert best is not None and best_state is not None
    torch.save(
        {
            "model_state": best_state,
            "mean": mean,
            "std": std,
            "target_len": int(args.target_len),
            "pre_ms": float(args.pre_ms),
            "post_ms": float(args.post_ms),
        },
        output_dir / "best_key_vs_trackpad_click.pt",
    )

    report = {
        "task": "真实密码按键 与 触控板点击峰 二分类",
        "device": str(device),
        "train_counts": {
            "total": len(train_samples),
            "positive": int(sum(s.label == 1 for s in train_samples)),
            "negative": int(sum(s.label == 0 for s in train_samples)),
            "by_source": _count_by_source(train_samples),
        },
        "val_counts": {
            "total": len(val_samples),
            "positive": int(sum(s.label == 1 for s in val_samples)),
            "negative": int(sum(s.label == 0 for s in val_samples)),
            "by_source": _count_by_source(val_samples),
        },
        "positive_train_sessions": pos_train_sessions,
        "positive_val_sessions": pos_val_sessions,
        "negative_train_sessions": neg_train_sessions,
        "negative_val_sessions": neg_val_sessions,
        "best_epoch": int(best["epoch"]),
        "best_train_metrics": best["train"],
        "best_val_metrics": best["val"],
        "trackpad_peak_collection": {
            "max_trackpad_peaks_per_session": int(args.max_trackpad_peaks_per_session),
            "trackpad_min_rel_energy": float(args.trackpad_min_rel_energy),
        },
        "hard_negative_mining": {
            "enabled": bool(args.hard_negative_result_json),
            "result_json": str(args.hard_negative_result_json),
            "session_dirs": [str(x) for x in args.hard_negative_session_dirs],
            "min_segment_confidence": float(args.hard_negative_min_segment_confidence),
            "min_peak_probability": float(args.hard_negative_min_peak_probability),
            "max_peaks_per_segment": int(args.hard_negative_max_peaks_per_segment),
            "num_hard_negative_samples": int(len(hard_neg_samples)),
        },
    }
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
