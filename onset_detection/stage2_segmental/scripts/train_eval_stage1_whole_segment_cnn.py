#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks, resample
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader
from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, estimate_sample_rate_hz
from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import _load_password_attempt_episodes, _smooth

TARGET_LEN = 256
SEED = 42
POS_PRE_MS_RANGE = (140.0, 360.0)
POS_POST_MS_RANGE = (220.0, 520.0)

rng = random.Random(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class SegmentExample:
    x_seq: np.ndarray
    x_aux: np.ndarray
    y: int
    session_id: str
    source: str
    meta: dict


class SegmentDataset(Dataset):
    def __init__(self, items: List[SegmentExample]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return (
            torch.tensor(it.x_seq, dtype=torch.float32),
            torch.tensor(it.x_aux, dtype=torch.float32),
            torch.tensor(it.y, dtype=torch.long),
        )


class WholeSegmentCNN(nn.Module):
    def __init__(self, in_ch: int, aux_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(96), nn.MaxPool1d(2),
            nn.Conv1d(96, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2),
            nn.Conv1d(128, 160, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(160), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(160 + aux_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(96, 2),
        )

    def forward(self, x_seq: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x_seq).flatten(1)
        h = torch.cat([h, x_aux], dim=1)
        return self.head(h)


def resolve_device(name: str) -> torch.device:
    req = name.lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"
    return torch.device(req)


def _compute_local_activity(imu: np.ndarray, sr: float, short_win_s: float = 0.12) -> np.ndarray:
    win = max(3, int(round(sr * short_win_s)))
    activity = np.zeros(len(imu), dtype=np.float64)
    for ch in range(min(imu.shape[1], 6)):
        col = imu[:, ch].astype(np.float64)
        mean = np.convolve(col, np.ones(win, dtype=np.float64) / float(win), mode="same")
        sq_mean = np.convolve(col ** 2, np.ones(win, dtype=np.float64) / float(win), mode="same")
        activity += np.maximum(sq_mean - mean ** 2, 0.0)
    return activity


def _acf_periodicity_score(
    segment: np.ndarray,
    sr: float,
    lag_range_s: tuple[float, float] = (0.8, 2.5),
) -> tuple[float, float]:
    seg = np.asarray(segment, dtype=np.float64)
    seg = seg - np.mean(seg)
    norm = float(np.sqrt(np.mean(seg ** 2)))
    if norm < 1e-12:
        return 0.0, 0.0
    seg = seg / norm
    acf = np.correlate(seg, seg, mode="full")[len(seg) - 1:] / max(len(seg), 1)
    min_lag = max(1, int(round(sr * lag_range_s[0])))
    max_lag = min(len(acf) - 1, int(round(sr * lag_range_s[1])))
    if max_lag <= min_lag:
        return 0.0, 0.0
    region = acf[min_lag : max_lag + 1]
    peaks, _ = find_peaks(region, prominence=0.02)
    if len(peaks) == 0:
        return 0.0, 0.0
    best_idx = peaks[np.argmax(region[peaks])]
    return float(region[best_idx]), float(best_idx + min_lag) / max(sr, 1.0)


def _iki_regularity(peaks: np.ndarray, sr: float) -> float:
    if len(peaks) < 3:
        return 0.0
    ikis = np.diff(peaks.astype(np.float64)) / max(sr, 1.0)
    mean_iki = float(np.mean(ikis))
    if mean_iki < 1e-6:
        return 0.0
    cv = float(np.std(ikis) / mean_iki)
    return float(np.exp(-cv / 0.35))


def _impulse_discreteness(activity: np.ndarray, peaks: np.ndarray, sr: float, guard_s: float = 0.15) -> float:
    if len(peaks) < 2:
        return 1.0
    peak_vals = activity[peaks]
    guard = max(1, int(round(sr * guard_s)))
    trough_vals = []
    for i in range(len(peaks) - 1):
        lo = int(peaks[i]) + guard
        hi = int(peaks[i + 1]) - guard
        if hi > lo:
            trough_vals.append(float(np.mean(activity[lo:hi])))
    if not trough_vals:
        return 1.0
    trough_mean = float(np.mean(trough_vals))
    if trough_mean < 1e-12:
        return 1000.0
    return float(np.mean(peak_vals)) / trough_mean


def _peak_count_score(n_peaks: int, expected_range: tuple[int, int] = (6, 12)) -> float:
    center = 0.5 * (expected_range[0] + expected_range[1])
    span = max(expected_range[1] - expected_range[0], 1)
    return float(np.exp(-abs(n_peaks - center) / (0.5 * span)))


def _build_segment_inputs(imu: np.ndarray, sr: float, target_len: int = TARGET_LEN) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(imu, dtype=np.float64)
    if len(arr) < 4:
        arr = np.pad(arr, ((0, max(0, 4 - len(arr))), (0, 0)), mode="edge")
    energy = _compute_energy_envelope(arr.astype(np.float32), max(1, int(round(sr * 0.10)))).astype(np.float64)
    energy = _smooth(energy, max(1, int(round(sr * 0.08))))
    activity = _compute_local_activity(arr.astype(np.float32), sr, short_win_s=0.12)
    activity = _smooth(activity, max(1, int(round(sr * 0.06))))
    activity_std = float(np.std(activity))
    peaks, _ = find_peaks(
        activity,
        distance=max(3, int(round(sr * 0.50))),
        prominence=max(1e-12, activity_std * 0.25),
    )
    pulse = np.zeros(len(activity), dtype=np.float64)
    if len(peaks):
        pulse[np.clip(peaks.astype(np.int64), 0, len(pulse) - 1)] = activity[np.clip(peaks.astype(np.int64), 0, len(activity) - 1)]
    pulse = _smooth(pulse, max(1, int(round(sr * 0.12))))
    sig = resample(arr, target_len, axis=0)
    eng = resample(energy[:, None], target_len, axis=0)
    act = resample(activity[:, None], target_len, axis=0)
    pul = resample(pulse[:, None], target_len, axis=0)
    if np.iscomplexobj(sig):
        sig = np.real(sig)
    if np.iscomplexobj(eng):
        eng = np.real(eng)
    if np.iscomplexobj(act):
        act = np.real(act)
    if np.iscomplexobj(pul):
        pul = np.real(pul)
    sig = np.asarray(sig, dtype=np.float32)
    eng = np.asarray(eng, dtype=np.float32)
    act = np.asarray(act, dtype=np.float32)
    pul = np.asarray(pul, dtype=np.float32)
    sig = sig - sig.mean(axis=0, keepdims=True)
    sig = sig / np.maximum(sig.std(axis=0, keepdims=True), 1e-6)
    eng = eng - eng.mean(axis=0, keepdims=True)
    eng = eng / np.maximum(eng.std(axis=0, keepdims=True), 1e-6)
    act = act - act.mean(axis=0, keepdims=True)
    act = act / np.maximum(act.std(axis=0, keepdims=True), 1e-6)
    pul = pul - pul.mean(axis=0, keepdims=True)
    pul = pul / np.maximum(pul.std(axis=0, keepdims=True), 1e-6)
    x = np.concatenate([sig, eng, act, pul], axis=1)

    acf_score, acf_lag_s = _acf_periodicity_score(activity, sr)
    regularity = _iki_regularity(peaks, sr)
    discreteness = _impulse_discreteness(activity, peaks, sr)
    log_disc = float(np.clip(np.log(max(discreteness, 1.0)) / 5.0, 0.0, 1.0))
    count_score = _peak_count_score(len(peaks))
    duration_s = float(len(arr) / max(sr, 1.0))
    peak_density = float(len(peaks)) / max(duration_s, 1e-6)
    energy_p90 = float(np.percentile(energy, 90)) if len(energy) else 0.0
    energy_p50 = float(np.percentile(energy, 50)) if len(energy) else 0.0
    activity_p90 = float(np.percentile(activity, 90)) if len(activity) else 0.0
    activity_p50 = float(np.percentile(activity, 50)) if len(activity) else 0.0
    aux = np.asarray([
        float(acf_score),
        float(np.clip(acf_lag_s / 3.0, 0.0, 1.5)),
        float(regularity),
        float(log_disc),
        float(count_score),
        float(min(len(peaks) / 16.0, 2.0)),
        float(min(peak_density / 8.0, 2.0)),
        float(np.clip(duration_s / 8.0, 0.0, 2.0)),
        float(energy_p90 / max(energy_p50, 1e-6)),
        float(activity_p90 / max(activity_p50, 1e-6)),
    ], dtype=np.float32)
    return x.T.astype(np.float32), aux


def _crop_by_ns(ts: np.ndarray, imu: np.ndarray, start_ns: int, end_ns: int):
    lo = int(np.searchsorted(ts, start_ns, side="left"))
    hi = int(np.searchsorted(ts, end_ns, side="right"))
    lo = max(0, min(lo, len(ts)))
    hi = max(lo, min(hi, len(ts)))
    if hi - lo < 8:
        return None
    return imu[lo:hi], ts[lo:hi], lo, hi


def _load_positive_password_examples(password_dirs: list[str]) -> list[SegmentExample]:
    items = []
    for d in password_dirs:
        for ep in _load_password_attempt_episodes(d):
            x_seq, x_aux = _build_segment_inputs(ep["imu"], ep["sample_rate_hz"])
            items.append(
                SegmentExample(
                    x_seq=x_seq,
                    x_aux=x_aux,
                    y=1,
                    session_id=ep["session_id"],
                    source="password_exact",
                    meta={"episode_id": ep["episode_id"], "password": ep["password"]},
                )
            )
    return items


def _load_complete_mixed_examples(mixed_dirs: list[str], variants_per_episode: int = 2) -> list[SegmentExample]:
    items = []
    for d in mixed_dirs:
        for ep in build_password_episodes(d):
            x_seq, x_aux = _build_segment_inputs(ep.imu, ep.sample_rate_hz)
            items.append(
                SegmentExample(
                    x_seq=x_seq,
                    x_aux=x_aux,
                    y=1,
                    session_id=ep.session_id,
                    source="mixed_exact",
                    meta={"episode_id": ep.episode_id, "password": ep.password},
                )
            )
            loader = SessionLoader(ep.session_path)
            full_ts, full_imu = loader.get_imu()
            if len(full_ts) == 0:
                continue
            start_key_ns = int(ep.key_timestamps_ns[0])
            end_key_ns = int(ep.key_timestamps_ns[-1])
            for _ in range(variants_per_episode):
                pre_ms = rng.uniform(*POS_PRE_MS_RANGE)
                post_ms = rng.uniform(*POS_POST_MS_RANGE)
                crop = _crop_by_ns(
                    full_ts,
                    full_imu,
                    start_key_ns - int(round(pre_ms * 1e6)),
                    end_key_ns + int(round(post_ms * 1e6)),
                )
                if crop is None:
                    continue
                crop_imu, _crop_ts, lo, hi = crop
                x_seq, x_aux = _build_segment_inputs(crop_imu, ep.sample_rate_hz)
                items.append(
                    SegmentExample(
                        x_seq=x_seq,
                        x_aux=x_aux,
                        y=1,
                        session_id=ep.session_id,
                        source="mixed_complete_aug",
                        meta={"episode_id": ep.episode_id, "crop_lo": int(lo), "crop_hi": int(hi)},
                    )
                )
    return items


def _load_activity_negative_examples(mixed_dirs: list[str], max_per_session: int = 10) -> list[SegmentExample]:
    items = []
    for d in mixed_dirs:
        for session_path in sorted(Path(d).glob("*_sensor.csv")):
            prefix = str(session_path)[:-11]
            loader = SessionLoader(prefix)
            ts, imu = loader.get_imu()
            sr = estimate_sample_rate_hz(ts)
            rows = loader.get_activity_log()
            session_id = Path(prefix).name
            cands = []
            for row in rows:
                typing_style = str(row.get("typing_style", ""))
                label = str(row.get("label", ""))
                if typing_style == "password" or label.startswith("typing_2"):
                    continue
                start_ns = int(row.get("start_ns", 0))
                end_ns = int(row.get("end_ns", 0))
                crop = _crop_by_ns(ts, imu, start_ns, end_ns)
                if crop is None:
                    continue
                crop_imu, _crop_ts, lo, hi = crop
                x_seq, x_aux = _build_segment_inputs(crop_imu, sr)
                cands.append(
                    SegmentExample(
                        x_seq=x_seq,
                        x_aux=x_aux,
                        y=0,
                        session_id=session_id,
                        source="activity_negative",
                        meta={"crop_lo": int(lo), "crop_hi": int(hi), "label": label, "typing_style": typing_style},
                    )
                )
            rng.shuffle(cands)
            items.extend(cands[:max_per_session])
    return items


def _load_onset_negative_examples(onset_negative_root: str, target_count: int, duration_s_range=(1.2, 6.0)) -> list[SegmentExample]:
    root = Path(onset_negative_root)
    sensor_files = sorted(root.glob("*/*_sensor.csv"))
    items = []
    for sensor_path in sensor_files:
        prefix = str(sensor_path)[:-11]
        meta_path = Path(prefix + "_meta.json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        rows = []
        with open(sensor_path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append([
                    int(row["timestamp_ns"]),
                    float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"]),
                    float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"]),
                ])
        arr = np.asarray(rows, dtype=np.float64)
        if len(arr) < 20:
            continue
        ts = arr[:, 0].astype(np.int64)
        imu = arr[:, 1:].astype(np.float32)
        sr = estimate_sample_rate_hz(ts)
        total_s = (ts[-1] - ts[0]) * 1e-9
        if total_s < duration_s_range[0]:
            continue
        draws = 1 if total_s < 20 else 2
        for _ in range(draws):
            dur_s = rng.uniform(*duration_s_range)
            span = int(round(dur_s * sr))
            if span >= len(imu):
                continue
            start = rng.randint(0, max(0, len(imu) - span - 1))
            seg_imu = imu[start:start + span]
            x_seq, x_aux = _build_segment_inputs(seg_imu, sr)
            items.append(
                SegmentExample(
                    x_seq=x_seq,
                    x_aux=x_aux,
                    y=0,
                    session_id=str(meta.get("session_id", Path(prefix).name)),
                    source="onset_negative",
                    meta={"activity": meta.get("activity", "negative")},
                )
            )
            if len(items) >= target_count:
                return items
    return items


def _load_same_session_structure_negatives(mixed_dirs: list[str], per_episode: int = 8) -> list[SegmentExample]:
    items = []
    for d in mixed_dirs:
        for ep in build_password_episodes(d):
            loader = SessionLoader(ep.session_path)
            full_ts, full_imu = loader.get_imu()
            sr = estimate_sample_rate_hz(full_ts)
            if len(full_ts) == 0:
                continue
            key_ts = np.asarray(ep.key_timestamps_ns, dtype=np.int64)
            if len(key_ts) < 4:
                continue
            gt_start_ns = int(key_ts[0])
            gt_end_ns = int(key_ts[-1])
            full_span_ns = max(gt_end_ns - gt_start_ns, int(1e8))
            neg_specs = []

            # Missing the left or right part of the password.
            left_cut_idx = min(max(1, len(key_ts) // 4), len(key_ts) - 2)
            right_cut_idx = max(1, len(key_ts) - 1 - max(1, len(key_ts) // 4))
            neg_specs.append((
                int(key_ts[left_cut_idx] - rng.uniform(40e6, 180e6)),
                int(gt_end_ns + rng.uniform(200e6, 520e6)),
                "truncated_left",
            ))
            neg_specs.append((
                int(gt_start_ns - rng.uniform(140e6, 320e6)),
                int(key_ts[right_cut_idx] + rng.uniform(60e6, 220e6)),
                "truncated_right",
            ))

            # Middle-only crop.
            neg_specs.append((
                int(key_ts[1] - rng.uniform(40e6, 160e6)),
                int(key_ts[-2] + rng.uniform(40e6, 200e6)),
                "middle_only",
            ))

            # Too much surrounding context even if all keys are present.
            neg_specs.append((
                int(gt_start_ns - rng.uniform(900e6, 1800e6)),
                int(gt_end_ns + rng.uniform(200e6, 520e6)),
                "overwide_left",
            ))
            neg_specs.append((
                int(gt_start_ns - rng.uniform(120e6, 320e6)),
                int(gt_end_ns + rng.uniform(900e6, 1800e6)),
                "overwide_right",
            ))
            neg_specs.append((
                int(gt_start_ns - rng.uniform(700e6, 1400e6)),
                int(gt_end_ns + rng.uniform(700e6, 1400e6)),
                "overwide_both",
            ))

            # Nearby shifted windows with similar duration but wrong center.
            shift_ns = int(round(full_span_ns * rng.uniform(0.65, 1.10)))
            exact_pre_ns = int(round(rng.uniform(*POS_PRE_MS_RANGE) * 1e6))
            exact_post_ns = int(round(rng.uniform(*POS_POST_MS_RANGE) * 1e6))
            exact_start_ns = gt_start_ns - exact_pre_ns
            exact_end_ns = gt_end_ns + exact_post_ns
            neg_specs.append((exact_start_ns - shift_ns, exact_end_ns - shift_ns, "shifted_left"))
            neg_specs.append((exact_start_ns + shift_ns, exact_end_ns + shift_ns, "shifted_right"))

            cands = []
            for start_ns, end_ns, source in neg_specs:
                crop = _crop_by_ns(full_ts, full_imu, int(start_ns), int(end_ns))
                if crop is None:
                    continue
                crop_imu, _crop_ts, lo, hi = crop
                x_seq, x_aux = _build_segment_inputs(crop_imu, sr)
                cands.append(
                    SegmentExample(
                        x_seq=x_seq,
                        x_aux=x_aux,
                        y=0,
                        session_id=ep.session_id,
                        source=source,
                        meta={"episode_id": ep.episode_id, "crop_lo": int(lo), "crop_hi": int(hi)},
                    )
                )
            rng.shuffle(cands)
            items.extend(cands[:per_episode])
    return items


def _group_split_items(items: List[SegmentExample], train_ratio: float = 0.85):
    groups = {}
    for it in items:
        key = (it.source, it.session_id)
        groups.setdefault(key, []).append(it)
    keys = list(groups.keys())
    rng.shuffle(keys)
    split = int(round(len(keys) * train_ratio))
    split = min(max(split, 1), max(1, len(keys) - 1))
    train_keys = set(keys[:split])
    train_items = []
    val_items = []
    for key, group_items in groups.items():
        (train_items if key in train_keys else val_items).extend(group_items)
    if not val_items:
        val_items = train_items[-min(64, len(train_items)):]
        train_items = train_items[:-len(val_items)] or train_items
    return train_items, val_items


def train_model(train_items: List[SegmentExample], val_items: List[SegmentExample], device: torch.device):
    model = WholeSegmentCNN(
        in_ch=train_items[0].x_seq.shape[0],
        aux_dim=train_items[0].x_aux.shape[0],
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = -1.0
    train_loader = DataLoader(SegmentDataset(train_items), batch_size=64, shuffle=True)
    val_loader = DataLoader(SegmentDataset(val_items), batch_size=128, shuffle=False) if val_items else None

    for _epoch in range(20):
        model.train()
        for x_seq, x_aux, yb in train_loader:
            x_seq = x_seq.to(device)
            x_aux = x_aux.to(device)
            yb = yb.to(device)
            logits = model(x_seq, x_aux)
            loss = F.cross_entropy(logits, yb, weight=torch.tensor([1.0, 1.35], dtype=torch.float32, device=device))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_seq, x_aux, yb in val_loader:
                    x_seq = x_seq.to(device)
                    x_aux = x_aux.to(device)
                    yb = yb.to(device)
                    pred = model(x_seq, x_aux).argmax(dim=1)
                    correct += int((pred == yb).sum())
                    total += int(len(yb))
            acc = correct / max(total, 1)
            if acc > best_val:
                best_val = acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val)


def _cluster_macro_peaks(peaks: np.ndarray, scores: np.ndarray, sample_rate: float, gap_s: float):
    if len(peaks) == 0:
        return []
    max_gap_frames = max(1, int(round(sample_rate * gap_s)))
    groups = []
    cur = [0]
    for i in range(1, len(peaks)):
        if int(peaks[i]) - int(peaks[i - 1]) <= max_gap_frames:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    out = []
    for g in groups:
        idx = np.asarray(g, dtype=np.int64)
        p = peaks[idx]
        s = scores[idx]
        out.append({
            "start_frame": int(p[0]),
            "end_frame": int(p[-1]),
            "num_peaks": int(len(p)),
            "score_mean": float(np.mean(s)),
            "score_sum": float(np.sum(s)),
        })
    return out


def _candidate_iou(a: dict, b: dict) -> float:
    lo = max(int(a["crop_start"]), int(b["crop_start"]))
    hi = min(int(a["crop_end"]), int(b["crop_end"]))
    inter = max(0, hi - lo)
    union = max(int(a["crop_end"]), int(b["crop_end"])) - min(int(a["crop_start"]), int(b["crop_start"]))
    return inter / max(union, 1)


def _pre_score_candidate(candidate: dict, expected_len: float = 9.0) -> float:
    n_peaks = float(candidate.get("macro_num_peaks", candidate.get("cluster_num_peaks", 0)))
    n_parts = float(candidate.get("num_parts", 1))
    gap_s = float(candidate.get("internal_gap_s", 0.0))
    score_mean = float(candidate.get("cluster_score_mean", 0.0))
    len_match = math.exp(-abs(n_peaks - expected_len) / max(2.0, 0.35 * max(expected_len, 1.0)))
    compactness = math.exp(-gap_s / 1.2)
    return 0.55 * len_match + 0.25 * compactness + 0.15 * score_mean + 0.05 * min(n_parts / 3.0, 1.0)


def _dedup_candidates(cands: list[dict]) -> list[dict]:
    ranked = sorted(
        cands,
        key=lambda c: (
            -int(c.get("num_parts", 1)),
            abs(int(c.get("macro_num_peaks", 0)) - 9),
            int(c["crop_end"]) - int(c["crop_start"]),
        ),
    )
    kept = []
    for cand in ranked:
        if any(_candidate_iou(cand, old) >= 0.92 and abs(int(cand.get("macro_num_peaks", 0)) - int(old.get("macro_num_peaks", 0))) <= 1 for old in kept):
            continue
        kept.append(cand)
    kept.sort(key=lambda c: (int(c["crop_start"]), int(c["crop_end"])))
    return kept


def _propose_candidates_fullstream(imu: np.ndarray, sr: float, include_sliding: bool = False) -> list[dict]:
    energy_raw = _compute_energy_envelope(imu, max(1, int(round(sr * 0.10)))).astype(np.float64)
    activity_raw = _compute_local_activity(imu.astype(np.float32), sr, short_win_s=0.12)
    energy_norm = (energy_raw - np.median(energy_raw)) / (np.std(energy_raw) + 1e-6)
    activity_norm = (activity_raw - np.median(activity_raw)) / (np.std(activity_raw) + 1e-6)
    proposal_signal = _smooth(0.45 * energy_norm + 0.55 * activity_norm, max(1, int(round(sr * 0.06))))
    seen = set()
    energy_candidates = []
    for smooth_s in (0.10, 0.15, 0.22, 0.30, 0.40):
        smoothed = _smooth(proposal_signal, max(1, int(round(sr * smooth_s))))
        q50, q90, q98 = np.quantile(smoothed, [0.50, 0.90, 0.98])
        prominence = max(1e-6, (q90 - q50) * 0.08)
        height = q50 + (q98 - q50) * 0.03
        for dist_s in (0.25, 0.35, 0.50, 0.70, 0.90):
            peaks, props = find_peaks(
                smoothed,
                distance=max(1, int(round(sr * dist_s))),
                prominence=prominence,
                height=height,
            )
            if len(peaks) == 0:
                continue
            heights = np.asarray(props.get("peak_heights", smoothed[peaks]), dtype=np.float64)
            peak_scores = heights / max(float(np.max(heights)), 1e-8)
            for gap_s in (0.85, 1.10, 1.35, 1.70, 2.10):
                for cluster in _cluster_macro_peaks(peaks, peak_scores, sr, gap_s=gap_s):
                    if not (3 <= cluster["num_peaks"] <= 18):
                        continue
                    pad_frames = int(round(sr * 0.80))
                    lo = max(0, int(cluster["start_frame"]) - pad_frames)
                    hi = min(len(imu), int(cluster["end_frame"]) + pad_frames + 1)
                    key = (lo, hi, cluster["num_peaks"])
                    if key in seen:
                        continue
                    seen.add(key)
                    energy_candidates.append({
                        "crop_start": int(lo),
                        "crop_end": int(hi),
                        "macro_num_peaks": int(cluster["num_peaks"]),
                        "cluster_score_mean": float(cluster["score_mean"]),
                        "internal_gap_s": float(gap_s),
                        "num_parts": 1,
                        "source": "energy_cluster",
                    })
    ordered = sorted(energy_candidates, key=lambda c: (int(c["crop_start"]), int(c["crop_end"])))
    union_seen = set()
    for i in range(len(ordered)):
        lo = int(ordered[i]["crop_start"])
        hi = int(ordered[i]["crop_end"])
        peak_sum = int(ordered[i].get("macro_num_peaks", 0))
        score_means = [float(ordered[i].get("cluster_score_mean", 0.0))]
        for j in range(i + 1, min(len(ordered), i + 3)):
            nxt = ordered[j]
            gap = int(nxt["crop_start"]) - hi
            if gap > int(round(sr * 2.2)):
                break
            lo = min(lo, int(nxt["crop_start"]))
            hi = max(hi, int(nxt["crop_end"]))
            peak_sum += int(nxt.get("macro_num_peaks", 0))
            score_means.append(float(nxt.get("cluster_score_mean", 0.0)))
            key = (lo, hi, j - i + 1, peak_sum)
            if key in union_seen:
                continue
            union_seen.add(key)
            energy_candidates.append({
                "crop_start": int(lo),
                "crop_end": int(hi),
                "macro_num_peaks": int(peak_sum),
                "cluster_score_mean": float(np.mean(score_means)),
                "internal_gap_s": float(max(0, gap) / max(sr, 1e-6)),
                "num_parts": int(j - i + 1),
                "source": "energy_union",
            })

    sliding_candidates = []
    if include_sliding:
        # Optional rescue path: disabled by default because naive sliding-window
        # candidates can dominate the ranker before we have a robust session-wise
        # ranking head for them.
        for span_s in (1.8, 2.3, 2.8, 3.3, 3.8, 4.4):
            span = int(round(sr * span_s))
            if span < 16 or span >= len(imu):
                continue
            stride = max(1, int(round(span * 0.35)))
            for lo in range(0, max(1, len(imu) - span + 1), stride):
                hi = min(len(imu), lo + span)
                if hi - lo < 16:
                    continue
                local_signal = proposal_signal[lo:hi]
                if len(local_signal) < 8:
                    continue
                q50, q90, q98 = np.quantile(local_signal, [0.50, 0.90, 0.98])
                prominence = max(1e-6, (q90 - q50) * 0.08)
                height = q50 + (q98 - q50) * 0.03
                peaks, props = find_peaks(
                    local_signal,
                    distance=max(1, int(round(sr * 0.28))),
                    prominence=prominence,
                    height=height,
                )
                peak_vals = np.asarray(props.get("peak_heights", local_signal[peaks] if len(peaks) else []), dtype=np.float64)
                score_mean = float(np.mean(peak_vals) / max(float(np.max(local_signal)), 1e-8)) if len(peak_vals) else 0.0
                sliding_candidates.append({
                    "crop_start": int(lo),
                    "crop_end": int(hi),
                    "macro_num_peaks": int(len(peaks)),
                    "cluster_score_mean": float(score_mean),
                    "internal_gap_s": 0.0,
                    "num_parts": 1,
                    "source": "sliding_window",
                })

    energy_candidates = _dedup_candidates(energy_candidates)
    sliding_candidates = _dedup_candidates(sliding_candidates)
    energy_ranked = sorted(energy_candidates, key=lambda c: _pre_score_candidate(c), reverse=True)
    sliding_ranked = sorted(sliding_candidates, key=lambda c: _pre_score_candidate(c), reverse=True)

    selected = list(energy_ranked[:80])
    if include_sliding:
        selected = list(energy_ranked[:56])
        for cand in sliding_ranked:
            if any(_candidate_iou(cand, old) >= 0.60 for old in selected):
                continue
            selected.append(cand)
            if len(selected) >= 80:
                break
        if len(selected) < 80:
            spill = energy_ranked[56:] + sliding_ranked
            for cand in spill:
                if any(
                    int(cand["crop_start"]) == int(old["crop_start"]) and
                    int(cand["crop_end"]) == int(old["crop_end"]) and
                    int(cand.get("macro_num_peaks", 0)) == int(old.get("macro_num_peaks", 0))
                    for old in selected
                ):
                    continue
                selected.append(cand)
                if len(selected) >= 80:
                    break
    return selected[:80]


def _score_candidates(model, candidates: list[dict], imu: np.ndarray, sr: float, device: torch.device) -> list[float]:
    if not candidates:
        return []
    xs = []
    auxs = []
    for cand in candidates:
        crop = imu[int(cand["crop_start"]):int(cand["crop_end"])]
        x_seq, x_aux = _build_segment_inputs(crop, sr)
        xs.append(x_seq)
        auxs.append(x_aux)
    xb = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    xa = torch.tensor(np.stack(auxs), dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(xb, xa), dim=1)[:, 1].detach().cpu().numpy()
    return probs.tolist()


def _candidate_quality(candidate: dict, ep, full_ts: np.ndarray) -> dict:
    start_idx = min(max(int(candidate["crop_start"]), 0), len(full_ts) - 1)
    end_idx = min(max(int(candidate["crop_end"]) - 1, 0), len(full_ts) - 1)
    cand_start_ns = int(full_ts[start_idx])
    cand_end_ns = int(full_ts[end_idx])
    gt_ts = np.asarray(ep.key_timestamps_ns, dtype=np.int64)
    gt_start_ns = int(gt_ts[0])
    gt_end_ns = int(gt_ts[-1])
    inter = max(0, min(cand_end_ns, gt_end_ns) - max(cand_start_ns, gt_start_ns))
    union = max(cand_end_ns, gt_end_ns) - min(cand_start_ns, gt_start_ns)
    iou = inter / max(union, 1)
    inside = int(np.sum((gt_ts >= cand_start_ns) & (gt_ts <= cand_end_ns)))
    key_frac = inside / max(len(gt_ts), 1)
    gt_span_s = max((gt_end_ns - gt_start_ns) * 1e-9, 1e-3)
    miss_left_s = max(0.0, (cand_start_ns - gt_start_ns) * 1e-9)
    miss_right_s = max(0.0, (gt_end_ns - cand_end_ns) * 1e-9)
    over_left_s = max(0.0, (gt_start_ns - cand_start_ns) * 1e-9)
    over_right_s = max(0.0, (cand_end_ns - gt_end_ns) * 1e-9)
    miss_penalty = (miss_left_s + miss_right_s) / gt_span_s
    over_penalty = (over_left_s + over_right_s) / gt_span_s
    boundary = math.exp(-0.45 * over_penalty) * math.exp(-1.30 * miss_penalty)
    quality = 0.55 * key_frac + 0.25 * boundary + 0.20 * iou
    complete_hit = bool(key_frac >= 0.999 and miss_penalty <= 0.10 and over_penalty <= 1.5)
    return {
        "iou": float(iou),
        "key_frac": float(key_frac),
        "boundary_score": float(boundary),
        "miss_penalty": float(miss_penalty),
        "over_penalty": float(over_penalty),
        "quality": float(quality),
        "complete_hit": complete_hit,
    }


def _aggregate_stage1(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_episodes": 0,
            "top1_quality": 0.0,
            "top1_key_frac": 0.0,
            "top1_iou": 0.0,
            "top1_complete_hit": 0.0,
            "top3_has_complete": 0.0,
            "oracle_quality": 0.0,
            "oracle_complete_hit": 0.0,
        }
    return {
        "num_episodes": int(len(rows)),
        "top1_quality": float(np.mean([r["top1_quality"] for r in rows])),
        "top1_key_frac": float(np.mean([r["top1_key_frac"] for r in rows])),
        "top1_iou": float(np.mean([r["top1_iou"] for r in rows])),
        "top1_complete_hit": float(np.mean([r["top1_complete_hit"] for r in rows])),
        "top3_has_complete": float(np.mean([r["top3_has_complete"] for r in rows])),
        "oracle_quality": float(np.mean([r["oracle_quality"] for r in rows])),
        "oracle_complete_hit": float(np.mean([r["oracle_complete_hit"] for r in rows])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--password_dirs", nargs="+", required=True)
    ap.add_argument("--train_mixed_dirs", nargs="+", required=True)
    ap.add_argument("--eval_dirs", nargs="+", required=True)
    ap.add_argument("--onset_negative_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    pos_password = _load_positive_password_examples(args.password_dirs)
    pos_mixed = _load_complete_mixed_examples(args.train_mixed_dirs, variants_per_episode=2)
    neg_activity = _load_activity_negative_examples(args.train_mixed_dirs, max_per_session=10)
    neg_struct = _load_same_session_structure_negatives(args.train_mixed_dirs, per_episode=8)
    target_neg = max(len(pos_password) + len(pos_mixed), len(neg_activity) + len(neg_struct))
    neg_onset = _load_onset_negative_examples(args.onset_negative_root, target_count=max(40, target_neg // 2))

    all_items = pos_password + pos_mixed + neg_activity + neg_struct + neg_onset
    rng.shuffle(all_items)
    train_items, val_items = _group_split_items(all_items, train_ratio=0.85)
    model, best_val = train_model(train_items, val_items, device)
    torch.save({"model_state": model.state_dict(), "target_len": TARGET_LEN}, out_dir / "stage1_whole_segment_cnn.pt")

    eval_eps = []
    for d in args.eval_dirs:
        eval_eps.extend(build_password_episodes(d))
    by_eval = {}
    for ep in eval_eps:
        by_eval.setdefault(ep.session_id, []).append(ep)

    summary_rows = []
    debug_rows = []
    for session_id, session_eps in sorted(by_eval.items()):
        loader = SessionLoader(session_eps[0].session_path)
        ts, imu = loader.get_imu()
        sr = estimate_sample_rate_hz(ts)
        candidates = _propose_candidates_fullstream(imu, sr)
        scores = _score_candidates(model, candidates, imu, sr, device)
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        if not scored:
            for ep in session_eps:
                debug_rows.append({"session_id": session_id, "episode_id": ep.episode_id, "error": "no_candidates"})
            continue

        for ep in session_eps:
            cand_debug = []
            for rank, (score, cand) in enumerate(scored[:10], start=1):
                qual = _candidate_quality(cand, ep, ts)
                cand_debug.append({
                    "rank": int(rank),
                    "segment_score": float(score),
                    "crop_start": int(cand["crop_start"]),
                    "crop_end": int(cand["crop_end"]),
                    "macro_num_peaks": int(cand.get("macro_num_peaks", 0)),
                    "source": cand.get("source", ""),
                    **qual,
                })
            top1 = cand_debug[0]
            top3_has_complete = any(row["complete_hit"] for row in cand_debug[:3])
            oracle = max(cand_debug, key=lambda row: row["quality"])
            oracle_complete = max((1 if row["complete_hit"] else 0) for row in cand_debug)
            summary_rows.append({
                "session_id": session_id,
                "episode_id": ep.episode_id,
                "top1_quality": float(top1["quality"]),
                "top1_key_frac": float(top1["key_frac"]),
                "top1_iou": float(top1["iou"]),
                "top1_complete_hit": float(1 if top1["complete_hit"] else 0),
                "top3_has_complete": float(1 if top3_has_complete else 0),
                "oracle_quality": float(oracle["quality"]),
                "oracle_complete_hit": float(oracle_complete),
            })
            debug_rows.append({
                "session_id": session_id,
                "episode_id": ep.episode_id,
                "selected_candidate": top1,
                "oracle_candidate": oracle,
                "top_candidates": cand_debug[:5],
            })

    report = {
        "mode": "stage1_whole_segment_cnn",
        "train_summary": {
            "num_pos_password": int(len(pos_password)),
            "num_pos_mixed": int(len(pos_mixed)),
            "num_neg_activity": int(len(neg_activity)),
            "num_neg_struct": int(len(neg_struct)),
            "num_neg_onset": int(len(neg_onset)),
            "num_train_items": int(len(train_items)),
            "num_val_items": int(len(val_items)),
            "best_val_acc": float(best_val),
        },
        "stage1_eval": _aggregate_stage1(summary_rows),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "debug_rows.json").write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
