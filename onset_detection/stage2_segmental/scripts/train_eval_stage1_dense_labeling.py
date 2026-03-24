#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader, discover_sessions
from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, estimate_sample_rate_hz
from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import _smooth
from phase3_password_inception.run_password_closure_inception import supported_key

SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class SegmentGT:
    episode_id: str
    password: str
    start_frame: int
    end_frame: int
    key_frames: np.ndarray
    key_timestamps_ns: np.ndarray


@dataclass
class SessionRecord:
    session_id: str
    session_path: str
    sample_rate_hz: float
    timestamps_ns: np.ndarray
    features: np.ndarray  # [C, T]
    labels: np.ndarray    # [T]
    gt_segments: list[SegmentGT]


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = ConvBlock1D(in_ch, out_ch, kernel_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.conv = ConvBlock1D(in_ch, out_ch, kernel_size, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diff = skip.shape[2] - x.shape[2]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            x = x[:, :, : skip.shape[2]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_filters: int = 24,
        depth: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        use_attention: bool = False,
    ):
        super().__init__()
        self.depth = depth
        self.use_attention = use_attention

        self.enc_first = ConvBlock1D(in_channels, base_filters, kernel_size, dropout)
        self.encoders = nn.ModuleList()
        ch = base_filters
        for _ in range(depth):
            out_ch = ch * 2
            self.encoders.append(DownBlock(ch, out_ch, kernel_size, dropout))
            ch = out_ch

        if use_attention:
            self.bottleneck_attn = nn.MultiheadAttention(
                embed_dim=ch,
                num_heads=max(1, ch // 64),
                dropout=dropout,
                batch_first=True,
            )
            self.bottleneck_norm = nn.LayerNorm(ch)
        self.bottleneck_conv = ConvBlock1D(ch, ch, kernel_size, dropout)

        self.decoders = nn.ModuleList()
        for _ in range(depth):
            in_ch = ch + ch // 2
            out_ch = ch // 2
            self.decoders.append(UpBlock(in_ch, out_ch, kernel_size, dropout))
            ch = out_ch

        self.head = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        x = self.enc_first(x)
        skips.append(x)
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
        if self.use_attention:
            x_t = x.permute(0, 2, 1)
            attn_out, _ = self.bottleneck_attn(x_t, x_t, x_t)
            x = self.bottleneck_norm(x_t + attn_out).permute(0, 2, 1)
        x = self.bottleneck_conv(x)
        skips = skips[:-1]
        for i, dec in enumerate(self.decoders):
            x = dec(x, skips[-(i + 1)])
        return self.head(x)


class DenseSegmentationLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 8.0,
        boundary_width: int = 80,
        boundary_boost: float = 4.0,
        smooth_weight: float = 0.03,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.boundary_width = int(boundary_width)
        self.boundary_boost = float(boundary_boost)
        self.smooth_weight = float(smooth_weight)
        self.dice_weight = float(dice_weight)

    def _boundary_mask(self, target: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, _, t = target.shape
        out = torch.ones_like(target)
        diff = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        for b in range(bsz):
            trans = torch.nonzero(diff[b, 0] > 0.5).squeeze(-1)
            for idx in trans:
                idx_i = int(idx.item())
                lo = max(0, idx_i - self.boundary_width)
                hi = min(t, idx_i + self.boundary_width + 1)
                out[b, 0, lo:hi] = self.boundary_boost
        if valid_mask is not None:
            out = out * valid_mask
        return out

    def _masked_mean(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x.mean()
        denom = torch.clamp(mask.sum(), min=1.0)
        return (x * mask).sum() / denom

    def _dice_loss(self, logits: torch.Tensor, target: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        if valid_mask is not None:
            pred = pred * valid_mask
            target = target * valid_mask
        smooth = 1.0
        inter = (pred * target).sum(dim=-1)
        union = pred.sum(dim=-1) + target.sum(dim=-1)
        dice = (2.0 * inter + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        pw = torch.tensor([self.pos_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pw,
            reduction="none",
        )
        weights = self._boundary_mask(target, valid_mask)
        bce = self._masked_mean(bce * weights, valid_mask)
        dice = self._dice_loss(logits, target, valid_mask)
        pred = torch.sigmoid(logits)
        diff = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
        smooth_mask = None if valid_mask is None else valid_mask[:, :, 1:] * valid_mask[:, :, :-1]
        smooth = self._masked_mean(diff, smooth_mask)
        total = bce + self.dice_weight * dice + self.smooth_weight * smooth
        return total, {
            "bce": float(bce.item()),
            "dice": float(dice.item()),
            "smooth": float(smooth.item()),
            "total": float(total.item()),
        }


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


def _zscore_channels(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = np.maximum(x.std(axis=0, keepdims=True), 1e-6)
    return ((x - mean) / std).astype(np.float32)


def _compute_local_activity(imu: np.ndarray, sr: float, short_win_s: float = 0.12) -> np.ndarray:
    win = max(3, int(round(sr * short_win_s)))
    activity = np.zeros(len(imu), dtype=np.float64)
    for ch in range(min(imu.shape[1], 6)):
        col = imu[:, ch].astype(np.float64)
        mean = np.convolve(col, np.ones(win, dtype=np.float64) / float(win), mode="same")
        sq_mean = np.convolve(col ** 2, np.ones(win, dtype=np.float64) / float(win), mode="same")
        activity += np.maximum(sq_mean - mean ** 2, 0.0)
    return activity


def _build_dense_features(imu: np.ndarray, sr: float, feature_mode: str) -> np.ndarray:
    raw = _zscore_channels(imu)
    if feature_mode == "raw6":
        return raw.T.astype(np.float32)

    energy = _compute_energy_envelope(imu.astype(np.float32), max(1, int(round(sr * 0.10)))).astype(np.float64)
    energy = _smooth(energy, max(1, int(round(sr * 0.08))))
    activity = _compute_local_activity(imu.astype(np.float32), sr, short_win_s=0.12)
    activity = _smooth(activity, max(1, int(round(sr * 0.06))))
    if feature_mode == "raw6_energy_activity":
        derived = np.stack([
            _zscore_channels(energy[:, None])[:, 0],
            _zscore_channels(activity[:, None])[:, 0],
        ], axis=1).astype(np.float32)
        feat = np.concatenate([raw, derived], axis=1)
        return feat.T.astype(np.float32)

    act_std = float(np.std(activity))
    peaks, _ = find_peaks(
        activity,
        distance=max(3, int(round(sr * 0.50))),
        prominence=max(1e-12, act_std * 0.25),
    )
    pulse = np.zeros(len(activity), dtype=np.float64)
    if len(peaks):
        idx = np.clip(peaks.astype(np.int64), 0, len(activity) - 1)
        pulse[idx] = activity[idx]
    pulse = _smooth(pulse, max(1, int(round(sr * 0.12))))

    derived = np.stack([
        _zscore_channels(energy[:, None])[:, 0],
        _zscore_channels(activity[:, None])[:, 0],
        _zscore_channels(pulse[:, None])[:, 0],
    ], axis=1).astype(np.float32)
    feat = np.concatenate([raw, derived], axis=1)
    return feat.T.astype(np.float32)


def _group_episodes_by_session(roots: list[str], min_password_len: int) -> dict[str, list]:
    grouped: dict[str, list] = {}
    for root in roots:
        for ep in build_password_episodes(root, min_len=min_password_len):
            grouped.setdefault(str(ep.session_path), []).append(ep)
    return grouped


def _build_session_records(
    roots: list[str],
    pre_pad_ms: float,
    post_pad_ms: float,
    feature_mode: str,
    min_password_len: int,
) -> list[SessionRecord]:
    grouped = _group_episodes_by_session(roots, min_password_len=min_password_len)
    records: list[SessionRecord] = []
    pre_pad_ns = int(round(pre_pad_ms * 1e6))
    post_pad_ns = int(round(post_pad_ms * 1e6))
    session_paths = sorted(grouped.keys())
    for session_path in session_paths:
        eps = sorted(grouped[session_path], key=lambda ep: int(ep.key_timestamps_ns[0]))
        loader = SessionLoader(session_path)
        ts_ns, imu = loader.get_imu()
        if len(ts_ns) < 8:
            continue
        sr = estimate_sample_rate_hz(ts_ns)
        labels = np.zeros(len(ts_ns), dtype=np.float32)
        gt_segments: list[SegmentGT] = []
        for ep in eps:
            start_ns = int(ep.key_timestamps_ns[0]) - pre_pad_ns
            end_ns = int(ep.key_timestamps_ns[-1]) + post_pad_ns
            start_frame = int(np.searchsorted(ts_ns, start_ns, side="left"))
            end_frame = int(np.searchsorted(ts_ns, end_ns, side="right"))
            start_frame = max(0, min(start_frame, len(ts_ns) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(ts_ns)))
            labels[start_frame:end_frame] = 1.0
            key_frames = np.searchsorted(ts_ns, np.asarray(ep.key_timestamps_ns, dtype=np.int64), side="left")
            key_frames = np.clip(key_frames, 0, len(ts_ns) - 1).astype(np.int64)
            gt_segments.append(
                SegmentGT(
                    episode_id=ep.episode_id,
                    password=ep.password,
                    start_frame=int(start_frame),
                    end_frame=int(end_frame),
                    key_frames=key_frames,
                    key_timestamps_ns=np.asarray(ep.key_timestamps_ns, dtype=np.int64),
                )
            )
        features = _build_dense_features(imu, sr, feature_mode=feature_mode)
        records.append(
            SessionRecord(
                session_id=Path(session_path).name,
                session_path=str(session_path),
                sample_rate_hz=float(sr),
                timestamps_ns=np.asarray(ts_ns, dtype=np.int64),
                features=features,
                labels=labels,
                gt_segments=gt_segments,
            )
        )
    return records


def _build_password_attempt_records(
    roots: list[str],
    pre_pad_ms: float,
    post_pad_ms: float,
    feature_mode: str,
    min_password_len: int,
    context_pre_ms: float,
    context_post_ms: float,
    require_match: bool = True,
) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    pre_pad_ns = int(round(pre_pad_ms * 1e6))
    post_pad_ns = int(round(post_pad_ms * 1e6))
    context_pre_ns = int(round(context_pre_ms * 1e6))
    context_post_ns = int(round(context_post_ms * 1e6))

    for root in roots:
        for session_path in sorted(set(discover_sessions(root))):
            loader = SessionLoader(session_path)
            ts_ns, imu = loader.get_imu()
            if len(ts_ns) < 8:
                continue
            sr = estimate_sample_rate_hz(ts_ns)
            all_events = loader.get_press_events()
            attempts = loader.get_attempts()
            session_id = Path(session_path).name
            for attempt_index, att in enumerate(attempts):
                if require_match and str(att.get("typed", "")) != str(att.get("prompt", "")):
                    continue
                start_ns = int(att.get("start_ns", 0))
                end_ns = int(att.get("end_ns", start_ns))
                keys = []
                for e in all_events:
                    if not (start_ns <= int(e["ts"]) <= end_ns):
                        continue
                    key = str(e["key"]).lower().strip()
                    if key == "enter" or not supported_key(key):
                        continue
                    keys.append({"ts": int(e["ts"]), "key": key})
                if len(keys) < min_password_len:
                    continue

                key_ts = np.asarray([k["ts"] for k in keys], dtype=np.int64)
                rec_start_ns = int(max(ts_ns[0], key_ts[0] - context_pre_ns))
                rec_end_ns = int(min(ts_ns[-1], key_ts[-1] + context_post_ns))
                lo = int(np.searchsorted(ts_ns, rec_start_ns, side="left"))
                hi = int(np.searchsorted(ts_ns, rec_end_ns, side="right"))
                if hi - lo < 8:
                    continue
                rec_ts = np.asarray(ts_ns[lo:hi], dtype=np.int64)
                rec_imu = np.asarray(imu[lo:hi], dtype=np.float32)
                labels = np.zeros(len(rec_ts), dtype=np.float32)
                seg_start_ns = int(key_ts[0]) - pre_pad_ns
                seg_end_ns = int(key_ts[-1]) + post_pad_ns
                seg_lo = int(np.searchsorted(rec_ts, seg_start_ns, side="left"))
                seg_hi = int(np.searchsorted(rec_ts, seg_end_ns, side="right"))
                seg_lo = max(0, min(seg_lo, len(rec_ts) - 1))
                seg_hi = max(seg_lo + 1, min(seg_hi, len(rec_ts)))
                labels[seg_lo:seg_hi] = 1.0
                key_frames = np.searchsorted(rec_ts, key_ts, side="left")
                key_frames = np.clip(key_frames, 0, len(rec_ts) - 1).astype(np.int64)
                password = "".join(str(k["key"]) for k in keys)
                features = _build_dense_features(rec_imu, sr, feature_mode=feature_mode)
                episode_id = f"{session_id}::attempt{attempt_index:02d}"
                records.append(
                    SessionRecord(
                        session_id=episode_id,
                        session_path=str(session_path),
                        sample_rate_hz=float(sr),
                        timestamps_ns=rec_ts,
                        features=features,
                        labels=labels,
                        gt_segments=[
                            SegmentGT(
                                episode_id=episode_id,
                                password=password,
                                start_frame=int(seg_lo),
                                end_frame=int(seg_hi),
                                key_frames=key_frames,
                                key_timestamps_ns=key_ts,
                            )
                        ],
                    )
                )
    return records


def _build_onset_negative_records(
    root: str,
    feature_mode: str,
    target_count: int,
    duration_s_range: tuple[float, float] = (1.2, 6.0),
) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    if not root:
        return records
    session_paths = sorted(set(discover_sessions(root)))
    target_per_session = max(1, int(np.ceil(target_count / max(len(session_paths), 1)))) if target_count > 0 else 8
    for session_path in session_paths:
        loader = SessionLoader(session_path)
        ts_ns, imu = loader.get_imu()
        if len(ts_ns) < 16:
            continue
        sr = estimate_sample_rate_hz(ts_ns)
        total_s = float((ts_ns[-1] - ts_ns[0]) * 1e-9)
        if total_s < duration_s_range[0]:
            continue
        draws = max(2, min(24, target_per_session, int(total_s // max(duration_s_range[0], 1.0))))
        for draw_idx in range(draws):
            dur_s = rng.uniform(*duration_s_range)
            span = int(round(dur_s * sr))
            if span >= len(imu) or span < 16:
                continue
            start = rng.randint(0, max(0, len(imu) - span - 1))
            end = start + span
            rec_ts = np.asarray(ts_ns[start:end], dtype=np.int64)
            rec_imu = np.asarray(imu[start:end], dtype=np.float32)
            features = _build_dense_features(rec_imu, sr, feature_mode=feature_mode)
            session_id = f"{Path(session_path).name}::neg{draw_idx:02d}"
            records.append(
                SessionRecord(
                    session_id=session_id,
                    session_path=str(session_path),
                    sample_rate_hz=float(sr),
                    timestamps_ns=rec_ts,
                    features=features,
                    labels=np.zeros(len(rec_ts), dtype=np.float32),
                    gt_segments=[],
                )
            )
            if target_count > 0 and len(records) >= target_count:
                return records
    return records


class FullStreamDataset(Dataset):
    def __init__(
        self,
        records: list[SessionRecord],
        train_max_len: Optional[int],
        augment: bool,
        positive_crop_prob: float = 0.8,
        label_jitter_frames: int = 0,
        time_stretch_prob: float = 0.0,
        time_stretch_min: float = 1.0,
        time_stretch_max: float = 1.0,
    ):
        self.records = records
        self.train_max_len = train_max_len
        self.augment = augment
        self.positive_crop_prob = float(positive_crop_prob)
        self.label_jitter_frames = int(max(0, label_jitter_frames))
        self.time_stretch_prob = float(max(0.0, time_stretch_prob))
        self.time_stretch_min = float(min(time_stretch_min, time_stretch_max))
        self.time_stretch_max = float(max(time_stretch_min, time_stretch_max))

    def __len__(self) -> int:
        return len(self.records)

    def _jitter_labels(self, y: np.ndarray) -> np.ndarray:
        if self.label_jitter_frames <= 0:
            return y
        y_bin = (np.asarray(y, dtype=np.float32) > 0.5).astype(np.float32)
        out = np.zeros_like(y_bin)
        in_seg = False
        start = 0
        for i, val in enumerate(y_bin):
            if val > 0.5 and not in_seg:
                in_seg = True
                start = i
            elif val <= 0.5 and in_seg:
                end = i
                left = rng.randint(-self.label_jitter_frames, self.label_jitter_frames)
                right = rng.randint(-self.label_jitter_frames, self.label_jitter_frames)
                lo = max(0, start + left)
                hi = min(len(y_bin), end + right)
                if hi <= lo:
                    lo, hi = start, end
                out[lo:hi] = 1.0
                in_seg = False
        if in_seg:
            end = len(y_bin)
            left = rng.randint(-self.label_jitter_frames, self.label_jitter_frames)
            right = rng.randint(-self.label_jitter_frames, self.label_jitter_frames)
            lo = max(0, start + left)
            hi = min(len(y_bin), end + right)
            if hi <= lo:
                lo, hi = start, end
            out[lo:hi] = 1.0
        return out.astype(np.float32)

    def _time_stretch(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.time_stretch_prob <= 0.0 or rng.random() >= self.time_stretch_prob:
            return x, y
        scale = rng.uniform(self.time_stretch_min, self.time_stretch_max)
        if abs(scale - 1.0) < 1e-3:
            return x, y
        src_len = x.shape[1]
        dst_len = max(16, int(round(src_len * scale)))
        src_grid = np.linspace(0.0, 1.0, num=src_len, dtype=np.float64)
        dst_grid = np.linspace(0.0, 1.0, num=dst_len, dtype=np.float64)
        x_out = np.stack([
            np.interp(dst_grid, src_grid, ch.astype(np.float64)).astype(np.float32)
            for ch in x
        ], axis=0)
        y_out = np.interp(dst_grid, src_grid, y.astype(np.float64)).astype(np.float32)
        y_out = (y_out >= 0.5).astype(np.float32)
        return x_out, y_out

    def _choose_crop(self, rec: SessionRecord) -> tuple[int, int]:
        t = rec.features.shape[1]
        max_len = self.train_max_len
        if max_len is None or t <= max_len:
            return 0, t
        if rec.gt_segments and rng.random() < self.positive_crop_prob:
            gt = rng.choice(rec.gt_segments)
            lo_min = max(0, gt.end_frame - max_len)
            lo_max = min(gt.start_frame, t - max_len)
            if lo_max < lo_min:
                lo = max(0, min(gt.start_frame, t - max_len))
            else:
                lo = rng.randint(lo_min, lo_max)
        else:
            lo = rng.randint(0, t - max_len)
        return int(lo), int(lo + max_len)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        lo, hi = self._choose_crop(rec)
        x = rec.features[:, lo:hi].copy()
        y = rec.labels[lo:hi].copy()
        valid_len = x.shape[1]

        if self.augment:
            x, y = self._time_stretch(x, y)
            y = self._jitter_labels(y)
            if rng.random() < 0.5:
                noise = np.random.randn(*x.shape).astype(np.float32) * 0.02
                x = x + noise
            if rng.random() < 0.2:
                ch = rng.randrange(x.shape[0])
                x[ch] = 0.0

        valid_len = x.shape[1]
        if self.train_max_len is not None:
            if valid_len > self.train_max_len:
                lo = rng.randint(0, valid_len - self.train_max_len)
                hi = lo + self.train_max_len
                x = x[:, lo:hi]
                y = y[lo:hi]
                valid_len = x.shape[1]
            elif valid_len < self.train_max_len:
                pad = self.train_max_len - valid_len
                x = np.pad(x, ((0, 0), (0, pad)), mode="constant")
                y = np.pad(y, (0, pad), mode="constant")
        mask = np.zeros(len(y), dtype=np.float32)
        mask[:valid_len] = 1.0
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float().unsqueeze(0),
            torch.from_numpy(mask).float().unsqueeze(0),
        )


def extract_segments(
    probs: np.ndarray,
    threshold: float,
    min_length: int,
    merge_gap: int,
    prob_smooth_window: int = 1,
    valley_merge_threshold: float = 0.0,
    valley_merge_max_gap: int = 0,
) -> list[tuple[int, int, float]]:
    probs = np.asarray(probs, dtype=np.float64)
    raw_probs = probs.copy()
    if prob_smooth_window > 1 and len(probs) > prob_smooth_window:
        kernel = np.ones(int(prob_smooth_window), dtype=np.float64) / float(prob_smooth_window)
        smoothed = np.convolve(probs, kernel, mode="same")
        half_w = int(prob_smooth_window) // 2
        smoothed[:half_w] = probs[:half_w]
        smoothed[-half_w:] = probs[-half_w:]
        probs = smoothed

    binary = (probs > threshold).astype(np.int64)
    segs: list[tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, val in enumerate(binary):
        if val and not in_seg:
            in_seg = True
            start = i
        elif not val and in_seg:
            segs.append((start, i))
            in_seg = False
    if in_seg:
        segs.append((start, len(binary)))

    merged: list[tuple[int, int]] = []
    for lo, hi in segs:
        if merged and lo - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))

    if valley_merge_threshold > 0.0 and valley_merge_max_gap > 0 and len(merged) > 1:
        further_merged = [merged[0]]
        for seg in merged[1:]:
            prev = further_merged[-1]
            gap = seg[0] - prev[1]
            if 0 < gap <= valley_merge_max_gap:
                valley_region = raw_probs[prev[1]:seg[0]]
                if len(valley_region) > 0:
                    valley_mean = float(np.mean(valley_region))
                    if valley_mean > valley_merge_threshold:
                        further_merged[-1] = (prev[0], seg[1])
                        continue
            further_merged.append(seg)
        merged = further_merged

    out = []
    for lo, hi in merged:
        if hi - lo >= min_length:
            out.append((int(lo), int(hi), float(np.mean(raw_probs[lo:hi]))))
    out.sort(key=lambda x: -x[2])
    return out


def compute_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def compute_key_recall(pred_start: int, pred_end: int, key_frames: np.ndarray) -> float:
    if len(key_frames) == 0:
        return 0.0
    inside = int(np.sum((key_frames >= pred_start) & (key_frames < pred_end)))
    return float(inside) / float(len(key_frames))


def _best_match_for_gt(
    pred_segments: list[tuple[int, int, float]],
    gt: SegmentGT,
) -> tuple[float, float, Optional[tuple[int, int, float]]]:
    best_iou = 0.0
    best_key_recall = 0.0
    best_pred = None
    for pred in pred_segments:
        iou = compute_iou(pred[0], pred[1], gt.start_frame, gt.end_frame)
        if iou > best_iou:
            best_iou = iou
            best_key_recall = compute_key_recall(pred[0], pred[1], gt.key_frames)
            best_pred = pred
    return best_iou, best_key_recall, best_pred


def evaluate_dense_labeling(
    model: nn.Module,
    records: list[SessionRecord],
    device: torch.device,
    threshold: float,
    min_segment_frames: int,
    merge_gap_frames: int,
    prob_smooth_window: int = 1,
    valley_merge_threshold: float = 0.0,
    valley_merge_max_gap_frames: int = 0,
) -> tuple[dict, list[dict]]:
    model.eval()
    details: list[dict] = []
    single_top1_ious = []
    single_top1_complete = []
    single_top1_key_recall = []
    all_gt_best_ious = []
    all_gt_best_key_recalls = []
    all_gt_complete = []
    pred_counts = []

    with torch.no_grad():
        for rec in records:
            x = torch.from_numpy(rec.features).float().unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
            pred_segments = extract_segments(
                probs,
                threshold=threshold,
                min_length=min_segment_frames,
                merge_gap=merge_gap_frames,
                prob_smooth_window=prob_smooth_window,
                valley_merge_threshold=valley_merge_threshold,
                valley_merge_max_gap=valley_merge_max_gap_frames,
            )
            pred_counts.append(len(pred_segments))

            if len(rec.gt_segments) == 1:
                gt = rec.gt_segments[0]
                if pred_segments:
                    pred = pred_segments[0]
                    iou = compute_iou(pred[0], pred[1], gt.start_frame, gt.end_frame)
                    key_recall = compute_key_recall(pred[0], pred[1], gt.key_frames)
                else:
                    pred = None
                    iou = 0.0
                    key_recall = 0.0
                single_top1_ious.append(float(iou))
                single_top1_key_recall.append(float(key_recall))
                single_top1_complete.append(float(key_recall >= 0.999))

            gt_rows = []
            for gt in rec.gt_segments:
                best_iou, best_key_recall, best_pred = _best_match_for_gt(pred_segments, gt)
                all_gt_best_ious.append(float(best_iou))
                all_gt_best_key_recalls.append(float(best_key_recall))
                all_gt_complete.append(float(best_key_recall >= 0.999))
                gt_rows.append({
                    "episode_id": gt.episode_id,
                    "password": gt.password,
                    "gt_start_frame": int(gt.start_frame),
                    "gt_end_frame": int(gt.end_frame),
                    "best_iou": float(best_iou),
                    "best_key_recall": float(best_key_recall),
                    "best_pred": None if best_pred is None else {
                        "start_frame": int(best_pred[0]),
                        "end_frame": int(best_pred[1]),
                        "confidence": float(best_pred[2]),
                    },
                })

            details.append({
                "session_id": rec.session_id,
                "num_gt_segments": len(rec.gt_segments),
                "num_pred_segments": len(pred_segments),
                "pred_segments_top5": [
                    {"start_frame": int(p[0]), "end_frame": int(p[1]), "confidence": float(p[2])}
                    for p in pred_segments[:5]
                ],
                "gt_rows": gt_rows,
            })

    report = {
        "method": "dense_frame_labeling_unet1d",
        "n_sessions": len(records),
        "mean_pred_segments": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "all_gt_oracle": {
            "mean_best_iou": float(np.mean(all_gt_best_ious)) if all_gt_best_ious else 0.0,
            "iou_ge_0.7": float(np.mean([x >= 0.7 for x in all_gt_best_ious])) if all_gt_best_ious else 0.0,
            "iou_ge_0.5": float(np.mean([x >= 0.5 for x in all_gt_best_ious])) if all_gt_best_ious else 0.0,
            "mean_best_key_recall": float(np.mean(all_gt_best_key_recalls)) if all_gt_best_key_recalls else 0.0,
            "complete_hit_rate": float(np.mean(all_gt_complete)) if all_gt_complete else 0.0,
        },
        "single_session_top1": {
            "n_sessions": int(len(single_top1_ious)),
            "mean_iou": float(np.mean(single_top1_ious)) if single_top1_ious else 0.0,
            "iou_ge_0.7": float(np.mean([x >= 0.7 for x in single_top1_ious])) if single_top1_ious else 0.0,
            "iou_ge_0.5": float(np.mean([x >= 0.5 for x in single_top1_ious])) if single_top1_ious else 0.0,
            "mean_key_recall": float(np.mean(single_top1_key_recall)) if single_top1_key_recall else 0.0,
            "complete_hit_rate": float(np.mean(single_top1_complete)) if single_top1_complete else 0.0,
        },
    }
    return report, details


def evaluate_posthoc_grid(
    model: nn.Module,
    records: list[SessionRecord],
    device: torch.device,
    thresholds: list[float],
    min_segment_seconds: list[float],
    merge_gap_seconds: list[float],
    prob_smooth_windows: list[int],
    valley_merge_thresholds: list[float],
    valley_merge_gap_seconds: list[float],
) -> tuple[dict, list[dict], dict]:
    if not records:
        empty = {
            "threshold": None,
            "min_segment_s": None,
            "merge_gap_s": None,
            "selection_score": 0.0,
            "report": {},
        }
        return empty, [], empty

    median_sr = float(np.median([r.sample_rate_hz for r in records]))
    best_bundle = None
    for thr in thresholds:
        for min_s in min_segment_seconds:
            for gap_s in merge_gap_seconds:
                for smooth_w in prob_smooth_windows:
                    for valley_thr in valley_merge_thresholds:
                        for valley_gap_s in valley_merge_gap_seconds:
                            report, details = evaluate_dense_labeling(
                                model,
                                records,
                                device,
                                threshold=float(thr),
                                min_segment_frames=int(round(float(min_s) * median_sr)),
                                merge_gap_frames=int(round(float(gap_s) * median_sr)),
                                prob_smooth_window=int(smooth_w),
                                valley_merge_threshold=float(valley_thr),
                                valley_merge_max_gap_frames=int(round(float(valley_gap_s) * median_sr)),
                            )
                            single = report["single_session_top1"]
                            oracle = report["all_gt_oracle"]
                            score = (
                                float(single["mean_iou"])
                                + 0.30 * float(single["complete_hit_rate"])
                                + 0.10 * float(oracle["mean_best_iou"])
                            )
                            bundle = {
                                "threshold": float(thr),
                                "min_segment_s": float(min_s),
                                "merge_gap_s": float(gap_s),
                                "prob_smooth_window": int(smooth_w),
                                "valley_merge_threshold": float(valley_thr),
                                "valley_merge_gap_s": float(valley_gap_s),
                                "selection_score": float(score),
                                "report": report,
                                "details": details,
                            }
                            if best_bundle is None or bundle["selection_score"] > best_bundle["selection_score"]:
                                best_bundle = bundle
    assert best_bundle is not None
    return best_bundle, best_bundle["details"], best_bundle


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: DenseSegmentationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict]:
    model.train()
    total = 0.0
    comps = {"bce": 0.0, "dice": 0.0, "smooth": 0.0}
    n_batches = 0
    for x, y, mask in dataloader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss, part = criterion(logits, y, valid_mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total += float(loss.item())
        for k in comps:
            comps[k] += float(part[k])
        n_batches += 1
    if n_batches == 0:
        return 0.0, comps
    return total / n_batches, {k: v / n_batches for k, v in comps.items()}


def _save_eval_preview(
    model: nn.Module,
    records: list[SessionRecord],
    device: torch.device,
    output_dir: Path,
    threshold: float,
    min_segment_frames: int,
    merge_gap_frames: int,
    prob_smooth_window: int = 1,
    valley_merge_threshold: float = 0.0,
    valley_merge_max_gap_frames: int = 0,
    limit: int = 5,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[Warn] matplotlib is not installed; skipping eval preview rendering.")
        return

    model.eval()
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for rec in records[:limit]:
            x = torch.from_numpy(rec.features).float().unsqueeze(0).to(device)
            probs = torch.sigmoid(model(x)).squeeze().cpu().numpy()
            preds = extract_segments(
                probs,
                threshold,
                min_segment_frames,
                merge_gap_frames,
                prob_smooth_window=prob_smooth_window,
                valley_merge_threshold=valley_merge_threshold,
                valley_merge_max_gap=valley_merge_max_gap_frames,
            )

            signal = np.linalg.norm(rec.features[:6].T, axis=1)
            fig, axes = plt.subplots(2, 1, figsize=(20, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
            axes[0].plot(signal, linewidth=0.6, color="#374151")
            for gt in rec.gt_segments:
                axes[0].axvspan(gt.start_frame, gt.end_frame, alpha=0.16, color="#10b981")
            for i, pred in enumerate(preds[:3]):
                color = "#ef4444" if i == 0 else "#f59e0b"
                axes[0].axvspan(pred[0], pred[1], alpha=0.14, color=color)
            axes[0].set_title(rec.session_id)
            axes[0].set_ylabel("IMU norm")

            axes[1].fill_between(np.arange(len(probs)), probs, color="#3b82f6", alpha=0.6)
            axes[1].axhline(threshold, color="#ef4444", linestyle="--", linewidth=0.8)
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_ylabel("P(password)")
            axes[1].set_xlabel("Frame")
            fig.tight_layout()
            fig.savefig(vis_dir / f"{rec.session_id}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def train(args) -> None:
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records: list[SessionRecord] = []
    if args.train_mixed_dirs:
        train_records.extend(
            _build_session_records(
                roots=args.train_mixed_dirs,
                pre_pad_ms=args.label_pre_pad_ms,
                post_pad_ms=args.label_post_pad_ms,
                feature_mode=args.feature_mode,
                min_password_len=args.min_password_len,
            )
        )
    if args.train_password_dirs:
        password_records = _build_password_attempt_records(
            roots=args.train_password_dirs,
            pre_pad_ms=args.label_pre_pad_ms,
            post_pad_ms=args.label_post_pad_ms,
            feature_mode=args.feature_mode,
            min_password_len=args.min_password_len,
            context_pre_ms=args.password_context_pre_ms,
            context_post_ms=args.password_context_post_ms,
            require_match=not args.allow_unmatched_password_attempts,
        )
        train_records.extend(password_records)
    if args.train_onset_negative_root:
        target_count = int(args.train_negative_target_count)
        if target_count <= 0:
            target_count = max(40, len(train_records))
        train_records.extend(
            _build_onset_negative_records(
                root=args.train_onset_negative_root,
                feature_mode=args.feature_mode,
                target_count=target_count,
            )
        )
    eval_records = _build_session_records(
        roots=args.eval_dirs,
        pre_pad_ms=args.label_pre_pad_ms,
        post_pad_ms=args.label_post_pad_ms,
        feature_mode=args.feature_mode,
        min_password_len=args.min_password_len,
    )
    if not train_records:
        raise RuntimeError("No train sessions were loaded.")
    if not eval_records:
        raise RuntimeError("No eval sessions were loaded.")

    train_dataset = FullStreamDataset(
        train_records,
        train_max_len=args.train_max_len,
        augment=True,
        positive_crop_prob=args.positive_crop_prob,
        label_jitter_frames=args.label_jitter_frames,
        time_stretch_prob=args.time_stretch_prob,
        time_stretch_min=args.time_stretch_min,
        time_stretch_max=args.time_stretch_max,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    in_channels = int(train_records[0].features.shape[0])
    model = UNet1D(
        in_channels=in_channels,
        base_filters=args.base_filters,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_attention=args.use_attention,
    ).to(device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=True)
    criterion = DenseSegmentationLoss(
        pos_weight=args.pos_weight,
        boundary_width=args.boundary_width,
        boundary_boost=args.boundary_boost,
        smooth_weight=args.smooth_weight,
        dice_weight=args.dice_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    history = []
    best_score = -1.0
    best_report = None
    best_details = None

    min_segment_frames = int(round(args.min_segment_s * np.median([r.sample_rate_hz for r in eval_records])))
    merge_gap_frames = int(round(args.merge_gap_s * np.median([r.sample_rate_hz for r in eval_records])))
    sweep_thresholds = [float(x) for x in args.sweep_thresholds]
    sweep_min_segments = [float(x) for x in args.sweep_min_segment_s]
    sweep_merge_gaps = [float(x) for x in args.sweep_merge_gap_s]
    sweep_prob_smooth_windows = [int(x) for x in args.sweep_prob_smooth_windows]
    sweep_valley_thresholds = [float(x) for x in args.sweep_valley_merge_thresholds]
    sweep_valley_gap_seconds = [float(x) for x in args.sweep_valley_merge_gap_s]

    print(f"[Train] device={device} train_sessions={len(train_records)} eval_sessions={len(eval_records)} in_channels={in_channels}")
    for epoch in range(1, args.epochs + 1):
        train_loss, parts = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"loss={train_loss:.4f} bce={parts['bce']:.4f} dice={parts['dice']:.4f} smooth={parts['smooth']:.4f}"
        )
        if epoch % args.eval_every != 0 and epoch != args.epochs:
            continue
        report, details = evaluate_dense_labeling(
            model,
            eval_records,
            device,
            threshold=args.threshold,
            min_segment_frames=min_segment_frames,
            merge_gap_frames=merge_gap_frames,
        )
        single = report["single_session_top1"]
        oracle = report["all_gt_oracle"]
        print(
            "  [Eval] "
            f"single_top1_iou={single['mean_iou']:.4f} "
            f"single_complete={single['complete_hit_rate']:.3f} "
            f"oracle_iou={oracle['mean_best_iou']:.4f} "
            f"oracle_complete={oracle['complete_hit_rate']:.3f}"
        )
        sweep_best = None
        score = float(single["mean_iou"]) + 0.5 * float(single["complete_hit_rate"])
        if args.select_by_posthoc_sweep:
            sweep_best, _, _ = evaluate_posthoc_grid(
                model,
                eval_records,
                device,
                thresholds=sweep_thresholds,
                min_segment_seconds=sweep_min_segments,
                merge_gap_seconds=sweep_merge_gaps,
                prob_smooth_windows=sweep_prob_smooth_windows,
                valley_merge_thresholds=sweep_valley_thresholds,
                valley_merge_gap_seconds=sweep_valley_gap_seconds,
            )
            sweep_rep = sweep_best["report"]
            sweep_single = sweep_rep["single_session_top1"]
            print(
                "  [Sweep] "
                f"thr={sweep_best['threshold']:.2f} "
                f"min_s={sweep_best['min_segment_s']:.2f} "
                f"gap_s={sweep_best['merge_gap_s']:.2f} "
                f"smooth_w={sweep_best['prob_smooth_window']} "
                f"valley_thr={sweep_best['valley_merge_threshold']:.2f} "
                f"valley_gap_s={sweep_best['valley_merge_gap_s']:.2f} "
                f"single_top1_iou={sweep_single['mean_iou']:.4f} "
                f"single_complete={sweep_single['complete_hit_rate']:.3f}"
            )
            score = float(sweep_best["selection_score"])
        hist_row = {"epoch": epoch, "train_loss": train_loss, **report}
        if sweep_best is not None:
            hist_row["posthoc_best"] = {
                "threshold": sweep_best["threshold"],
                "min_segment_s": sweep_best["min_segment_s"],
                "merge_gap_s": sweep_best["merge_gap_s"],
                "selection_score": sweep_best["selection_score"],
                "report": sweep_best["report"],
            }
        history.append(hist_row)
        if score > best_score:
            best_score = score
            best_report = sweep_best["report"] if sweep_best is not None else report
            best_details = sweep_best["details"] if sweep_best is not None else details
            torch.save(model.state_dict(), output_dir / "best_dense_labeling.pt")
            with open(output_dir / "best_report.json", "w") as f:
                json.dump(best_report, f, indent=2)
            with open(output_dir / "best_details.json", "w") as f:
                json.dump(best_details, f, indent=2)
            if sweep_best is not None:
                with open(output_dir / "best_posthoc.json", "w") as f:
                    json.dump(
                        {
                            "threshold": sweep_best["threshold"],
                            "min_segment_s": sweep_best["min_segment_s"],
                            "merge_gap_s": sweep_best["merge_gap_s"],
                            "prob_smooth_window": sweep_best["prob_smooth_window"],
                            "valley_merge_threshold": sweep_best["valley_merge_threshold"],
                            "valley_merge_gap_s": sweep_best["valley_merge_gap_s"],
                            "selection_score": sweep_best["selection_score"],
                            "report": sweep_best["report"],
                        },
                        f,
                        indent=2,
                    )
            _save_eval_preview(
                model,
                eval_records,
                device,
                output_dir,
                threshold=float(sweep_best["threshold"]) if sweep_best is not None else args.threshold,
                min_segment_frames=int(round(float(sweep_best["min_segment_s"]) * np.median([r.sample_rate_hz for r in eval_records]))) if sweep_best is not None else min_segment_frames,
                merge_gap_frames=int(round(float(sweep_best["merge_gap_s"]) * np.median([r.sample_rate_hz for r in eval_records]))) if sweep_best is not None else merge_gap_frames,
                prob_smooth_window=int(sweep_best["prob_smooth_window"]) if sweep_best is not None else 1,
                valley_merge_threshold=float(sweep_best["valley_merge_threshold"]) if sweep_best is not None else 0.0,
                valley_merge_max_gap_frames=int(round(float(sweep_best["valley_merge_gap_s"]) * np.median([r.sample_rate_hz for r in eval_records]))) if sweep_best is not None else 0,
            )

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    final_report = {
        "method": "dense_frame_labeling_unet1d",
        "feature_mode": args.feature_mode,
        "train_mixed_dirs": [str(x) for x in args.train_mixed_dirs],
        "train_password_dirs": [str(x) for x in args.train_password_dirs],
        "train_onset_negative_root": str(args.train_onset_negative_root),
        "train_negative_target_count": int(args.train_negative_target_count),
        "eval_dirs": [str(x) for x in args.eval_dirs],
        "init_checkpoint": str(args.init_checkpoint),
        "label_pre_pad_ms": args.label_pre_pad_ms,
        "label_post_pad_ms": args.label_post_pad_ms,
        "password_context_pre_ms": args.password_context_pre_ms,
        "password_context_post_ms": args.password_context_post_ms,
        "label_jitter_frames": args.label_jitter_frames,
        "time_stretch_prob": args.time_stretch_prob,
        "time_stretch_min": args.time_stretch_min,
        "time_stretch_max": args.time_stretch_max,
        "best_selection_score": float(best_score),
        "best_report": best_report,
        "train_sessions": [r.session_id for r in train_records],
        "eval_sessions": [r.session_id for r in eval_records],
    }
    with open(output_dir / "report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    if best_details is not None:
        with open(output_dir / "details.json", "w") as f:
            json.dump(best_details, f, indent=2)
    print(f"[Done] saved to {output_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Stage1 dense frame labeling on real full-stream IMU sessions")
    ap.add_argument("--train_mixed_dirs", nargs="*", default=[])
    ap.add_argument("--train_password_dirs", nargs="*", default=[])
    ap.add_argument("--train_onset_negative_root", default="")
    ap.add_argument("--train_negative_target_count", type=int, default=0)
    ap.add_argument("--eval_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--init_checkpoint", default="")

    ap.add_argument("--label_pre_pad_ms", type=float, default=220.0)
    ap.add_argument("--label_post_pad_ms", type=float, default=380.0)
    ap.add_argument("--password_context_pre_ms", type=float, default=1200.0)
    ap.add_argument("--password_context_post_ms", type=float, default=1200.0)
    ap.add_argument("--allow_unmatched_password_attempts", action="store_true")
    ap.add_argument("--min_password_len", type=int, default=4)
    ap.add_argument("--feature_mode", choices=["raw6", "raw6_energy_activity", "raw6_energy_activity_pulse"], default="raw6_energy_activity")

    ap.add_argument("--train_max_len", type=int, default=8192)
    ap.add_argument("--positive_crop_prob", type=float, default=0.8)
    ap.add_argument("--label_jitter_frames", type=int, default=120)
    ap.add_argument("--time_stretch_prob", type=float, default=0.7)
    ap.add_argument("--time_stretch_min", type=float, default=0.75)
    ap.add_argument("--time_stretch_max", type=float, default=1.35)

    ap.add_argument("--base_filters", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--kernel_size", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_attention", action="store_true")

    ap.add_argument("--pos_weight", type=float, default=8.0)
    ap.add_argument("--boundary_width", type=int, default=80)
    ap.add_argument("--boundary_boost", type=float, default=4.0)
    ap.add_argument("--smooth_weight", type=float, default=0.03)
    ap.add_argument("--dice_weight", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--eval_every", type=int, default=5)

    ap.add_argument("--threshold", type=float, default=0.50)
    ap.add_argument("--min_segment_s", type=float, default=0.80)
    ap.add_argument("--merge_gap_s", type=float, default=0.25)
    ap.add_argument("--select_by_posthoc_sweep", action="store_true")
    ap.add_argument("--sweep_thresholds", nargs="+", type=float, default=[0.5, 0.55, 0.6])
    ap.add_argument("--sweep_min_segment_s", nargs="+", type=float, default=[0.5, 0.8])
    ap.add_argument("--sweep_merge_gap_s", nargs="+", type=float, default=[0.25, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--sweep_prob_smooth_windows", nargs="+", type=int, default=[1, 161])
    ap.add_argument("--sweep_valley_merge_thresholds", nargs="+", type=float, default=[0.0, 0.30])
    ap.add_argument("--sweep_valley_merge_gap_s", nargs="+", type=float, default=[0.0, 1.5])
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
