#!/usr/bin/env python3
from __future__ import annotations

"""
Stage1 bag-ranking replacement for the current slice-CNN / pointwise-regression branch.

Core changes versus the old stage1 scripts:
1. Candidate proposal no longer relies only on a single peak-cluster span.  It builds
   micro-bursts from the full-stream energy trace, then enumerates unions of adjacent
   micro-bursts.  This fixes sessions where the true password is split into two nearby
   sub-clusters and the old proposer never creates a full candidate.
2. Training is bag/listwise, not pointwise.  We train *within each session/episode bag*
   so the model learns which candidate should be ranked above its session-local rivals.
3. The supervision target is downstream recoverability, not IoU alone.  A candidate is
   good if the existing peak-keyness model can recover the full key sequence from it,
   even if the crop is a little long.  Missing one key is penalized more strongly than
   having a modest amount of extra context.
4. The representation includes context channels (outer-shell energy before/after the
   candidate), because humans use that context when visually spotting the password burst.
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from numpy.fft import irfft, rfft
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, resample
from sklearn.ensemble import RandomForestClassifier

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
from onset_detection.stage2_segmental.length_model import load_length_model
from onset_detection.stage2_segmental.metrics import aggregate_episode_results
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.scripts.eval_fullstream_clusterfirst import (
    _extract_window_from_signal,
    _evaluate_fixed_from_frames,
    _evaluate_overlap_from_frames,
)
from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import (
    _build_dataset as _build_peak_dataset,
    _load_password_attempt_episodes,
    _peak_feature_vector,
    _propose_peaks,
    _select_k_peaks,
    _smooth,
)
from onset_detection.stage2_segmental.scripts.train_eval_segment_passwordness import (
    _propose_candidates_fullstream,
)

SEED = 42
TARGET_LEN = 192
EPS = 1e-6
rng = random.Random(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class CandidateBag:
    session_id: str
    episode_id: str
    expected_len: int
    xs_seq: np.ndarray   # [N, C, T]
    xs_aux: np.ndarray   # [N, F]
    targets: np.ndarray  # [N]
    candidates: list[dict]
    target_debug: list[dict]


class ContextRankCNN(nn.Module):
    def __init__(self, in_ch: int, aux_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(96), nn.MaxPool1d(2),
            nn.Conv1d(96, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + aux_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(48, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x_seq).squeeze(-1)
        h = torch.cat([h, x_aux], dim=1)
        return self.head(h).squeeze(1)


def _predict_length_from_crop(length_model, crop_imu: np.ndarray, crop_ts: np.ndarray, fallback: int) -> int:
    pred, _conf, _dbg = _predict_length_with_debug(length_model, crop_imu, crop_ts, fallback)
    return pred


def _predict_length_with_debug(length_model, crop_imu: np.ndarray, crop_ts: np.ndarray, fallback: int) -> tuple[int, float, dict]:
    if length_model is None:
        return int(fallback), 0.0, {'used_length_model': False}
    try:
        model, labels, meta = length_model
        from onset_detection.stage2_segmental.length_model import compute_region_length_features
        feat = compute_region_length_features(
            crop_imu,
            crop_ts,
            feature_mode=str(meta.get('feature_mode', 'no_time')),
        ).reshape(1, -1)
        pred = int(model.predict(feat)[0])
        conf = 0.0
        probs_dict = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(feat)[0]
            conf = float(np.max(proba))
            probs_dict = {str(int(lbl)): float(p) for lbl, p in zip(model.classes_, proba)}
        return max(4, pred), conf, {'used_length_model': True, 'predicted_length': int(pred), 'length_probs': probs_dict}
    except Exception:
        return int(fallback), 0.0, {'used_length_model': False, 'predict_failed': True}


def resolve_device(name: str) -> torch.device:
    req = name.lower()
    if req == 'auto':
        if torch.cuda.is_available():
            req = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            req = 'mps'
        else:
            req = 'cpu'
    return torch.device(req)


def _proposal_env_window(sr: float) -> int:
    # Match peak proposal: a short envelope preserves password pulses for downstream clustering.
    return max(1, int(round(float(sr) * 0.10)))


def _proposal_energy_envelope(imu: np.ndarray, sr: float) -> np.ndarray:
    return _compute_energy_envelope(imu, _proposal_env_window(sr)).astype(np.float64)


def _normalized_curve(values: np.ndarray, out_len: int = TARGET_LEN) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if len(vals) < 4:
        vals = np.pad(vals, (0, max(0, 4 - len(vals))), mode='edge')
    out = resample(vals, out_len)
    if np.iscomplexobj(out):
        out = np.real(out)
    out = np.asarray(out, dtype=np.float32)
    out -= float(out.min())
    mx = float(out.max())
    if mx > 1e-8:
        out /= mx
    return out


def _resample_multichannel(arr: np.ndarray, out_len: int = TARGET_LEN) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if len(x) < 4:
        x = np.pad(x, ((0, max(0, 4 - len(x))), (0, 0)), mode='edge')
    out = resample(x, out_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    out = np.asarray(out, dtype=np.float32)
    out = out - out.mean(axis=0, keepdims=True)
    out = out / np.maximum(out.std(axis=0, keepdims=True), 1e-6)
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad + 1e-6
    return ((x - med) / scale).astype(np.float32)


def _segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    segs = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segs.append((start, i))
            in_seg = False
    if in_seg:
        segs.append((start, len(mask)))
    return segs


def _bridge_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    segs = _segments_from_mask(out)
    if len(segs) < 2:
        return out
    for (a0, a1), (b0, b1) in zip(segs[:-1], segs[1:]):
        if 0 < b0 - a1 <= max_gap:
            out[a1:b0] = True
    return out


def _micro_segments_from_energy(energy: np.ndarray, sr: float) -> list[tuple[int, int, dict]]:
    z = _robust_z(energy)
    seen = set()
    out = []
    for smooth_s in (0.08, 0.12, 0.18, 0.25):
        sm = _smooth(z, max(1, int(round(sr * smooth_s))))
        for hi_thr, lo_thr in ((1.8, 0.8), (1.5, 0.7), (1.2, 0.5)):
            seed = sm >= hi_thr
            grow = sm >= lo_thr
            if not np.any(seed):
                continue
            active = seed.copy()
            changed = True
            while changed:
                changed = False
                idx = np.where(active)[0]
                if len(idx) == 0:
                    break
                lo = max(0, int(idx[0]) - 1)
                hi = min(len(active), int(idx[-1]) + 2)
                new = active.copy()
                new[lo:hi] |= grow[lo:hi]
                if np.any(new != active):
                    active = new
                    changed = True
            active = _bridge_short_gaps(active, max_gap=max(1, int(round(sr * 0.18))))
            for lo, hi in _segments_from_mask(active):
                span = hi - lo
                if span < int(round(sr * 0.18)):
                    continue
                key = (int(lo), int(hi))
                if key in seen:
                    continue
                seen.add(key)
                seg_energy = sm[lo:hi]
                out.append((lo, hi, {
                    'smooth_s': float(smooth_s),
                    'energy_mean': float(np.mean(seg_energy)) if len(seg_energy) else 0.0,
                    'energy_max': float(np.max(seg_energy)) if len(seg_energy) else 0.0,
                }))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _enumerate_union_candidates(imu: np.ndarray, energy: np.ndarray, micro: list[tuple[int, int, dict]], sr: float) -> list[dict]:
    seen = set()
    out = []
    n = len(micro)
    if n == 0:
        return out
    for i in range(n):
        union_lo = micro[i][0]
        union_hi = micro[i][1]
        total_gap = 0
        for j in range(i, min(n, i + 4)):
            if j > i:
                gap = max(0, micro[j][0] - micro[j - 1][1])
                total_gap += gap
                union_hi = micro[j][1]
                if total_gap > int(round(sr * 1.30)):
                    break
            for pre_s in (0.18, 0.35, 0.55):
                for post_s in (0.22, 0.40, 0.65):
                    lo = max(0, int(round(union_lo - sr * pre_s)))
                    hi = min(len(imu), int(round(union_hi + sr * post_s)))
                    if hi - lo < int(round(sr * 0.80)):
                        continue
                    span_s = (hi - lo) / max(sr, 1e-6)
                    if span_s > 12.0:
                        continue
                    peaks, _ = find_peaks(
                        _smooth(energy[lo:hi], max(1, int(round(sr * 0.12)))),
                        distance=max(1, int(round(sr * 0.22))),
                        prominence=max(1e-6, float(np.std(energy[lo:hi])) * 0.20),
                    )
                    key = (lo, hi, j - i + 1, len(peaks))
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        'crop_start': int(lo),
                        'crop_end': int(hi),
                        'macro_num_peaks': int(len(peaks)),
                        'num_micro_segments': int(j - i + 1),
                        'internal_gap_s': float(total_gap / max(sr, 1e-6)),
                        'source': 'micro_union',
                    })
    return out


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
            'start_frame': int(p[0]),
            'end_frame': int(p[-1]),
            'num_peaks': int(len(p)),
            'score_sum': float(np.sum(s)),
            'score_mean': float(np.mean(s)),
        })
    return out


def _cluster_key_peaks(peaks: np.ndarray, probs: np.ndarray, sample_rate: float, gap_s: float) -> list[dict]:
    if len(peaks) == 0:
        return []
    order = np.argsort(peaks)
    peaks = peaks[order]
    probs = probs[order]
    max_gap_frames = max(1, int(round(sample_rate * gap_s)))
    groups: list[list[int]] = []
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
        s = probs[idx]
        out.append({
            'start_frame': int(p[0]),
            'end_frame': int(p[-1]),
            'num_peaks': int(len(p)),
            'score_mean': float(np.mean(s)),
            'score_sum': float(np.sum(s)),
            'peak_indices': p.tolist(),
        })
    return out


def _keyness_union_candidates(
    clusters: list[dict],
    signal_len: int,
    sample_rate_hz: float,
    max_union_size: int = 3,
    max_gap_s: float = 2.4,
) -> list[dict]:
    if not clusters:
        return []
    ordered = sorted(clusters, key=lambda c: (int(c['start_frame']), int(c['end_frame'])))
    out = []
    seen = set()
    max_gap_frames = int(round(sample_rate_hz * max_gap_s))
    for i in range(len(ordered)):
        lo = int(ordered[i]['start_frame'])
        hi = int(ordered[i]['end_frame'])
        peak_sum = int(ordered[i].get('num_peaks', 0))
        score_means = [float(ordered[i].get('score_mean', 0.0))]
        total_gap = 0
        for j in range(i + 1, min(len(ordered), i + max_union_size)):
            nxt = ordered[j]
            gap = int(nxt['start_frame']) - hi
            if gap > max_gap_frames:
                break
            total_gap += max(0, gap)
            lo = min(lo, int(nxt['start_frame']))
            hi = max(hi, int(nxt['end_frame']))
            peak_sum += int(nxt.get('num_peaks', 0))
            score_means.append(float(nxt.get('score_mean', 0.0)))
            for pre_s in (0.35, 0.55, 0.75):
                for post_s in (0.35, 0.55, 0.75):
                    crop_lo = max(0, int(round(lo - sample_rate_hz * pre_s)))
                    crop_hi = min(signal_len, int(round(hi + sample_rate_hz * post_s)) + 1)
                    key = (crop_lo, crop_hi, j - i + 1, peak_sum)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        'crop_start': int(crop_lo),
                        'crop_end': int(crop_hi),
                        'macro_num_peaks': int(peak_sum),
                        'num_micro_segments': int(j - i + 1),
                        'internal_gap_s': float(total_gap / max(sample_rate_hz, 1e-6)),
                        'cluster_score_mean': float(np.mean(score_means)),
                        'source': 'keyness_union',
                    })
    return out


def _legacy_cluster_candidates(imu: np.ndarray, sr: float) -> list[dict]:
    energy_raw = _proposal_energy_envelope(imu, sr)
    seen = set()
    candidates = []
    for smooth_s in (0.10, 0.15, 0.22, 0.30):
        smoothed = _smooth(energy_raw, max(1, int(round(sr * smooth_s))))
        q50, q90 = np.quantile(smoothed, [0.50, 0.90])
        prominence = max(1e-6, (q90 - q50) * 0.08)
        for dist_s in (0.25, 0.35, 0.50, 0.70):
            peaks, props = find_peaks(
                smoothed,
                distance=max(1, int(round(sr * dist_s))),
                prominence=prominence,
            )
            if len(peaks) == 0:
                continue
            heights = np.asarray(props.get('peak_heights', smoothed[peaks]), dtype=np.float64)
            peak_scores = heights / max(float(np.max(heights)), 1e-8)
            for gap_s in (0.9, 1.2, 1.6, 2.0):
                for cluster in _cluster_macro_peaks(peaks, peak_scores, sr, gap_s=gap_s):
                    if not (3 <= cluster['num_peaks'] <= 20):
                        continue
                    pad_frames = int(round(sr * 0.60))
                    lo = max(0, int(cluster['start_frame']) - pad_frames)
                    hi = min(len(imu), int(cluster['end_frame']) + pad_frames + 1)
                    key = (lo, hi, cluster['num_peaks'])
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append({
                        'crop_start': int(lo),
                        'crop_end': int(hi),
                        'macro_num_peaks': int(cluster['num_peaks']),
                        'num_micro_segments': 1,
                        'internal_gap_s': float(gap_s),
                        'source': 'legacy_cluster',
                    })
    return candidates


def _keyness_candidates_fullstream(
    imu: np.ndarray,
    ts: np.ndarray,
    sr: float,
    peak_model,
    strong_threshold: float = 0.35,
    min_strong_peaks: int = 3,
) -> list[dict]:
    peaks, sm, _ = _propose_peaks({'imu': imu, 'sample_rate_hz': sr, 'timestamps_ns': ts})
    if len(peaks) == 0:
        return []
    Xp = np.stack([_peak_feature_vector(sm, peaks, i, sr) for i in range(len(peaks))]).astype(np.float32)
    probs = peak_model.predict_proba(Xp)[:, 1].astype(np.float32)

    strong_mask = probs >= float(strong_threshold)
    if int(np.sum(strong_mask)) < int(min_strong_peaks):
        topk = min(max(6, len(peaks) // 8), len(peaks))
        strong_idx = np.argsort(probs)[-topk:]
        strong_idx = np.sort(strong_idx)
        strong_peaks = peaks[strong_idx]
        strong_probs = probs[strong_idx]
    else:
        strong_peaks = peaks[strong_mask]
        strong_probs = probs[strong_mask]

    seen = set()
    out = []
    all_clusters = []
    for gap_s in (1.2, 1.6, 2.0, 2.4):
        cur_clusters = _cluster_key_peaks(strong_peaks, strong_probs, sr, gap_s=gap_s)
        all_clusters.extend(cur_clusters)
        for cluster in cur_clusters:
            if not (3 <= cluster['num_peaks'] <= 20):
                continue
            for pre_s in (0.35, 0.55, 0.75):
                for post_s in (0.35, 0.55, 0.75):
                    lo = max(0, int(round(cluster['start_frame'] - sr * pre_s)))
                    hi = min(len(imu), int(round(cluster['end_frame'] + sr * post_s)) + 1)
                    if hi - lo < int(round(sr * 0.70)):
                        continue
                    key = (lo, hi, cluster['num_peaks'])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({
                        'crop_start': int(lo),
                        'crop_end': int(hi),
                        'macro_num_peaks': int(cluster['num_peaks']),
                        'num_micro_segments': 1,
                        'internal_gap_s': float(gap_s),
                        'cluster_score_mean': float(cluster['score_mean']),
                        'source': 'keyness_cluster',
                    })
    for cand in _keyness_union_candidates(all_clusters, len(imu), sr):
        key = (int(cand['crop_start']), int(cand['crop_end']), int(cand['macro_num_peaks']))
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _candidate_iou(a: dict, b: dict) -> float:
    lo = max(int(a['crop_start']), int(b['crop_start']))
    hi = min(int(a['crop_end']), int(b['crop_end']))
    inter = max(0, hi - lo)
    union = max(int(a['crop_end']), int(b['crop_end'])) - min(int(a['crop_start']), int(b['crop_start']))
    return inter / max(union, 1)


def _oldpool_union_candidates(
    candidates: list[dict],
    sample_rate_hz: float,
    max_union_size: int = 3,
    max_gap_s: float = 2.2,
) -> list[dict]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: (int(c['crop_start']), int(c['crop_end'])))
    out = []
    seen = set()
    max_gap_frames = int(round(sample_rate_hz * max_gap_s))
    for i in range(len(ordered)):
        lo = int(ordered[i]['crop_start'])
        hi = int(ordered[i]['crop_end'])
        peak_sum = int(ordered[i].get('cluster_num_peaks', ordered[i].get('macro_num_peaks', 0)))
        score_means = [float(ordered[i].get('cluster_score_mean', 0.0))]
        for j in range(i + 1, min(len(ordered), i + max_union_size)):
            nxt = ordered[j]
            gap = int(nxt['crop_start']) - hi
            if gap > max_gap_frames:
                break
            lo = min(lo, int(nxt['crop_start']))
            hi = max(hi, int(nxt['crop_end']))
            peak_sum += int(nxt.get('cluster_num_peaks', nxt.get('macro_num_peaks', 0)))
            score_means.append(float(nxt.get('cluster_score_mean', 0.0)))
            key = (lo, hi, j - i + 1)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                'crop_start': int(lo),
                'crop_end': int(hi),
                'macro_num_peaks': int(peak_sum),
                'num_micro_segments': int(j - i + 1),
                'internal_gap_s': float(max(0, gap) / max(sample_rate_hz, 1e-6)),
                'cluster_score_mean': float(np.mean(score_means)),
                'source': 'oldpool_union',
            })
    return out


def _pre_score_candidate(candidate: dict, expected_len: Optional[int]) -> float:
    n_peaks = float(candidate.get('macro_num_peaks', candidate.get('cluster_num_peaks', 0)))
    n_micro = float(candidate.get('num_micro_segments', 1))
    gap_s = float(candidate.get('internal_gap_s', 0.0))
    score_mean = float(candidate.get('cluster_score_mean', 0.0))
    if expected_len is None:
        # Broad prior: password bursts in our current scope are typically around 8-10 keys.
        len_match = math.exp(-abs(n_peaks - 9.0) / 4.5)
    else:
        len_match = math.exp(-abs(n_peaks - float(expected_len)) / max(2.0, 0.35 * max(float(expected_len), 1.0)))
    compactness = math.exp(-gap_s / 1.2)
    return 0.55 * len_match + 0.25 * compactness + 0.15 * score_mean + 0.05 * min(n_micro / 3.0, 1.0)


def _dedup_candidates(cands: list[dict]) -> list[dict]:
    ranked = sorted(
        cands,
        key=lambda c: (
            -int(c.get('num_micro_segments', 1)),
            abs(int(c.get('macro_num_peaks', 0)) - 9),
            int(c['crop_end']) - int(c['crop_start']),
        ),
    )
    kept = []
    for cand in ranked:
        if any(_candidate_iou(cand, old) >= 0.92 and abs(int(cand.get('macro_num_peaks', 0)) - int(old.get('macro_num_peaks', 0))) <= 1 for old in kept):
            continue
        kept.append(cand)
    kept.sort(key=lambda c: (int(c['crop_start']), int(c['crop_end'])))
    return kept


def _select_oldpool_union_candidates(
    base_cands: list[dict],
    union_cands: list[dict],
    expected_len: int,
    max_candidates_per_episode: int,
) -> list[dict]:
    """Keep a protected slice of strong base candidates, then add union rescues.

    In our real data, oldpool already contains excellent len9 candidates, while
    union candidates mainly rescue fragmented len8 sessions. Mixing them and
    truncating globally can evict the good oldpool entries, so we reserve budget
    for each source separately.
    """
    if max_candidates_per_episode <= 0:
        return []

    base_ranked = sorted(
        base_cands,
        key=lambda cand: _pre_score_candidate(cand, expected_len),
        reverse=True,
    )
    union_ranked = sorted(
        union_cands,
        key=lambda cand: _pre_score_candidate(cand, expected_len),
        reverse=True,
    )

    # Oldpool is already strong on many len9 sessions; unions are mainly a rescue
    # path for fragmented len8 cases. Keep oldpool dominant and only reserve a
    # small slice for unions.
    union_budget_cap = max(4, max_candidates_per_episode // 4)
    union_budget = min(len(union_ranked), union_budget_cap)
    base_budget = min(len(base_ranked), max_candidates_per_episode - union_budget)
    selected = list(base_ranked[:base_budget]) + list(union_ranked[:union_budget])

    if len(selected) < max_candidates_per_episode:
        spill = base_ranked[base_budget:] + union_ranked[union_budget:]
        selected.extend(spill[: max_candidates_per_episode - len(selected)])

    merged = _dedup_candidates(selected)
    if len(merged) <= max_candidates_per_episode:
        return merged

    merged_ranked = sorted(
        merged,
        key=lambda cand: _pre_score_candidate(cand, expected_len),
        reverse=True,
    )
    return merged_ranked[:max_candidates_per_episode]


def _select_primary_rescue_candidates(
    primary_cands: list[dict],
    rescue_cands: list[dict],
    expected_len: int,
    max_candidates_per_episode: int,
    rescue_fraction: float = 0.25,
    rescue_min: int = 4,
    rescue_iou_threshold: float = 0.40,
) -> list[dict]:
    """Keep the main proposer dominant while reserving a small rescue slice.

    `keynesspool` is the current primary Stage1/Stage2 direction, but our audit
    shows there are still a few sessions where the older coarse pool contains a
    much better full-burst candidate.  We therefore keep a protected rescue
    budget for the secondary source instead of mixing globally and letting the
    dominant source evict that rescue.
    """
    if max_candidates_per_episode <= 0:
        return []

    primary_ranked = sorted(
        primary_cands,
        key=lambda cand: _pre_score_candidate(cand, expected_len),
        reverse=True,
    )
    rescue_ranked = sorted(
        rescue_cands,
        key=lambda cand: _pre_score_candidate(cand, expected_len),
        reverse=True,
    )

    rescue_budget_cap = max(int(round(max_candidates_per_episode * rescue_fraction)), rescue_min)
    rescue_budget = min(len(rescue_ranked), rescue_budget_cap)

    rescue_selected: list[dict] = []
    rescue_spill: list[dict] = []
    for cand in rescue_ranked:
        if all(_candidate_iou(cand, kept) < rescue_iou_threshold for kept in rescue_selected):
            rescue_selected.append(cand)
        else:
            rescue_spill.append(cand)
        if len(rescue_selected) >= rescue_budget:
            break
    if len(rescue_selected) < rescue_budget:
        remaining_ranked = [cand for cand in rescue_ranked if cand not in rescue_selected and cand not in rescue_spill]
        rescue_spill.extend(remaining_ranked)
        rescue_selected.extend(rescue_spill[: rescue_budget - len(rescue_selected)])

    primary_budget = min(len(primary_ranked), max_candidates_per_episode - len(rescue_selected))
    selected = list(primary_ranked[:primary_budget]) + list(rescue_selected)

    if len(selected) < max_candidates_per_episode:
        spill = primary_ranked[primary_budget:] + rescue_spill
        selected.extend(spill[: max_candidates_per_episode - len(selected)])

    merged = _dedup_candidates(selected)
    if len(merged) <= max_candidates_per_episode:
        if len(merged) < max_candidates_per_episode:
            merged_keys = {
                (
                    int(c['crop_start']),
                    int(c['crop_end']),
                    int(c.get('macro_num_peaks', c.get('cluster_num_peaks', 0))),
                )
                for c in merged
            }
            spill_ranked = primary_ranked[primary_budget:] + rescue_spill
            for cand in spill_ranked:
                key = (
                    int(cand['crop_start']),
                    int(cand['crop_end']),
                    int(cand.get('macro_num_peaks', cand.get('cluster_num_peaks', 0))),
                )
                if key in merged_keys:
                    continue
                merged.append(cand)
                merged_keys.add(key)
                if len(merged) >= max_candidates_per_episode:
                    break
        return merged[:max_candidates_per_episode]

    merged_ranked = sorted(
        merged,
        key=lambda cand: _pre_score_candidate(cand, expected_len),
        reverse=True,
    )
    return merged_ranked[:max_candidates_per_episode]


def propose_candidates_fullstream_v2(imu: np.ndarray, ts: np.ndarray, sample_rate_hz: float) -> list[dict]:
    del ts
    energy = _proposal_energy_envelope(imu, sample_rate_hz)
    micro = _micro_segments_from_energy(energy, sample_rate_hz)
    cands = _enumerate_union_candidates(imu, energy, micro, sample_rate_hz)
    cands.extend(_legacy_cluster_candidates(imu, sample_rate_hz))
    cands = [c for c in cands if int(c['crop_end']) - int(c['crop_start']) >= max(8, int(round(sample_rate_hz * 0.70)))]
    cands = _dedup_candidates(cands)
    return cands[:80]


def _train_peak_model(train_peak_episodes):
    normalized = []
    for ep in train_peak_episodes:
        if isinstance(ep, dict):
            normalized.append(ep)
        else:
            normalized.append({
                'session_id': ep.session_id,
                'episode_id': ep.episode_id,
                'password': ep.password,
                'chars': ep.chars,
                'imu': ep.imu,
                'timestamps_ns': ep.timestamps_ns,
                'key_timestamps_ns': ep.key_timestamps_ns,
                'key_frames': ep.key_frames,
                'sample_rate_hz': ep.sample_rate_hz,
            })
    X, y, _ = _build_peak_dataset(normalized)
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=42,
    )
    model.fit(X, y)
    return model


def _match_count_with_tolerance(pred_ts_ns: np.ndarray, gt_ts_ns: np.ndarray, tol_ns: int) -> int:
    pred = np.asarray(np.sort(pred_ts_ns), dtype=np.int64)
    gt = np.asarray(np.sort(gt_ts_ns), dtype=np.int64)
    i = 0
    j = 0
    matched = 0
    while i < len(pred) and j < len(gt):
        d = int(pred[i]) - int(gt[j])
        if abs(d) <= tol_ns:
            matched += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1
    return matched


def _compute_local_activity(imu: np.ndarray, sr: float, short_win_s: float = 0.12) -> np.ndarray:
    win = max(3, int(round(sr * short_win_s)))
    activity = np.zeros(len(imu), dtype=np.float64)
    for ch in range(min(imu.shape[1], 6)):
        col = imu[:, ch].astype(np.float64)
        mean = uniform_filter1d(col, win)
        sq_mean = uniform_filter1d(col ** 2, win)
        activity += np.maximum(sq_mean - mean ** 2, 0.0)
    return activity


def _compute_rhythm_features(imu_seg: np.ndarray, sr: float) -> dict:
    result = {
        'acf_score': 0.0,
        'regularity': 0.0,
        'log_discreteness': 0.0,
        'iki_cv': 3.0,
        'n_peaks': 0,
        'peak_density': 0.0,
    }
    if len(imu_seg) < max(8, int(round(sr * 1.5))):
        return result

    activity = _compute_local_activity(imu_seg, sr)
    act_std = max(float(np.std(activity)), 1e-12)
    peaks, _ = find_peaks(
        activity,
        distance=max(3, int(round(sr * 0.40))),
        prominence=act_std * 0.20,
    )
    result['n_peaks'] = int(len(peaks))
    result['peak_density'] = float(len(peaks)) / max(len(imu_seg) / max(sr, 1e-6), 0.1)

    if len(peaks) >= 3:
        ikis = np.diff(peaks.astype(np.float64)) / max(sr, 1e-6)
        iki_mean = float(np.mean(ikis))
        iki_cv = float(np.std(ikis) / max(iki_mean, 1e-6))
        result['iki_cv'] = iki_cv
        result['regularity'] = float(np.exp(-iki_cv / 0.30))

    if len(peaks) >= 2:
        peak_vals = activity[peaks]
        guard = max(1, int(round(sr * 0.10)))
        trough_vals = []
        for i in range(len(peaks) - 1):
            lo = int(peaks[i]) + guard
            hi = int(peaks[i + 1]) - guard
            if hi > lo:
                trough_vals.append(float(np.mean(activity[lo:hi])))
        if trough_vals:
            discreteness = float(np.mean(peak_vals)) / max(float(np.mean(trough_vals)), 1e-12)
            result['log_discreteness'] = float(np.clip(np.log(max(discreteness, 1.0)) / 5.0, 0.0, 1.0))

    if len(activity) > int(round(sr * 3.0)):
        centered = activity - float(np.mean(activity))
        norm = float(np.sqrt(np.mean(centered ** 2)))
        if norm > 1e-12:
            centered = centered / norm
            n = len(centered)
            fft_val = rfft(centered, n=2 * n)
            acf = irfft(fft_val * np.conj(fft_val))[:n] / n
            min_lag = max(1, int(round(sr * 0.8)))
            max_lag = min(n - 1, int(round(sr * 2.5)))
            if max_lag > min_lag:
                acf_region = acf[min_lag:max_lag + 1]
                acf_peaks, _ = find_peaks(acf_region, prominence=0.02)
                if len(acf_peaks) > 0:
                    best_idx = acf_peaks[np.argmax(acf_region[acf_peaks])]
                    result['acf_score'] = float(acf_region[best_idx])
    return result


def _analyze_candidate(
    candidate: dict,
    full_imu: np.ndarray,
    full_ts: np.ndarray,
    sr: float,
    peak_model,
    expected_len: int,
    use_rhythm_aux: bool = True,
):
    lo = int(candidate['crop_start'])
    hi = int(candidate['crop_end'])
    crop_imu = full_imu[lo:hi]
    crop_ts = full_ts[lo:hi]
    energy = _proposal_energy_envelope(crop_imu, sr)
    sm = _smooth(energy, max(1, int(round(sr * 0.10))))
    peaks, sm2, _ = _propose_peaks({'imu': crop_imu, 'sample_rate_hz': sr, 'timestamps_ns': crop_ts})
    peak_probs = np.zeros(len(peaks), dtype=np.float32)
    if len(peaks):
        Xp = np.stack([_peak_feature_vector(sm2, peaks, i, sr) for i in range(len(peaks))]).astype(np.float32)
        peak_probs = peak_model.predict_proba(Xp)[:, 1].astype(np.float32)
    chosen = np.asarray([], dtype=np.int64)
    chosen_ts = np.asarray([], dtype=np.int64)
    if len(peaks):
        k = max(1, int(expected_len))
        chosen = _select_k_peaks(peaks, peak_probs, k, sr, gap_prior_s=1.3)
        chosen_global = chosen + lo
        chosen_ts = full_ts[np.clip(chosen_global, 0, len(full_ts) - 1)]

    local_energy = _normalized_curve(sm, out_len=TARGET_LEN)
    peak_profile = np.zeros(max(len(sm), 1), dtype=np.float32)
    if len(peaks):
        idx = np.clip(peaks.astype(np.int64), 0, len(peak_profile) - 1)
        np.maximum.at(peak_profile, idx, peak_probs.astype(np.float32))
    peak_profile = _normalized_curve(_smooth(peak_profile.astype(np.float64), max(1, len(peak_profile) // 32)), out_len=TARGET_LEN)

    sig = _resample_multichannel(crop_imu, out_len=TARGET_LEN)

    crop_span = max(1, hi - lo)
    ctx_pad = int(min(len(full_imu) * 0.25, max(round(crop_span * 1.6), round(sr * 1.5))))
    ctx_lo = max(0, lo - ctx_pad)
    ctx_hi = min(len(full_imu), hi + ctx_pad)
    ctx_energy = _proposal_energy_envelope(full_imu[ctx_lo:ctx_hi], sr)
    ctx_energy = _normalized_curve(_smooth(ctx_energy, max(1, int(round(sr * 0.10)))), out_len=TARGET_LEN)

    inside_mean = float(np.mean(sm)) if len(sm) else 0.0
    left = full_imu[max(0, lo - int(round(sr * 0.8))):lo]
    right = full_imu[hi:min(len(full_imu), hi + int(round(sr * 0.8)))]
    left_e = _proposal_energy_envelope(left, sr) if len(left) else np.asarray([], dtype=np.float64)
    right_e = _proposal_energy_envelope(right, sr) if len(right) else np.asarray([], dtype=np.float64)
    left_mean = float(np.mean(left_e)) if len(left_e) else 0.0
    right_mean = float(np.mean(right_e)) if len(right_e) else 0.0

    strong_count = int(np.sum(peak_probs >= 0.50)) if len(peak_probs) else 0
    topk = float(np.mean(np.sort(peak_probs)[-min(5, len(peak_probs)):])) if len(peak_probs) else 0.0
    reg_cv = 1.0
    if len(peaks) >= 3:
        gaps = np.diff(peaks.astype(np.float64)) / max(sr, 1e-6)
        reg_cv = float(np.std(gaps) / max(np.mean(gaps), 1e-6))
    rhythm = _compute_rhythm_features(crop_imu, sr) if use_rhythm_aux else {
        'acf_score': 0.0,
        'regularity': 0.0,
        'log_discreteness': 0.0,
        'iki_cv': 3.0,
    }

    aux = np.asarray([
        float(expected_len) / 12.0,
        float(candidate.get('macro_num_peaks', len(peaks))) / 20.0,
        float(strong_count) / 20.0,
        float(candidate.get('num_micro_segments', 1)) / 4.0,
        float(candidate.get('internal_gap_s', 0.0)) / 2.0,
        float(topk),
        float(np.mean(peak_probs >= 0.8)) if len(peak_probs) else 0.0,
        float(np.mean(peak_probs >= 0.5)) if len(peak_probs) else 0.0,
        float(reg_cv),
        left_mean / max(inside_mean, 1e-6),
        right_mean / max(inside_mean, 1e-6),
        abs(float(candidate.get('macro_num_peaks', len(peaks))) - float(expected_len)) / max(float(expected_len), 1.0),
        float(rhythm['acf_score']),
        float(rhythm['regularity']),
        float(rhythm['log_discreteness']),
        min(float(rhythm['iki_cv']), 3.0) / 3.0,
    ], dtype=np.float32)

    seq = np.concatenate([
        sig.T.astype(np.float32),
        local_energy[None, :].astype(np.float32),
        ctx_energy[None, :].astype(np.float32),
        peak_profile[None, :].astype(np.float32),
    ], axis=0)

    dbg = {
        'num_peaks_prop': int(len(peaks)),
        'num_strong_peaks': int(strong_count),
        'top5_peak_prob_mean': float(topk),
        'chosen_peak_ts_ns': chosen_ts.tolist(),
        'rhythm_acf_score': float(rhythm['acf_score']),
        'rhythm_regularity': float(rhythm['regularity']),
        'rhythm_log_discreteness': float(rhythm['log_discreteness']),
        'rhythm_iki_cv': float(rhythm['iki_cv']),
    }
    return {
        'seq': seq,
        'aux': aux,
        'crop_imu': crop_imu,
        'crop_ts': crop_ts,
        'chosen_ts_ns': chosen_ts,
        'num_prop_peaks': int(len(peaks)),
        'num_strong_peaks': int(strong_count),
        'rhythm': rhythm,
        'debug': dbg,
    }


def _candidate_recoverability_target(candidate: dict, analysis: dict, ep) -> tuple[float, dict]:
    gt_ts = np.asarray(ep.key_timestamps_ns, dtype=np.int64)
    gt_len = len(gt_ts)
    if gt_len == 0:
        return 0.0, {'reason': 'empty_gt'}

    start_ns = int(ep.timestamps_ns[max(int(ep.key_frames[0]), 0)])
    end_ns = int(ep.timestamps_ns[min(int(ep.key_frames[-1]), len(ep.timestamps_ns) - 1)])
    cand_start_ns = int(candidate['crop_start_ns'])
    cand_end_ns = int(candidate['crop_end_ns'])

    inside = int(np.sum((gt_ts >= cand_start_ns) & (gt_ts <= cand_end_ns)))
    key_recall = inside / max(gt_len, 1)

    gt_span = max(float(end_ns - start_ns) / 1e9, 1e-3)
    miss_left_s = max(0.0, (cand_start_ns - start_ns) * 1e-9)
    miss_right_s = max(0.0, (end_ns - cand_end_ns) * 1e-9)
    over_left_s = max(0.0, (start_ns - cand_start_ns) * 1e-9)
    over_right_s = max(0.0, (cand_end_ns - end_ns) * 1e-9)
    miss_penalty = (miss_left_s + miss_right_s) / gt_span
    over_penalty = (over_left_s + over_right_s) / gt_span

    tol_ns = int(round(0.18 * 1e9))
    matched = _match_count_with_tolerance(np.asarray(analysis['chosen_ts_ns'], dtype=np.int64), gt_ts, tol_ns=tol_ns)
    peak_recall = matched / max(gt_len, 1)

    count_score = math.exp(-abs(float(analysis['num_strong_peaks']) - float(gt_len)) / max(2.0, 0.35 * gt_len))
    boundary_score = math.exp(-0.40 * over_penalty) * math.exp(-1.20 * miss_penalty)

    utility = 0.65 * (peak_recall ** 2) + 0.20 * boundary_score + 0.15 * count_score
    utility = float(max(0.0, min(1.0, utility)))
    dbg = {
        'peak_recall': float(peak_recall),
        'key_recall': float(key_recall),
        'count_score': float(count_score),
        'boundary_score': float(boundary_score),
        'miss_penalty': float(miss_penalty),
        'over_penalty': float(over_penalty),
        'matched_peaks': int(matched),
        'utility': float(utility),
    }
    return utility, dbg


def _build_episode_bags(
    mixed_dirs: list[str],
    peak_model,
    length_model,
    candidate_mode: str = "oldpool",
    max_candidates_per_episode: int = 32,
    use_gt_len_hint: bool = True,
    use_rhythm_aux: bool = True,
    keyness_strong_threshold: float = 0.35,
    keyness_min_strong_peaks: int = 3,
) -> list[CandidateBag]:
    episodes = []
    for d in mixed_dirs:
        episodes.extend(build_password_episodes(d))
    by_session = {}
    for ep in episodes:
        by_session.setdefault(ep.session_id, []).append(ep)

    bags: list[CandidateBag] = []
    for session_id, session_eps in sorted(by_session.items()):
        loader = SessionLoader(session_eps[0].session_path)
        ts, imu = loader.get_imu()
        sr = estimate_sample_rate_hz(ts)
        for ep in session_eps:
            bag_len_hint: Optional[int] = len(ep.chars) if use_gt_len_hint else None
            if use_gt_len_hint and length_model is not None:
                try:
                    model, labels, meta = length_model
                    from onset_detection.stage2_segmental.length_model import compute_region_length_features
                    # Use a GT-centered coarse crop only during training/eval to keep parity with the current bundle.
                    lo = max(0, int(ep.key_frames[0]) - int(round(sr * 0.5)))
                    hi = min(len(ep.imu), int(ep.key_frames[-1]) + int(round(sr * 0.5)) + 1)
                    feat = compute_region_length_features(ep.imu[lo:hi], ep.timestamps_ns[lo:hi], feature_mode=str(meta.get('feature_mode', 'no_time'))).reshape(1, -1)
                    pred_len = int(model.predict(feat)[0])
                    bag_len_hint = max(4, pred_len)
                except Exception:
                    bag_len_hint = len(ep.chars)
            if candidate_mode == "v2":
                scored_cands = propose_candidates_fullstream_v2(imu, ts, sr)
                scored_cands = sorted(
                    scored_cands,
                    key=lambda cand: _pre_score_candidate(cand, bag_len_hint),
                    reverse=True,
                )[:max_candidates_per_episode]
            elif candidate_mode == "keynesspool":
                scored_cands = _keyness_candidates_fullstream(
                    imu, ts, sr, peak_model,
                    strong_threshold=keyness_strong_threshold,
                    min_strong_peaks=keyness_min_strong_peaks,
                )
                scored_cands = sorted(
                    scored_cands,
                    key=lambda cand: _pre_score_candidate(cand, bag_len_hint),
                    reverse=True,
                )[:max_candidates_per_episode]
            elif candidate_mode == "keynesspool_oldpool":
                keyness_cands = _keyness_candidates_fullstream(
                    imu, ts, sr, peak_model,
                    strong_threshold=keyness_strong_threshold,
                    min_strong_peaks=keyness_min_strong_peaks,
                )
                oldpool_cands = _propose_candidates_fullstream(imu, ts, sr)
                scored_cands = _select_primary_rescue_candidates(
                    keyness_cands,
                    oldpool_cands,
                    expected_len=int(bag_len_hint) if bag_len_hint is not None else 9,
                    max_candidates_per_episode=max_candidates_per_episode,
                )
            else:
                base_cands = _propose_candidates_fullstream(imu, ts, sr)
                if candidate_mode == "oldpool_union":
                    union_cands = _oldpool_union_candidates(base_cands, sr)
                    scored_cands = _select_oldpool_union_candidates(
                        base_cands,
                        union_cands,
                        expected_len=int(bag_len_hint) if bag_len_hint is not None else 9,
                        max_candidates_per_episode=max_candidates_per_episode,
                    )
                else:
                    scored_cands = sorted(
                        base_cands,
                        key=lambda cand: _pre_score_candidate(cand, bag_len_hint),
                        reverse=True,
                    )[:max_candidates_per_episode]

            if not scored_cands:
                continue

            xs_seq = []
            xs_aux = []
            ys = []
            cand_dbg = []
            kept = []
            for cand in scored_cands:
                lo = int(cand['crop_start'])
                hi = int(cand['crop_end'])
                if hi - lo < 8:
                    continue
                cand_len_hint = bag_len_hint
                if cand_len_hint is None:
                    fallback_len = int(np.clip(cand.get('macro_num_peaks', cand.get('cluster_num_peaks', 9)), 4, 16))
                    cand_len_hint, _cand_len_conf, _cand_len_dbg = _predict_length_with_debug(
                        length_model, imu[lo:hi], ts[lo:hi], fallback=fallback_len
                    )
                cand_ns = dict(cand)
                cand_ns['crop_start_ns'] = int(ts[min(max(lo, 0), len(ts) - 1)])
                cand_ns['crop_end_ns'] = int(ts[min(max(hi - 1, 0), len(ts) - 1)])
                if use_gt_len_hint:
                    analysis = _analyze_candidate(
                        cand_ns, imu, ts, sr, peak_model,
                        expected_len=int(cand_len_hint),
                        use_rhythm_aux=use_rhythm_aux,
                    )
                    utility, dbg = _candidate_recoverability_target(cand_ns, analysis, ep)
                else:
                    k_center = int(np.clip(cand_len_hint, 4, 16))
                    k_grid = sorted({
                        max(4, len(ep.chars) - 1),
                        len(ep.chars),
                        min(16, len(ep.chars) + 1),
                        max(4, k_center - 1),
                        k_center,
                        min(16, k_center + 1),
                    })
                    best_utility = -1.0
                    best_analysis = None
                    best_dbg = None
                    for k_try in k_grid:
                        analysis_try = _analyze_candidate(
                            cand_ns, imu, ts, sr, peak_model,
                            expected_len=int(k_try),
                            use_rhythm_aux=use_rhythm_aux,
                        )
                        utility_try, dbg_try = _candidate_recoverability_target(cand_ns, analysis_try, ep)
                        if utility_try > best_utility:
                            best_utility = float(utility_try)
                            best_analysis = analysis_try
                            best_dbg = dbg_try
                    analysis = best_analysis
                    utility = float(best_utility)
                    dbg = best_dbg
                    if analysis is None or dbg is None:
                        continue
                xs_seq.append(analysis['seq'])
                xs_aux.append(analysis['aux'])
                ys.append(utility)
                kept.append(cand_ns)
                cand_dbg.append({**dbg, **analysis['debug']})
            if not xs_seq:
                continue
            bags.append(CandidateBag(
                session_id=session_id,
                episode_id=ep.episode_id,
                expected_len=int(bag_len_hint) if bag_len_hint is not None else len(ep.chars),
                xs_seq=np.stack(xs_seq).astype(np.float32),
                xs_aux=np.stack(xs_aux).astype(np.float32),
                targets=np.asarray(ys, dtype=np.float32),
                candidates=kept,
                target_debug=cand_dbg,
            ))
    return bags


def _split_bags_by_session(bags: list[CandidateBag], train_ratio: float = 0.8):
    session_ids = sorted({b.session_id for b in bags})
    rng.shuffle(session_ids)
    split = int(round(len(session_ids) * train_ratio))
    split = min(max(split, 1), max(1, len(session_ids) - 1))
    train_sessions = set(session_ids[:split])
    train = [b for b in bags if b.session_id in train_sessions]
    val = [b for b in bags if b.session_id not in train_sessions]
    if not val:
        val = train[-1:]
        train = train[:-1] or train
    return train, val


def _soft_target(y: np.ndarray, tau: float = 0.10) -> np.ndarray:
    z = np.asarray(y, dtype=np.float64) / max(tau, 1e-6)
    z = z - np.max(z)
    p = np.exp(z)
    p = p / max(np.sum(p), 1e-8)
    return p.astype(np.float32)


def _pairwise_margin_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    best_idx = int(torch.argmax(targets).item())
    s_best = scores[best_idx]
    t_best = targets[best_idx]
    margins = 0.10 + 0.35 * torch.clamp(t_best - targets, min=0.0)
    raw = F.relu(margins - s_best + scores)
    mask = torch.ones_like(raw)
    mask[best_idx] = 0.0
    denom = torch.clamp(torch.sum(mask), min=1.0)
    return torch.sum(raw * mask) / denom


def train_bag_ranker(train_bags: list[CandidateBag], val_bags: list[CandidateBag], device: torch.device):
    in_ch = int(train_bags[0].xs_seq.shape[1])
    aux_dim = int(train_bags[0].xs_aux.shape[1])
    model = ContextRankCNN(in_ch=in_ch, aux_dim=aux_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = -1e18

    for _epoch in range(20):
        model.train()
        train_order = list(range(len(train_bags)))
        rng.shuffle(train_order)
        for idx in train_order:
            bag = train_bags[idx]
            x_seq = torch.tensor(bag.xs_seq, dtype=torch.float32, device=device)
            x_aux = torch.tensor(bag.xs_aux, dtype=torch.float32, device=device)
            targets = torch.tensor(bag.targets, dtype=torch.float32, device=device)
            teacher = torch.tensor(_soft_target(bag.targets), dtype=torch.float32, device=device)
            scores = model(x_seq, x_aux)
            loss_rank = F.kl_div(F.log_softmax(scores, dim=0), teacher, reduction='batchmean')
            loss_margin = _pairwise_margin_loss(scores, targets)
            loss = loss_rank + 0.35 * loss_margin
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        val_metric = 0.0
        with torch.no_grad():
            for bag in val_bags:
                x_seq = torch.tensor(bag.xs_seq, dtype=torch.float32, device=device)
                x_aux = torch.tensor(bag.xs_aux, dtype=torch.float32, device=device)
                scores = model(x_seq, x_aux).detach().cpu().numpy()
                pred_idx = int(np.argmax(scores))
                oracle_idx = int(np.argmax(bag.targets))
                val_metric += float(bag.targets[pred_idx])
                val_metric += 0.20 if pred_idx == oracle_idx else 0.0
        val_metric /= max(len(val_bags), 1)
        if val_metric > best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, float(best_val)


def _score_bag(model, bag: CandidateBag, device: torch.device) -> np.ndarray:
    x_seq = torch.tensor(bag.xs_seq, dtype=torch.float32, device=device)
    x_aux = torch.tensor(bag.xs_aux, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = model(x_seq, x_aux).detach().cpu().numpy()
    return scores


def _recover_with_peak_model(candidate, full_imu, full_ts, sr, peak_model, classifier, overlap, device, ep_eval, expected_len: Optional[int] = None):
    lo = int(candidate['crop_start'])
    hi = int(candidate['crop_end'])
    crop_imu = full_imu[lo:hi]
    crop_ts = full_ts[lo:hi]
    peaks, sm, _ = _propose_peaks({'imu': crop_imu, 'sample_rate_hz': sr, 'timestamps_ns': crop_ts})
    if len(peaks) == 0:
        return None, None, {'error': 'no_peaks'}
    Xp = np.stack([_peak_feature_vector(sm, peaks, i, sr) for i in range(len(peaks))]).astype(np.float32)
    peak_probs = peak_model.predict_proba(Xp)[:, 1].astype(np.float32)
    if expected_len is None:
        k_grid = [8, 9, 10]
    else:
        center = int(expected_len)
        k_grid = sorted({max(4, center - 1), center, min(16, center + 1)})

    best = None
    for k in k_grid:
        chosen = _select_k_peaks(peaks, peak_probs, k, sr, gap_prior_s=1.3)
        pred_global_frames = chosen + lo
        pred_global_ts = full_ts[np.clip(pred_global_frames, 0, len(full_ts) - 1)]
        local_frames = np.searchsorted(ep_eval.timestamps_ns, pred_global_ts, side='left')
        local_frames = np.clip(local_frames, 0, len(ep_eval.timestamps_ns) - 1).astype(np.int64)
        if len(local_frames) != len(ep_eval.chars):
            xs = np.linspace(0, len(local_frames) - 1, len(ep_eval.chars))
            local_frames = np.interp(xs, np.arange(len(local_frames)), local_frames.astype(np.float64)).round().astype(np.int64)
        base = _evaluate_fixed_from_frames(classifier, ep_eval, local_frames)
        ov, ov_dbg = _evaluate_overlap_from_frames(overlap, ep_eval, local_frames, device)
        if base is None or ov is None:
            continue
        conf = _fixed_classifier_confidence(classifier, ep_eval, local_frames)
        cand = {
            'base': base,
            'ov': ov,
            'dbg': {
                'mapped_local_frames': local_frames.tolist(),
                'chosen_local_frames': chosen.tolist(),
                'peak_probs_top': np.sort(peak_probs)[-min(8, len(peak_probs)):].tolist(),
                'overlap_debug': ov_dbg,
                'selected_k': int(k),
            },
            'conf': float(conf),
        }
        if best is None or cand['conf'] > best['conf']:
            best = cand
    if best is None:
        return None, None, {'error': 'no_valid_k'}
    return best['base'], best['ov'], {**best['dbg'], 'classifier_conf': best['conf']}


def _fixed_classifier_confidence(classifier, ep, local_frames: np.ndarray) -> float:
    windows = []
    for frame in local_frames.tolist():
        win = _extract_window_from_signal(ep.imu, int(frame), ep.sample_rate_hz, classifier.target_len)
        if win is None:
            return 0.0
        windows.append(win)
    xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=next(classifier.parameters()).device)
    with torch.no_grad():
        probs = torch.softmax(classifier(xb), dim=1).cpu().numpy()
    max_prob = probs.max(axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    margin = np.clip(max_prob - top2, 0.0, 1.0)
    return float(np.mean(max_prob) + 0.20 * np.mean(margin))


def _candidate_classifier_confidence(classifier, crop_imu: np.ndarray, sample_rate_hz: float, local_frames: np.ndarray) -> float:
    windows = []
    for frame in local_frames.tolist():
        win = _extract_window_from_signal(crop_imu, int(frame), sample_rate_hz, classifier.target_len)
        if win is None:
            return 0.0
        windows.append(win)
    xb = torch.tensor(np.stack(windows), dtype=torch.float32, device=next(classifier.parameters()).device)
    with torch.no_grad():
        probs = torch.softmax(classifier(xb), dim=1).cpu().numpy()
    max_prob = probs.max(axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2]
    margin = np.clip(max_prob - top2, 0.0, 1.0)
    return float(np.mean(max_prob) + 0.20 * np.mean(margin))


def _frame_regularity_score(frames: np.ndarray) -> float:
    if len(frames) < 3:
        return 0.0
    frames = np.asarray(frames, dtype=np.float64)
    gaps = np.diff(frames)
    mean_gap = float(np.mean(gaps))
    if mean_gap <= 1e-6:
        return 0.0
    cv = float(np.std(gaps) / mean_gap)
    return float(np.exp(-cv / 0.6))


def _oracle_summary(bags: list[CandidateBag]) -> dict:
    if not bags:
        return {'num_bags': 0}
    best = np.asarray([float(np.max(b.targets)) for b in bags], dtype=np.float64)
    good = np.asarray([float(np.max(b.targets) >= 0.90) for b in bags], dtype=np.float64)
    usable = np.asarray([float(np.max(b.targets) >= 0.75) for b in bags], dtype=np.float64)
    return {
        'num_bags': int(len(bags)),
        'mean_best_target': float(np.mean(best)),
        'bags_with_candidate_ge_0.90': float(np.mean(good)),
        'bags_with_candidate_ge_0.75': float(np.mean(usable)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--password_dirs', nargs='*', default=[])
    ap.add_argument('--train_mixed_dirs', nargs='+', required=True)
    ap.add_argument('--eval_dirs', nargs='+', required=True)
    ap.add_argument('--classifier_checkpoint', required=True)
    ap.add_argument('--classifier_scaler', required=True)
    ap.add_argument('--overlap_checkpoint', required=True)
    ap.add_argument('--length_model', default='')
    ap.add_argument('--candidate_mode', choices=['oldpool', 'oldpool_union', 'v2', 'keynesspool', 'keynesspool_oldpool'], default='oldpool')
    ap.add_argument('--max_candidates_per_episode', type=int, default=32)
    ap.add_argument('--disable_rhythm_aux', action='store_true')
    ap.add_argument('--keyness_strong_threshold', type=float, default=0.35)
    ap.add_argument('--keyness_min_strong_peaks', type=int, default=3)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    peak_train_eps = []
    for d in args.password_dirs:
        peak_train_eps.extend(_load_password_attempt_episodes(d))
    for d in args.train_mixed_dirs:
        peak_train_eps.extend(build_password_episodes(d))
    peak_model = _train_peak_model(peak_train_eps)

    length_model = load_length_model(args.length_model) if args.length_model else None

    train_bags_all = _build_episode_bags(
        args.train_mixed_dirs,
        peak_model=peak_model,
        length_model=length_model,
        candidate_mode=args.candidate_mode,
        max_candidates_per_episode=args.max_candidates_per_episode,
        use_gt_len_hint=False,
        use_rhythm_aux=not args.disable_rhythm_aux,
        keyness_strong_threshold=args.keyness_strong_threshold,
        keyness_min_strong_peaks=args.keyness_min_strong_peaks,
    )
    train_bags, val_bags = _split_bags_by_session(train_bags_all, train_ratio=0.8)
    model, best_val = train_bag_ranker(train_bags, val_bags, device)

    classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    classifier.eval()
    overlap = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap.eval()
    overlap.freeze_classifier(True)

    eval_bags = _build_episode_bags(
        args.eval_dirs,
        peak_model=peak_model,
        length_model=length_model,
        candidate_mode=args.candidate_mode,
        max_candidates_per_episode=args.max_candidates_per_episode,
        use_gt_len_hint=False,
        use_rhythm_aux=not args.disable_rhythm_aux,
        keyness_strong_threshold=args.keyness_strong_threshold,
        keyness_min_strong_peaks=args.keyness_min_strong_peaks,
    )
    eval_bag_map = {(b.session_id, b.episode_id): b for b in eval_bags}

    eval_eps = []
    for d in args.eval_dirs:
        eval_eps.extend(build_password_episodes(d))

    baseline_rows = []
    overlap_rows = []
    debug_rows = []
    for ep in eval_eps:
        bag = eval_bag_map.get((ep.session_id, ep.episode_id))
        if bag is None:
            debug_rows.append({'session_id': ep.session_id, 'episode_id': ep.episode_id, 'error': 'no_bag'})
            continue
        loader = SessionLoader(ep.session_path)
        ts, imu = loader.get_imu()
        sr = estimate_sample_rate_hz(ts)

        scores = _score_bag(model, bag, device)
        order = np.argsort(scores)[::-1]
        top_candidates = []
        for idx in order[:5]:
            cand = bag.candidates[int(idx)]
            top_candidates.append({
                'rank_score': float(scores[int(idx)]),
                'target_utility': float(bag.targets[int(idx)]),
                'crop_start': int(cand['crop_start']),
                'crop_end': int(cand['crop_end']),
                'macro_num_peaks': int(cand.get('macro_num_peaks', 0)),
                'source': cand.get('source', ''),
                **bag.target_debug[int(idx)],
            })

        if len(order) == 0:
            debug_rows.append({'session_id': ep.session_id, 'episode_id': ep.episode_id, 'error': 'no_candidates'})
            continue
        best_idx = -1
        best_proxy = -1e18
        best_cand = None
        best_base = None
        best_ov = None
        best_dbg = None
        rerank_rows = []
        for idx in order[: min(5, len(order))]:
            cand = bag.candidates[int(idx)]
            lo = int(cand['crop_start'])
            hi = int(cand['crop_end'])
            crop_imu = imu[lo:hi]
            crop_ts = ts[lo:hi]
            fallback_len = int(np.clip(cand.get('macro_num_peaks', bag.expected_len), 4, 16))
            recover_len = _predict_length_from_crop(length_model, crop_imu, crop_ts, fallback=fallback_len)
            base, ov, rec_dbg = _recover_with_peak_model(
                cand, imu, ts, sr, peak_model, classifier, overlap, device, ep, expected_len=recover_len
            )
            if base is None or ov is None:
                continue
            cand_frames = np.asarray(rec_dbg.get('chosen_local_frames', []), dtype=np.int64)
            cls_conf = _candidate_classifier_confidence(classifier, crop_imu, sr, cand_frames)
            peak_conf = float(np.mean(rec_dbg.get('peak_probs_top', [])[-min(5, len(rec_dbg.get('peak_probs_top', []))):])) if rec_dbg.get('peak_probs_top') else 0.0
            reg_conf = _frame_regularity_score(cand_frames)
            selected_k = int(rec_dbg.get('selected_k', recover_len))
            chosen_count = int(len(cand_frames))
            count_consistency = math.exp(-abs(chosen_count - selected_k) / max(1.5, 0.25 * max(selected_k, 1)))
            proxy = 0.40 * cls_conf + 0.15 * peak_conf + 0.20 * reg_conf + 0.25 * count_consistency
            rerank_rows.append({
                'idx': int(idx),
                'proxy_score': float(proxy),
                'cls_conf': float(cls_conf),
                'peak_conf': float(peak_conf),
                'reg_conf': float(reg_conf),
                'count_consistency': float(count_consistency),
                'chosen_count': int(chosen_count),
                'selected_k': int(selected_k),
                'recover_len': int(recover_len),
            })
            if proxy > best_proxy:
                best_proxy = proxy
                best_idx = int(idx)
                best_cand = cand
                best_base = base
                best_ov = ov
                best_dbg = rec_dbg

        if best_idx < 0 or best_cand is None or best_base is None or best_ov is None or best_dbg is None:
            debug_rows.append({'session_id': ep.session_id, 'episode_id': ep.episode_id, 'error': 'recover_failed', 'top_candidates': top_candidates})
            continue
        best_base['session_id'] = ep.session_id
        best_base['episode_id'] = ep.episode_id
        best_ov['session_id'] = ep.session_id
        best_ov['episode_id'] = ep.episode_id
        baseline_rows.append(best_base)
        overlap_rows.append(best_ov)
        debug_rows.append({
            'session_id': ep.session_id,
            'episode_id': ep.episode_id,
            'expected_len_used': int(rerank_rows[0]['recover_len']) if rerank_rows else int(bag.expected_len),
            'selected_rank_score': float(scores[best_idx]),
            'selected_target_utility': float(bag.targets[best_idx]),
            'selected_proxy_score': float(best_proxy),
            'selected_candidate': {
                'crop_start': int(best_cand['crop_start']),
                'crop_end': int(best_cand['crop_end']),
                'macro_num_peaks': int(best_cand.get('macro_num_peaks', 0)),
                'source': best_cand.get('source', ''),
            },
            'top_candidates': top_candidates,
            'rerank_rows': rerank_rows,
            **best_dbg,
        })

    report = {
        'mode': 'segment_bagrank_context_v2',
        'candidate_mode': args.candidate_mode,
        'use_rhythm_aux': bool(not args.disable_rhythm_aux),
        'keyness_strong_threshold': float(args.keyness_strong_threshold),
        'keyness_min_strong_peaks': int(args.keyness_min_strong_peaks),
        'train_summary': {
            'num_train_bags': len(train_bags),
            'num_val_bags': len(val_bags),
            'best_val_metric': float(best_val),
            'train_oracle': _oracle_summary(train_bags),
            'val_oracle': _oracle_summary(val_bags),
            'eval_oracle': _oracle_summary(eval_bags),
        },
        'baseline_fixed_window': aggregate_episode_results(baseline_rows),
        'overlap_refine': aggregate_episode_results(overlap_rows),
    }
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'debug_rows.json').write_text(json.dumps(debug_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
