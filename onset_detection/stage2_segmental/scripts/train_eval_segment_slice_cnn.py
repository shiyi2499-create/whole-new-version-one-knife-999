#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
from scipy.signal import resample
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
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
from onset_detection.stage2_segmental.length_model import load_length_model
from onset_detection.stage2_segmental.metrics import aggregate_episode_results
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.scripts.eval_fullstream_clusterfirst import _evaluate_fixed_from_frames, _evaluate_overlap_from_frames
from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import (
    _build_dataset as _build_peak_dataset,
    _load_password_attempt_episodes,
    _peak_feature_vector,
    _propose_peaks,
    _select_k_peaks,
    _smooth,
)

TARGET_LEN = 256
SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class SegmentExample:
    x: np.ndarray
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
        return torch.tensor(it.x, dtype=torch.float32), torch.tensor(it.y, dtype=torch.long)


class SegmentCNN(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(96), nn.MaxPool1d(2),
            nn.Conv1d(96, 128, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(128), nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def forward(self, x):
        return self.head(self.net(x))


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


def _resample_segment(imu: np.ndarray, sr: float, target_len: int = TARGET_LEN) -> np.ndarray:
    arr = np.asarray(imu, dtype=np.float64)
    if len(arr) < 4:
        arr = np.pad(arr, ((0, max(0, 4 - len(arr))), (0, 0)), mode='edge')
    energy = _compute_energy_envelope(arr.astype(np.float32), int(round(sr))).astype(np.float64)
    energy = _smooth(energy, max(1, int(round(sr * 0.08))))
    em = resample(energy[:, None], target_len, axis=0)
    sig = resample(arr, target_len, axis=0)
    if np.iscomplexobj(sig): sig = np.real(sig)
    if np.iscomplexobj(em): em = np.real(em)
    sig = np.asarray(sig, dtype=np.float32)
    em = np.asarray(em, dtype=np.float32)
    sig = sig - sig.mean(axis=0, keepdims=True)
    sig = sig / np.maximum(sig.std(axis=0, keepdims=True), 1e-6)
    em = em - em.mean(axis=0, keepdims=True)
    em = em / np.maximum(em.std(axis=0, keepdims=True), 1e-6)
    x = np.concatenate([sig, em], axis=1)  # [T,7]
    return x.T.astype(np.float32)  # [C,T]


def _load_positive_password_examples(password_dirs: list[str]) -> list[SegmentExample]:
    items = []
    for d in password_dirs:
        for ep in _load_password_attempt_episodes(d):
            x = _resample_segment(ep['imu'], ep['sample_rate_hz'])
            items.append(SegmentExample(x=x, y=1, session_id=ep['session_id'], source='password', meta={'episode_id': ep['episode_id'], 'password': ep['password']}))
    return items


def _load_positive_mixed_password_examples(mixed_dirs: list[str]) -> list[SegmentExample]:
    items = []
    for d in mixed_dirs:
        for ep in build_password_episodes(d):
            x = _resample_segment(ep.imu, ep.sample_rate_hz)
            items.append(SegmentExample(x=x, y=1, session_id=ep.session_id, source='mixed_password', meta={'episode_id': ep.episode_id, 'password': ep.password}))
    return items


def _extract_segment_by_ns(ts: np.ndarray, imu: np.ndarray, start_ns: int, end_ns: int):
    idx = np.where((ts >= start_ns) & (ts <= end_ns))[0]
    if len(idx) < 8:
        return None
    return imu[idx], ts[idx]


def _load_mixed_negative_examples(mixed_dirs: list[str], max_per_session: int = 8) -> list[SegmentExample]:
    items = []
    for d in mixed_dirs:
        for session_path in sorted(Path(d).glob('*_sensor.csv')):
            prefix = str(session_path)[:-11]
            loader = SessionLoader(prefix)
            ts, imu = loader.get_imu()
            sr = estimate_sample_rate_hz(ts)
            rows = loader.get_activity_log()
            session_id = Path(prefix).name
            cands = []
            for row in rows:
                label = str(row.get('label', ''))
                typing_style = str(row.get('typing_style', ''))
                activity = str(row.get('activity', ''))
                if typing_style == 'password' or label.startswith('typing_2'):
                    continue
                start_ns = int(row.get('start_ns', row.get('start_time_ns', 0)))
                end_ns = int(row.get('end_ns', row.get('end_time_ns', 0)))
                seg = _extract_segment_by_ns(ts, imu, start_ns, end_ns)
                if seg is None:
                    continue
                seg_imu, seg_ts = seg
                cands.append(SegmentExample(
                    x=_resample_segment(seg_imu, sr),
                    y=0,
                    session_id=session_id,
                    source='mixed_negative',
                    meta={'label': label, 'activity': activity, 'typing_style': typing_style}
                ))
            rng.shuffle(cands)
            items.extend(cands[:max_per_session])
    return items


def _load_mixed_hard_negative_examples(mixed_dirs: list[str], per_episode: int = 2) -> list[SegmentExample]:
    items = []
    for d in mixed_dirs:
        by_session = {}
        for ep in build_password_episodes(d):
            by_session.setdefault(ep.session_id, []).append(ep)
        for session_id, eps in by_session.items():
            loader = SessionLoader(eps[0].session_path)
            ts, imu = loader.get_imu()
            sr = estimate_sample_rate_hz(ts)
            n = len(imu)
            cands = []
            for ep in eps:
                key_start = int(ep.key_frames[0])
                key_end = int(ep.key_frames[-1])
                pw_span = max(8, key_end - key_start + 1)
                pad = int(round(sr * 0.8))
                crop_span = pw_span + 2 * pad
                shifts = [
                    -int(round(1.4 * crop_span)),
                    -int(round(0.9 * crop_span)),
                    int(round(0.9 * crop_span)),
                    int(round(1.4 * crop_span)),
                ]
                gt_lo = max(0, key_start - pad)
                gt_hi = min(n, key_end + pad + 1)
                for sh in shifts:
                    center = (gt_lo + gt_hi) // 2 + sh
                    lo = max(0, center - crop_span // 2)
                    hi = min(n, lo + crop_span)
                    lo = max(0, hi - crop_span)
                    if hi - lo < 16:
                        continue
                    # Skip windows that still overlap the true password body too much.
                    inter = max(0, min(hi, gt_hi) - max(lo, gt_lo))
                    if inter / max(1, (hi - lo)) > 0.20:
                        continue
                    cands.append(SegmentExample(
                        x=_resample_segment(imu[lo:hi], sr),
                        y=0,
                        session_id=session_id,
                        source='mixed_hard_negative',
                        meta={'episode_id': ep.episode_id, 'lo': int(lo), 'hi': int(hi)}
                    ))
            rng.shuffle(cands)
            items.extend(cands[: max(1, per_episode * max(1, len(eps)))])
    return items


def _load_onset_negative_examples(onset_negative_root: str, target_count: int, duration_s_range=(1.2, 6.0)) -> list[SegmentExample]:
    root = Path(onset_negative_root)
    sensor_files = sorted(root.glob('*/*_sensor.csv'))
    items = []
    for sensor_path in sensor_files:
        prefix = str(sensor_path)[:-11]
        meta_path = Path(prefix + '_meta.json')
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        rows = []
        with open(sensor_path, 'r') as f:
            for row in csv.DictReader(f):
                rows.append([
                    int(row['timestamp_ns']),
                    float(row['accel_x']), float(row['accel_y']), float(row['accel_z']),
                    float(row['gyro_x']), float(row['gyro_y']), float(row['gyro_z']),
                ])
        arr = np.asarray(rows, dtype=np.float64)
        if len(arr) < 20:
            continue
        ts = arr[:,0].astype(np.int64)
        imu = arr[:,1:].astype(np.float32)
        sr = estimate_sample_rate_hz(ts)
        total_s = (ts[-1] - ts[0]) * 1e-9
        if total_s < duration_s_range[0]:
            continue
        n_draw = 1 if total_s < 20 else 2
        for _ in range(n_draw):
            dur_s = rng.uniform(*duration_s_range)
            span = int(round(dur_s * sr))
            if span >= len(imu):
                continue
            start = rng.randint(0, max(0, len(imu) - span - 1))
            seg_imu = imu[start:start+span]
            items.append(SegmentExample(
                x=_resample_segment(seg_imu, sr),
                y=0,
                session_id=str(meta.get('session_id', Path(prefix).name)),
                source='onset_negative',
                meta={'activity': meta.get('activity', 'negative')}
            ))
            if len(items) >= target_count:
                return items
    return items


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
    model = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=2, class_weight='balanced_subsample', random_state=42)
    model.fit(X, y)
    return model


def _score_candidates_segment_cnn(model, candidates, imu, ts, sr, device):
    if not candidates:
        return []
    xs = []
    for cand in candidates:
        crop = imu[int(cand['crop_start']):int(cand['crop_end'])]
        xs.append(_resample_segment(crop, sr))
    xb = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(xb), dim=1)[:,1].detach().cpu().numpy()
    return probs.tolist()


def _cluster_macro_peaks(peaks: np.ndarray, scores: np.ndarray, sample_rate: float, gap_s: float):
    if len(peaks) == 0:
        return []
    max_gap_frames = max(1, int(round(sample_rate * gap_s)))
    groups=[]; cur=[0]
    for i in range(1,len(peaks)):
        if int(peaks[i]) - int(peaks[i-1]) <= max_gap_frames:
            cur.append(i)
        else:
            groups.append(cur); cur=[i]
    groups.append(cur)
    out=[]
    for g in groups:
        idx=np.asarray(g, dtype=np.int64)
        p=peaks[idx]; s=scores[idx]
        out.append({'start_frame': int(p[0]), 'end_frame': int(p[-1]), 'num_peaks': int(len(p)), 'score_sum': float(np.sum(s)), 'score_mean': float(np.mean(s)), 'peaks': p.astype(np.int64), 'scores': s.astype(np.float64)})
    return out


def _propose_candidates_fullstream(imu: np.ndarray, sr: float):
    energy_raw = _compute_energy_envelope(imu, int(round(sr))).astype(np.float64)
    seen=set(); candidates=[]
    for smooth_s in (0.10,0.15,0.22,0.30,0.40):
        smoothed = _smooth(energy_raw, max(1, int(round(sr*smooth_s))))
        q50,q90,q98=np.quantile(smoothed,[0.50,0.90,0.98])
        prominence=max(1e-6,(q90-q50)*0.08)
        height=q50+(q98-q50)*0.03
        for dist_s in (0.25,0.35,0.50,0.70,0.90):
            peaks, props = find_peaks(smoothed, distance=max(1,int(round(sr*dist_s))), prominence=prominence, height=height)
            if len(peaks)==0:
                continue
            heights=np.asarray(props.get('peak_heights', smoothed[peaks]), dtype=np.float64)
            peak_scores=heights / max(float(np.max(heights)), 1e-8)
            for gap_s in (0.85,1.1,1.35,1.7,2.1):
                for cluster in _cluster_macro_peaks(peaks, peak_scores, sr, gap_s=gap_s):
                    if not (3 <= cluster['num_peaks'] <= 18):
                        continue
                    pad_frames=int(round(sr*0.80))
                    lo=max(0,int(cluster['start_frame'])-pad_frames)
                    hi=min(len(imu), int(cluster['end_frame'])+pad_frames+1)
                    key=(lo,hi,cluster['num_peaks'])
                    if key in seen:
                        continue
                    seen.add(key)
                    cand=dict(cluster)
                    cand['crop_start']=int(lo); cand['crop_end']=int(hi)
                    candidates.append(cand)
    return candidates


def _recover_with_peak_model(candidate, full_imu, full_ts, sr, peak_model, classifier, overlap, device, ep_eval):
    lo=int(candidate['crop_start']); hi=int(candidate['crop_end'])
    crop_imu = full_imu[lo:hi]
    crop_ts = full_ts[lo:hi]
    peaks, sm, _ = _propose_peaks({'imu': crop_imu, 'sample_rate_hz': sr, 'timestamps_ns': crop_ts})
    if len(peaks)==0:
        return None, None, {'error':'no_peaks'}
    Xp = np.stack([_peak_feature_vector(sm, peaks, i, sr) for i in range(len(peaks))]).astype(np.float32)
    peak_probs = peak_model.predict_proba(Xp)[:,1].astype(np.float32)
    k = len(ep_eval.chars)
    chosen = _select_k_peaks(peaks, peak_probs, k, sr, gap_prior_s=1.3)
    pred_global_frames = chosen + lo
    pred_global_ts = full_ts[np.clip(pred_global_frames, 0, len(full_ts)-1)]
    local_frames = np.searchsorted(ep_eval.timestamps_ns, pred_global_ts, side='left')
    local_frames = np.clip(local_frames, 0, len(ep_eval.timestamps_ns)-1).astype(np.int64)
    if len(local_frames) != len(ep_eval.chars):
        xs = np.linspace(0, len(local_frames)-1, len(ep_eval.chars))
        local_frames = np.interp(xs, np.arange(len(local_frames)), local_frames.astype(np.float64)).round().astype(np.int64)
    base = _evaluate_fixed_from_frames(classifier, ep_eval, local_frames)
    ov, ov_dbg = _evaluate_overlap_from_frames(overlap, ep_eval, local_frames, device)
    return base, ov, {'mapped_local_frames': local_frames.tolist(), 'peak_probs_top': np.sort(peak_probs)[-min(8,len(peak_probs)):].tolist(), 'overlap_debug': ov_dbg}


def _group_split_items(items: List[SegmentExample], train_ratio: float = 0.85):
    groups = {}
    for it in items:
        key = (it.source, it.session_id)
        groups.setdefault(key, []).append(it)
    keys = list(groups.keys())
    rng.shuffle(keys)
    split = int(round(len(keys) * train_ratio))
    if split <= 0:
        split = max(1, len(keys) - 1)
    if split >= len(keys):
        split = max(1, len(keys) - 1)
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
    model = SegmentCNN(in_ch=train_items[0].x.shape[0]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state=None; best_val=-1.0
    train_loader = DataLoader(SegmentDataset(train_items), batch_size=64, shuffle=True)
    val_loader = DataLoader(SegmentDataset(val_items), batch_size=128, shuffle=False) if val_items else None
    for epoch in range(18):
        model.train()
        for xb, yb in train_loader:
            xb=xb.to(device); yb=yb.to(device)
            logits=model(xb)
            cls_w = torch.tensor([1.0, 1.2], dtype=torch.float32, device=device)
            loss=F.cross_entropy(logits, yb, weight=cls_w)
            opt.zero_grad(); loss.backward(); opt.step()
        if val_loader is not None:
            model.eval(); correct=0; total=0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb=xb.to(device); yb=yb.to(device)
                    pred=model(xb).argmax(dim=1)
                    correct += int((pred==yb).sum())
                    total += int(len(yb))
            acc = correct/max(total,1)
            if acc > best_val:
                best_val=acc; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--password_dirs', nargs='+', required=True)
    ap.add_argument('--train_mixed_dirs', nargs='+', required=True)
    ap.add_argument('--eval_dirs', nargs='+', required=True)
    ap.add_argument('--onset_negative_root', required=True)
    ap.add_argument('--classifier_checkpoint', required=True)
    ap.add_argument('--classifier_scaler', required=True)
    ap.add_argument('--overlap_checkpoint', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    pos_password = _load_positive_password_examples(args.password_dirs)
    pos_mixed = _load_positive_mixed_password_examples(args.train_mixed_dirs)
    neg_mixed = _load_mixed_negative_examples(args.train_mixed_dirs, max_per_session=10)
    neg_hard = _load_mixed_hard_negative_examples(args.train_mixed_dirs, per_episode=3)
    target_neg = max(len(pos_password) + len(pos_mixed), len(neg_mixed) + len(neg_hard))
    neg_onset = _load_onset_negative_examples(args.onset_negative_root, target_count=max(40, target_neg // 2))

    train_items = pos_password + pos_mixed + neg_mixed + neg_hard + neg_onset
    rng.shuffle(train_items)
    train_split, val_split = _group_split_items(train_items, train_ratio=0.85)

    model, best_val = train_model(train_split, val_split, device)

    classifier = load_external_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    classifier.eval()
    overlap = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap.eval(); overlap.freeze_classifier(True)

    peak_train_eps = []
    for d in args.password_dirs:
        peak_train_eps.extend(_load_password_attempt_episodes(d))
    for d in args.train_mixed_dirs:
        peak_train_eps.extend(build_password_episodes(d))
    peak_model = _train_peak_model(peak_train_eps)

    eval_eps = []
    for d in args.eval_dirs:
        eval_eps.extend(build_password_episodes(d))
    by_eval = {}
    for ep in eval_eps:
        by_eval.setdefault(ep.session_id, []).append(ep)

    baseline_rows=[]; overlap_rows=[]; debug=[]
    for session_id, session_eps in sorted(by_eval.items()):
        loader = SessionLoader(session_eps[0].session_path)
        ts, imu = loader.get_imu(); sr = estimate_sample_rate_hz(ts)
        cands = _propose_candidates_fullstream(imu, sr)
        seg_scores = _score_candidates_segment_cnn(model, cands, imu, ts, sr, device)
        scored = sorted(zip(seg_scores, cands), key=lambda x: x[0], reverse=True)
        top_candidates = [{'segment_score': float(s), 'crop_start': int(c['crop_start']), 'crop_end': int(c['crop_end']), 'cluster_num_peaks': int(c['num_peaks'])} for s,c in scored[:5]]
        for ep in session_eps:
            if not scored:
                debug.append({'session_id': session_id, 'episode_id': ep.episode_id, 'error': 'no_candidates'})
                continue
            best_score, best_cand = scored[0]
            base, ov, rec_dbg = _recover_with_peak_model(best_cand, imu, ts, sr, peak_model, classifier, overlap, device, ep)
            if base is None or ov is None:
                debug.append({'session_id': session_id, 'episode_id': ep.episode_id, 'error': 'recover_failed', 'top_candidates': top_candidates})
                continue
            base['session_id']=session_id; base['episode_id']=ep.episode_id
            ov['session_id']=session_id; ov['episode_id']=ep.episode_id
            baseline_rows.append(base); overlap_rows.append(ov)
            debug.append({'session_id': session_id, 'episode_id': ep.episode_id, 'segment_score': float(best_score), 'selected_candidate': {'crop_start': int(best_cand['crop_start']), 'crop_end': int(best_cand['crop_end']), 'cluster_num_peaks': int(best_cand['num_peaks'])}, 'top_candidates': top_candidates, **rec_dbg})

    report = {
        'mode': 'segment_slice_cnn',
        'train_summary': {
            'num_pos_password': len(pos_password),
            'num_pos_mixed': len(pos_mixed),
            'num_neg_mixed': len(neg_mixed),
            'num_neg_hard': len(neg_hard),
            'num_neg_onset': len(neg_onset),
            'num_train_items': len(train_split),
            'num_val_items': len(val_split),
            'best_val_acc': float(best_val),
        },
        'baseline_fixed_window': aggregate_episode_results(baseline_rows),
        'overlap_refine': aggregate_episode_results(overlap_rows),
    }
    (out_dir/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir/'debug_rows.json').write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
