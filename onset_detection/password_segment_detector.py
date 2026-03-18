"""
Password Segment Detector — Two-Stage + Classifier
====================================================

Full Path B pipeline:
  mixed2 stream
    → Stage 1: binary segment classifier → coarse password region
    → Stage 2: onset detector + IKI rhythm → refined boundary + per-password onset groups
    → Stage 3: password classifier → char top-k / sequence_topN / CER
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from onset_model import build_onset_model
from onset_preprocessor import (
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_WINDOW_MS,
    DEFAULT_STRIDE_MS,
    DEFAULT_LABEL_RADIUS_MS,
    load_sensor_csv,
    load_events_csv,
    load_activity_log,
    resample_window,
    window_samples,
    get_password_segments_from_activity_log,
    refine_password_segments_with_events,
)
from onset_utils import (
    Episode,
    detect_peaks,
    nms_1d,
    match_episodes,
)
from password_segment_preprocessor import (
    SEGMENT_WINDOW_MS,
    SEGMENT_STRIDE_MS,
    N_CHANNELS,
    _iterate_window_chunks,
    discover_sessions,
)


# ── Lazy imports for password classifier (set up via _setup_imports) ──

_classifier_imported = False


def _setup_imports(project_root: str = ""):
    global _classifier_imported
    if _classifier_imported:
        return
    root = os.path.abspath(project_root) if project_root else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (root, os.path.join(root, "phase3_password_inception")):
        if p not in sys.path:
            sys.path.insert(0, p)
    _classifier_imported = True


def _load_press_rows(events_path: str) -> list[dict]:
    rows = []
    if not os.path.exists(events_path):
        return rows
    with open(events_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("event_type") != "press":
                continue
            try:
                ts = int(row["timestamp_ns"])
            except Exception:
                continue
            rows.append({"timestamp_ns": ts, "key": (row.get("key") or "").lower()})
    return rows


def _load_supported_press_timestamps(events_path: str) -> np.ndarray:
    from run_password_closure_inception import supported_key

    keep = []
    for row in _load_press_rows(events_path):
        key = row["key"]
        if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
                   "left", "right", "up", "down", "delete", "enter", "return",
                   "space", "backspace"}:
            continue
        if not supported_key(key):
            continue
        keep.append(row["timestamp_ns"])
    return np.asarray(keep, dtype=np.int64)


def _extract_gt_password_groups(events_path: str, gt_refined_segs: list[dict]) -> list[list[int]]:
    """
    Split GT password typing into per-password groups using Enter as delimiter.

    We keep only classifier-supported character keys inside each group, and we
    drop spaces/backspaces/modifiers so the GT baseline matches the main
    password-classifier evaluation path more closely.
    """
    from run_password_closure_inception import supported_key

    rows = _load_press_rows(events_path)
    if not rows:
        return []

    all_groups: list[list[int]] = []
    for seg in gt_refined_segs:
        seg_rows = [
            r for r in rows
            if int(seg["start_time_ns"]) <= r["timestamp_ns"] <= int(seg["end_time_ns"])
        ]
        cur: list[int] = []
        for row in seg_rows:
            key = row["key"]
            ts = row["timestamp_ns"]
            if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
                       "left", "right", "up", "down", "delete"}:
                continue
            if key in {"enter", "return"}:
                if cur:
                    all_groups.append(cur.copy())
                    cur = []
                continue
            if key in {"space", "backspace"}:
                continue
            if not supported_key(key):
                continue
            cur.append(ts)
        if cur:
            all_groups.append(cur.copy())
    return all_groups


# ── Stage 1: Binary Segment Classifier ──────────────────────

@dataclass
class CoarseRegion:
    start_s: float
    end_s: float
    mean_prob: float = 0.0
    max_prob: float = 0.0

    @property
    def duration_s(self):
        return max(0.0, self.end_s - self.start_s)


def load_segment_detector(checkpoint_path, scaler_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_onset_model(
        ckpt.get("model_name", "cnn"),
        n_channels=ckpt.get("n_channels", 6),
        n_classes=int(ckpt.get("n_classes", 1)),
        task=ckpt.get("task", "password_segment"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    scaler = np.load(scaler_path)
    meta = {
        "window_ms": int(ckpt.get("window_ms", SEGMENT_WINDOW_MS)),
        "stride_ms": int(ckpt.get("stride_ms", SEGMENT_STRIDE_MS)),
        "target_rate_hz": int(ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)),
        "n_channels": int(ckpt.get("n_channels", 6)),
    }
    return model, scaler["means"].astype(np.float32), np.maximum(scaler["stds"].astype(np.float32), 1e-10), meta


def run_binary_inference(model, sensor, means, stds, window_ms, stride_ms,
                         target_rate_hz, device, batch_size=256):
    windows, times_s = [], []
    for centre, win in _iterate_window_chunks(sensor, window_ms, stride_ms, target_rate_hz):
        windows.append((win - means) / stds)
        times_s.append(centre / 1e9)
    if not windows:
        return np.array([]), np.array([])
    X = np.stack(windows).astype(np.float32)
    ts = np.asarray(times_s)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.from_numpy(X[i:i+batch_size]).to(device)
            logits = model(b)
            p = torch.sigmoid(logits.squeeze(-1)) if logits.shape[-1] == 1 else torch.softmax(logits, -1)[:, 1]
            probs.append(p.cpu().numpy())
    return np.concatenate(probs), ts


def extract_coarse_regions(probs, times_s, threshold=0.5, merge_gap_s=1.5,
                           min_duration_s=2.0, margin_s=1.0):
    if len(probs) == 0:
        return []
    active = probs >= threshold
    # Bridge short gaps
    in_act, last_end = False, -1
    for i in range(len(active)):
        if active[i]:
            if not in_act and last_end >= 0 and (times_s[i] - times_s[last_end]) <= merge_gap_s:
                active[last_end:i+1] = True
            in_act = True
        else:
            if in_act: last_end = i - 1
            in_act = False
    # Contiguous regions
    regions = []
    start = None
    for i in range(len(active)):
        if active[i] and start is None:
            start = i
        elif not active[i] and start is not None:
            rp = probs[start:i]
            regions.append(CoarseRegion(times_s[start] - margin_s, times_s[i-1] + margin_s,
                                        float(np.mean(rp)), float(np.max(rp))))
            start = None
    if start is not None:
        rp = probs[start:]
        regions.append(CoarseRegion(times_s[start] - margin_s, times_s[-1] + margin_s,
                                    float(np.mean(rp)), float(np.max(rp))))
    return [r for r in regions if r.duration_s >= min_duration_s]


# ── Stage 2: Onset detection + IKI rhythm ────────────────────

def load_onset_detector(checkpoint_path, scaler_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_onset_model(
        ckpt.get("model_name", "cnn"),
        n_channels=ckpt.get("n_channels", 6),
        n_classes=int(ckpt.get("n_classes", 1)),
        task=ckpt.get("task", "onset"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    scaler = np.load(scaler_path)
    meta = {"window_ms": int(ckpt.get("window_ms", DEFAULT_WINDOW_MS)),
            "stride_ms": int(ckpt.get("stride_ms", DEFAULT_STRIDE_MS)),
            "target_rate_hz": int(ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ))}
    return model, scaler["means"].astype(np.float32), np.maximum(scaler["stds"].astype(np.float32), 1e-10), meta


def detect_onsets_in_region(model, sensor, means, stds, region,
                            window_ms, stride_ms, target_rate_hz,
                            device, threshold=0.5, nms_radius_s=0.08, batch_size=256):
    ts_ns = sensor[:, 0]
    mask = (ts_ns >= region.start_s * 1e9) & (ts_ns <= region.end_s * 1e9)
    if mask.sum() < 10:
        return []
    rsensor = sensor[mask]
    windows, times = [], []
    for c, w in _iterate_window_chunks(rsensor, window_ms, stride_ms, target_rate_hz):
        windows.append((w - means) / stds)
        times.append(c / 1e9)
    if not windows:
        return []
    X = np.stack(windows).astype(np.float32)
    ts = np.asarray(times)
    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            b = torch.from_numpy(X[i:i+batch_size]).to(device)
            probs.append(torch.sigmoid(model(b).squeeze(-1)).cpu().numpy())
    p = np.concatenate(probs)
    peaks = nms_1d(detect_peaks(p, ts, threshold=threshold), radius_s=nms_radius_s)
    return [pk["time_s"] for pk in peaks]


def find_password_rhythm_clusters(onset_times, expected_len=8, iki_min_s=0.15,
                                  iki_max_s=2.0, iki_cv_max=0.8, group_gap_s=2.5,
                                  min_onsets_per_group=4):
    if len(onset_times) < min_onsets_per_group:
        return []
    ot = sorted(onset_times)
    groups, cur = [], [ot[0]]
    for i in range(1, len(ot)):
        if ot[i] - ot[i-1] > group_gap_s:
            groups.append(cur); cur = [ot[i]]
        else:
            cur.append(ot[i])
    groups.append(cur)

    clusters = []
    for g in groups:
        if len(g) < min_onsets_per_group:
            continue
        ikis = np.diff(g)
        valid = ikis[(ikis >= iki_min_s) & (ikis <= iki_max_s)]
        if len(valid) < max(2, min_onsets_per_group - 2):
            continue
        med = float(np.median(valid))
        cv = float(np.std(valid) / max(np.mean(valid), 1e-6))
        if cv <= iki_cv_max:
            clusters.append({"onsets": g, "start_s": g[0], "end_s": g[-1],
                             "n_onsets": len(g), "median_iki": med, "iki_cv": cv})
    return clusters


def split_cluster_into_passwords(cluster, enter_gap_min_s=0.8):
    onsets = cluster["onsets"]
    med_iki = cluster["median_iki"]
    if len(onsets) < 3:
        return [onsets]
    ikis = np.diff(onsets)
    subs, cur = [], [onsets[0]]
    for i in range(len(ikis)):
        if ikis[i] >= enter_gap_min_s and ikis[i] > med_iki * 2.5:
            if len(cur) >= 3:
                subs.append(cur)
            cur = [onsets[i+1]]
        else:
            cur.append(onsets[i+1])
    if len(cur) >= 3:
        subs.append(cur)
    return subs


# ── Stage 3: Classifier + Metrics ────────────────────────────

def cut_classifier_windows(sensor, onset_times_ns, pre_ms=100, post_ms=200,
                           target_rate_hz=DEFAULT_TARGET_RATE_HZ):
    ts = sensor[:, 0]
    vals = sensor[:, 1:]
    tgt = window_samples(pre_ms + post_ms, target_rate_hz)
    out = []
    for t_ns in onset_times_ns:
        i0 = np.searchsorted(ts, t_ns - pre_ms * 1e6, side="left")
        i1 = np.searchsorted(ts, t_ns + post_ms * 1e6, side="right")
        if i1 - i0 < 4:
            out.append(None)
        else:
            out.append(resample_window(vals[i0:i1], tgt))
    return out


def classify_windows(windows, classifier, means, stds, device):
    valid_idx = [i for i, w in enumerate(windows) if w is not None]
    if not valid_idx:
        return [None] * len(windows)
    X = np.stack([windows[i] for i in valid_idx]).astype(np.float32)
    for ch in range(X.shape[2]):
        X[:, :, ch] = (X[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)
    classifier.eval()
    with torch.no_grad():
        probs = torch.softmax(classifier(torch.from_numpy(X).to(device)), dim=1).cpu().numpy()
    out = [None] * len(windows)
    for bi, oi in enumerate(valid_idx):
        out[oi] = probs[bi]
    return out


def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1]+1, prev[j]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = cur
    return prev[-1]


SEQ_HIT_CUTOFFS = (10, 50, 100)


def score_one_password(onset_times_ns, ref, sensor, classifier, cls_classes,
                       cls_means, cls_stds, device, target_rate_hz):
    """Score a single predicted password group against its reference string."""
    from run_password_closure_inception import topk_strings_from_prob_vectors

    windows = cut_classifier_windows(sensor, onset_times_ns, target_rate_hz=target_rate_hz)
    prob_vecs = classify_windows(windows, classifier, cls_means, cls_stds, device)
    valid = [p for p in prob_vecs if p is not None]

    result = {"n_onsets": len(onset_times_ns), "n_valid_windows": len(valid),
              "reference": ref, "hypothesis": "", "cer": 1.0,
              "char_top1": 0.0, "char_top3": 0.0, "char_top5": 0.0}
    for cutoff in SEQ_HIT_CUTOFFS:
        result[f"seq_top{cutoff}"] = 0

    if not valid:
        return result

    hyp = "".join(cls_classes[int(np.argmax(p))] for p in valid)
    result["hypothesis"] = hyp
    result["cer"] = levenshtein(ref, hyp) / max(len(ref), 1)

    topk_hits = {1: 0, 3: 0, 5: 0}
    for i, ref_ch in enumerate(ref):
        if i >= len(valid): break
        ranked = [cls_classes[r] for r in np.argsort(-valid[i])]
        for k in (1, 3, 5):
            if ref_ch in ranked[:k]:
                topk_hits[k] += 1
    n = max(len(ref), 1)
    result["char_top1"] = topk_hits[1] / n
    result["char_top3"] = topk_hits[3] / n
    result["char_top5"] = topk_hits[5] / n

    try:
        cands = topk_strings_from_prob_vectors(
            np.stack(valid), cls_classes, branch_topk=5, beam_width=max(SEQ_HIT_CUTOFFS))
        cand_strs = [c["candidate"] for c in cands]
        for cutoff in SEQ_HIT_CUTOFFS:
            result[f"seq_top{cutoff}"] = 1 if ref in cand_strs[:cutoff] else 0
    except Exception:
        pass

    return result


# ── Full pipeline ────────────────────────────────────────────

def run_full_pipeline(
    sensor, sess_prefix,
    seg_model, seg_means, seg_stds, seg_meta,
    onset_model, onset_means, onset_stds, onset_meta,
    classifier, cls_classes, cls_means, cls_stds,
    device,
    gt_passwords, gt_refined_segs, gt_password_groups_ns,
    segment_threshold=0.5, onset_threshold=0.5,
    expected_password_len=8,
):
    target_rate_hz = seg_meta["target_rate_hz"]

    # ── Stage 1: coarse regions ──
    probs, times = run_binary_inference(
        seg_model, sensor, seg_means, seg_stds,
        seg_meta["window_ms"], seg_meta["stride_ms"], target_rate_hz, device)
    coarse = extract_coarse_regions(probs, times, threshold=segment_threshold)

    # ── Stage 2: onset + IKI within coarse ──
    all_onsets = []
    for region in coarse:
        onsets = detect_onsets_in_region(
            onset_model, sensor, onset_means, onset_stds, region,
            onset_meta["window_ms"], onset_meta["stride_ms"], target_rate_hz,
            device, threshold=onset_threshold)
        all_onsets.extend(onsets)
    all_onsets = sorted(all_onsets)

    # Rhythm analysis → password clusters → per-password onset groups
    clusters = find_password_rhythm_clusters(all_onsets, expected_len=expected_password_len)
    password_groups_s = []  # list of list[float] (seconds)
    for cluster in clusters:
        subs = split_cluster_into_passwords(cluster)
        password_groups_s.extend(subs)

    # Fallback: if rhythm analysis returns nothing, use all onsets as one group
    if not password_groups_s and all_onsets:
        password_groups_s = [all_onsets]

    # Convert to ns for classifier
    password_groups_ns = [[int(t * 1e9) for t in g] for g in password_groups_s]

    # Build refined episodes from groups
    refined_episodes = []
    for g in password_groups_s:
        if g:
            refined_episodes.append(Episode(start_s=g[0] - 0.15, end_s=g[-1] + 0.25, label="password"))

    # GT episodes for boundary evaluation
    gt_episodes = [Episode(start_s=s["start_time_ns"]/1e9, end_s=s["end_time_ns"]/1e9, label="password")
                   for s in gt_refined_segs]
    ep_match = match_episodes(refined_episodes, gt_episodes, min_iou=0.3)

    # ── Stage 3: classify each password group ──
    # Predicted path (e2e_full): use predicted groups in order
    # GT baseline: use GT onset times directly
    e2e_results = []
    gt_results = []

    for pw_idx, ref in enumerate(gt_passwords):
        # E2E full: predicted groups
        if pw_idx < len(password_groups_ns):
            e2e_r = score_one_password(
                password_groups_ns[pw_idx], ref, sensor,
                classifier, cls_classes, cls_means, cls_stds, device, target_rate_hz)
        else:
            e2e_r = {"reference": ref, "hypothesis": "", "cer": 1.0,
                     "char_top1": 0.0, "char_top3": 0.0, "char_top5": 0.0,
                     "n_onsets": 0, "n_valid_windows": 0}
            for c in SEQ_HIT_CUTOFFS: e2e_r[f"seq_top{c}"] = 0
        e2e_results.append(e2e_r)

        # GT baseline: GT onset times within GT segment
        if pw_idx < len(gt_password_groups_ns):
            gt_in = gt_password_groups_ns[pw_idx]
            gt_r = score_one_password(
                gt_in, ref, sensor,
                classifier, cls_classes, cls_means, cls_stds, device, target_rate_hz)
        else:
            gt_r = dict(e2e_r)
        gt_results.append(gt_r)

    # ── Aggregate metrics ──
    def aggregate(results, n_seqs, n_chars):
        out = {}
        for k in ("char_top1", "char_top3", "char_top5"):
            out[k] = sum(r[k] * len(r["reference"]) for r in results) / max(n_chars, 1)
        out["cer"] = sum(levenshtein(r["reference"], r.get("hypothesis", "")) for r in results) / max(n_chars, 1)
        for cutoff in SEQ_HIT_CUTOFFS:
            out[f"sequence_top{cutoff}"] = sum(r.get(f"seq_top{cutoff}", 0) for r in results) / max(n_seqs, 1)
        return out

    n_seqs = len(gt_passwords)
    n_chars = sum(len(pw) for pw in gt_passwords)

    return {
        "session": os.path.basename(sess_prefix),
        "n_gt_passwords": n_seqs,
        "n_gt_chars": n_chars,
        "n_coarse_regions": len(coarse),
        "n_refined_episodes": len(refined_episodes),
        "n_predicted_groups": len(password_groups_ns),
        "n_total_onsets": len(all_onsets),
        "episode_iou": ep_match.mean_iou,
        "episode_precision": ep_match.precision,
        "episode_recall": ep_match.recall,
        "start_error_ms": ep_match.mean_start_error_ms,
        "end_error_ms": ep_match.mean_end_error_ms,
        "e2e_full": aggregate(e2e_results, n_seqs, n_chars),
        "gt_baseline": aggregate(gt_results, n_seqs, n_chars),
        "e2e_examples": [{"ref": r["reference"], "hyp": r.get("hypothesis", "")} for r in e2e_results[:5]],
    }


# ── CLI entry point ──────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Two-stage password segment detection + classifier (Path B)")
    p.add_argument("--project-root", default="")
    p.add_argument("--segment-checkpoint", default="results/password_segment_detector.pt")
    p.add_argument("--segment-scaler", default="results/password_segment_scaler.npz")
    p.add_argument("--onset-checkpoint", default="results/onset_detector.pt")
    p.add_argument("--onset-scaler", default="results/onset_scaler.npz")
    p.add_argument("--classifier-checkpoint", default="results/inception_password_final.pt")
    p.add_argument("--classifier-scaler", default="results/inception_password_scaler.npz")
    p.add_argument("--mixed2-dirs", nargs="+", default=["data/raw/onset_mixed2"])
    p.add_argument("--report", default="results/password_segment_e2e_report.json")
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--segment-threshold", type=float, default=0.5)
    p.add_argument("--onset-threshold", type=float, default=0.5)
    p.add_argument("--expected-password-len", type=int, default=8)
    args = p.parse_args()

    if args.project_root:
        _setup_imports(args.project_root)
        root = os.path.abspath(args.project_root)
        for attr in ["segment_checkpoint", "segment_scaler", "onset_checkpoint",
                     "onset_scaler", "classifier_checkpoint", "classifier_scaler", "report"]:
            v = getattr(args, attr)
            if not os.path.isabs(v):
                setattr(args, attr, os.path.join(root, v))
        args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.mixed2_dirs]
    else:
        _setup_imports()

    from run_password_closure_inception import load_final_inception, normalize_sequence

    req = (args.device or "auto").lower()
    if req == "auto":
        req = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    device = torch.device(req)
    print(f"Device: {device}")

    seg_model, seg_means, seg_stds, seg_meta = load_segment_detector(args.segment_checkpoint, args.segment_scaler, device)
    print(f"Segment detector loaded")
    onset_model, onset_means, onset_stds, onset_meta = load_onset_detector(args.onset_checkpoint, args.onset_scaler, device)
    print(f"Onset detector loaded")
    classifier, cls_classes, cls_means, cls_stds = load_final_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    print(f"Classifier loaded ({len(cls_classes)} classes)")

    sessions = discover_sessions(args.mixed2_dirs, mode_filter="mixed2", dedup=False)
    if not sessions:
        sessions = discover_sessions(args.mixed2_dirs, mode_filter="", dedup=False)
    print(f"Found {len(sessions)} mixed2 sessions\n")

    all_results = []
    for sess in sessions:
        alog = sess + "_activity_log.csv"
        events_path = sess + "_events.csv"
        if not os.path.exists(alog):
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        activity_segments = load_activity_log(alog)
        events = _load_supported_press_timestamps(events_path) if os.path.exists(events_path) else np.array([], dtype=np.int64)
        gt_refined = refine_password_segments_with_events(activity_segments, events)
        if not gt_refined:
            continue

        # Extract GT password strings from prompts
        gt_passwords = []
        for seg in gt_refined:
            gt_passwords.extend([normalize_sequence(p) for p in seg.get("prompts", []) if p])
        if not gt_passwords:
            continue

        gt_password_groups = _extract_gt_password_groups(events_path, gt_refined) if os.path.exists(events_path) else []

        print(f"  Session: {os.path.basename(sess)}  ({len(gt_passwords)} passwords)")
        result = run_full_pipeline(
            sensor, sess,
            seg_model, seg_means, seg_stds, seg_meta,
            onset_model, onset_means, onset_stds, onset_meta,
            classifier, cls_classes, cls_means, cls_stds,
            device, gt_passwords, gt_refined, gt_password_groups,
            segment_threshold=args.segment_threshold,
            onset_threshold=args.onset_threshold,
            expected_password_len=args.expected_password_len,
        )
        all_results.append(result)

        e = result["e2e_full"]
        g = result["gt_baseline"]
        print(f"    Coarse regions: {result['n_coarse_regions']}  |  Pred groups: {result['n_predicted_groups']}  |  Onsets: {result['n_total_onsets']}")
        print(f"    Episode IoU: {result['episode_iou']:.3f}  P={result['episode_precision']:.3f}  R={result['episode_recall']:.3f}")
        print(f"    E2E  char_top1={e['char_top1']:.1%}  top3={e['char_top3']:.1%}  top5={e['char_top5']:.1%}  CER={e['cer']:.1%}")
        print(f"    GT   char_top1={g['char_top1']:.1%}  top3={g['char_top3']:.1%}  top5={g['char_top5']:.1%}  CER={g['cer']:.1%}")
        for ex in result.get("e2e_examples", []):
            print(f"      ref={ex['ref']}  hyp={ex['hyp']}")

    # Aggregate
    if all_results:
        n_seqs = sum(r["n_gt_passwords"] for r in all_results)
        n_chars = sum(r["n_gt_chars"] for r in all_results)
        print(f"\n{'='*60}")
        print(f"  AGGREGATE ({len(all_results)} sessions, {n_seqs} passwords, {n_chars} chars)")
        print(f"{'='*60}")
        for tag in ("e2e_full", "gt_baseline"):
            label = "E2E Full" if tag == "e2e_full" else "GT Baseline"
            # Weighted average
            metrics = {}
            for k in ("char_top1", "char_top3", "char_top5", "cer"):
                metrics[k] = sum(r[tag][k] * r["n_gt_chars"] for r in all_results) / max(n_chars, 1)
            for cutoff in SEQ_HIT_CUTOFFS:
                key = f"sequence_top{cutoff}"
                metrics[key] = sum(r[tag][key] * r["n_gt_passwords"] for r in all_results) / max(n_seqs, 1)
            print(f"\n  {label}:")
            print(f"    char_top1: {metrics['char_top1']:.1%}   top3: {metrics['char_top3']:.1%}   top5: {metrics['char_top5']:.1%}")
            for c in SEQ_HIT_CUTOFFS:
                print(f"    seq_top{c}: {metrics[f'sequence_top{c}']:.1%}")
            print(f"    CER: {metrics['cer']:.1%}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump({"sessions": all_results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {args.report}")


if __name__ == "__main__":
    main()
