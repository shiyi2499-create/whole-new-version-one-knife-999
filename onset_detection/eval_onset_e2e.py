"""
End-to-End Evaluation
=====================

Path A:
  password session -> onset detector -> password classifier

Path B:
  mixed2 continuous stream
    -> password_boundary detector
    -> predicted password episode(s)
    -> onset detector inside episode(s)
    -> gap-based grouping
    -> existing password classifier

This file does not modify the existing password classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import torch

from onset_model import build_onset_model
from onset_preprocessor import (
    DEFAULT_LABEL_RADIUS_MS,
    DEFAULT_STRIDE_MS,
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_WINDOW_MS,
    PASSWORD_BOUNDARY_STRIDE_MS,
    PASSWORD_BOUNDARY_WINDOW_MS,
    extract_sliding_windows,
    get_password_segments_from_activity_log,
    refine_password_segments_with_events,
    load_activity_log,
    load_events_csv,
    load_sensor_csv,
    resample_window,
    window_samples,
)
from onset_utils import (
    Episode,
    decode_password_boundary_predictions,
    group_onsets_by_gap,
    match_episodes,
)
from eval_onset import run_password_boundary_inference


# ── Password-classifier imports ──────────────────────────────

_PROJECT_ROOT = None


def _setup_project_imports(project_root: str = ""):
    global _PROJECT_ROOT
    if project_root:
        root = os.path.abspath(project_root)
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _PROJECT_ROOT = root
    phase3 = os.path.join(root, "phase3_password_inception")
    for p in (root, phase3):
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_project_imports()

try:
    from run_password_closure_inception import (
        load_final_inception,
        normalize_sequence,
        supported_key,
        topk_strings_from_prob_vectors,
    )
except ImportError:
    print("⚠ Could not import password classifier. Run from the project root or pass --project-root.")
    sys.exit(1)


# ── Device / model loaders ───────────────────────────────────

def resolve_device(device: str = "auto") -> torch.device:
    req = (device or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"
    return torch.device(req)



def load_detector(checkpoint_path: str, scaler_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_classes = int(ckpt.get("n_classes", 1))
    model = build_onset_model(
        ckpt.get("model_name", "cnn"),
        n_channels=ckpt.get("n_channels", 6),
        n_classes=n_classes,
        task=ckpt.get("task", "onset"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = np.load(scaler_path)
    return model, scaler["means"], scaler["stds"], ckpt


# ── Onset detection on a stream ──────────────────────────────

def detect_onsets_in_stream(
    sensor: np.ndarray,
    onset_model: torch.nn.Module,
    onset_means: np.ndarray,
    onset_stds: np.ndarray,
    device: torch.device,
    onset_window_ms: int = DEFAULT_WINDOW_MS,
    onset_stride_ms: int = DEFAULT_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    threshold: float = 0.5,
    nms_radius_ms: float = 100.0,
) -> list[int]:
    from onset_utils import detect_peaks, nms_1d

    result = extract_sliding_windows(
        sensor,
        np.array([], dtype=np.int64),
        window_ms=onset_window_ms,
        stride_ms=onset_stride_ms,
        label_radius_ms=DEFAULT_LABEL_RADIUS_MS,
        target_rate_hz=target_rate_hz,
    )
    if len(result["windows"]) == 0:
        return []

    X = result["windows"].astype(np.float32)
    for ch in range(X.shape[-1]):
        X[:, :, ch] = (X[:, :, ch] - onset_means[ch]) / max(onset_stds[ch], 1e-10)

    all_probs = []
    batch_size = 256
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i + batch_size]).to(device)
        with torch.no_grad():
            logits = onset_model(batch)
            probs = torch.sigmoid(logits.squeeze(-1))
        all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs) if all_probs else np.array([])
    times_s = result["times_s"]
    peaks = detect_peaks(probs, times_s, threshold=threshold, smooth_n=3)
    peaks = nms_1d(peaks, radius_s=nms_radius_ms / 1000.0)
    return [int(p["time_s"] * 1e9) for p in peaks]


# ── Classifier windows ───────────────────────────────────────

def cut_classifier_windows(
    sensor: np.ndarray,
    onset_times_ns: list[int],
    pre_ms: int = 100,
    post_ms: int = 200,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> list[Optional[np.ndarray]]:
    ts = sensor[:, 0]
    vals = sensor[:, 1:]
    target_len = window_samples(pre_ms + post_ms, target_rate_hz)
    windows = []
    for onset_ns in onset_times_ns:
        w_start = onset_ns - pre_ms * 1_000_000
        w_end = onset_ns + post_ms * 1_000_000
        idx_start = np.searchsorted(ts, w_start, side="left")
        idx_end = np.searchsorted(ts, w_end, side="right")
        if idx_end - idx_start < 4:
            windows.append(None)
            continue
        chunk = vals[idx_start:idx_end]
        windows.append(resample_window(chunk, target_len))
    return windows



def classify_windows(
    windows: list[Optional[np.ndarray]],
    classifier: torch.nn.Module,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
) -> list[Optional[np.ndarray]]:
    valid_indices = [i for i, w in enumerate(windows) if w is not None]
    if not valid_indices:
        return [None] * len(windows)

    X = np.stack([windows[i] for i in valid_indices]).astype(np.float32)
    for ch in range(X.shape[2]):
        X[:, :, ch] = (X[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)

    classifier.eval()
    with torch.no_grad():
        logits = classifier(torch.from_numpy(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    out: list[Optional[np.ndarray]] = [None] * len(windows)
    for batch_idx, orig_idx in enumerate(valid_indices):
        out[orig_idx] = probs[batch_idx]
    return out



def collect_onsets_inside_episodes(onset_times_ns: list[int], episodes: list[Episode]) -> list[list[int]]:
    groups = []
    for ep in episodes:
        groups.append([t for t in onset_times_ns if ep.start_s <= (t / 1e9) <= ep.end_s])
    return groups


# ── Metrics helpers ──────────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


SEQ_HIT_CUTOFFS = (10, 50, 100)


def score_sequence_paths(
    path_onsets: dict[str, list[int]],
    ref: str,
    sensor: np.ndarray,
    classifier,
    cls_classes,
    cls_means,
    cls_stds,
    device,
    target_rate_hz: int,
    stats: dict,
):
    tags = ["e2e_full", "e2e_gt_seg", "e2e_gt_aligned", "gt_baseline"]
    for tag in tags:
        windows = cut_classifier_windows(sensor, path_onsets[tag], target_rate_hz=target_rate_hz)
        prob_vecs = classify_windows(windows, classifier, cls_means, cls_stds, device)
        valid_probs = [p for p in prob_vecs if p is not None]
        if not valid_probs:
            hyp = ""
            prob_matrix = None
        else:
            prob_matrix = np.stack(valid_probs)
            hyp = "".join(cls_classes[int(np.argmax(p))] for p in valid_probs)

        for i, ref_ch in enumerate(ref):
            if i >= len(valid_probs):
                break
            ranked = np.argsort(-valid_probs[i])
            ranked_classes = [cls_classes[r] for r in ranked]
            for k in (1, 3, 5):
                if ref_ch in ranked_classes[:k]:
                    stats["topk_correct"][tag][k] += 1

        stats["total_edits"][tag] += levenshtein(ref, hyp)

        if prob_matrix is not None:
            try:
                candidates = topk_strings_from_prob_vectors(
                    prob_matrix,
                    cls_classes,
                    branch_topk=5,
                    beam_width=max(SEQ_HIT_CUTOFFS),
                )
                candidate_strings = [c["candidate"] for c in candidates]
                for cutoff in SEQ_HIT_CUTOFFS:
                    if ref in candidate_strings[:cutoff]:
                        stats["seq_hits"][tag][cutoff] += 1
            except Exception:
                pass


# ── Path A helpers ───────────────────────────────────────────

PART_RE = re.compile(r"_part(\d+)_")


def discover_password_sessions(dirs: list[str]) -> list[str]:
    sessions = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _subdirs, files in os.walk(d):
            for f in sorted(files):
                if f.startswith(".") or f.startswith("._"):
                    continue
                if f.endswith("_sensor.csv") and "_free_type_" in f:
                    prefix = os.path.join(root, f.replace("_sensor.csv", ""))
                    if os.path.exists(prefix + "_events.csv"):
                        sessions.append(prefix)
    return sorted(sessions)



def parse_part(sess: str) -> int:
    m = PART_RE.search(os.path.basename(sess))
    return int(m.group(1)) if m else -1



def load_ground_truth_sequences(session_prefix: str) -> list[dict]:
    events_path = session_prefix + "_events.csv"
    attempts_path = session_prefix + "_attempts.csv"

    attempts = []
    if os.path.exists(attempts_path):
        with open(attempts_path) as f:
            attempts = list(csv.DictReader(f))

    sequences = []
    cur_events = []
    with open(events_path) as f:
        for row in csv.DictReader(f):
            if row["event_type"] != "press":
                continue
            key = row["key"].lower()
            ts = int(row["timestamp_ns"])
            if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc", "left", "right", "up", "down", "delete"}:
                continue
            if key in {"enter", "return"}:
                sequences.append(cur_events.copy())
                cur_events = []
                continue
            if key not in {"space", "backspace"} and supported_key(key):
                cur_events.append({"key": key, "timestamp_ns": ts})

    out = []
    for idx, seq_events in enumerate(sequences):
        att = attempts[idx] if idx < len(attempts) else {}
        match = (att.get("match") or "").upper()
        if match and match != "YES":
            continue
        ref = normalize_sequence(att.get("typed_text", ""))
        if not ref or not seq_events:
            continue
        out.append({
            "reference": ref,
            "gt_onset_times_ns": [e["timestamp_ns"] for e in seq_events],
        })
    return out


# ── Path A eval ──────────────────────────────────────────────

def eval_e2e_on_sessions(
    onset_model,
    onset_means,
    onset_stds,
    onset_ckpt,
    classifier,
    cls_classes,
    cls_means,
    cls_stds,
    device,
    sessions,
    threshold=0.5,
    nms_radius_ms=100.0,
) -> dict:
    onset_window_ms = onset_ckpt.get("window_ms", DEFAULT_WINDOW_MS)
    onset_stride_ms = onset_ckpt.get("stride_ms", DEFAULT_STRIDE_MS)
    target_rate_hz = onset_ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)

    total_chars = 0
    total_seqs = 0
    missed_chars = 0
    extra_chars = 0
    topk_correct = {1: 0, 3: 0, 5: 0}
    seq_hits = {c: 0 for c in SEQ_HIT_CUTOFFS}
    total_edits = 0

    for sess in sessions:
        sensor = load_sensor_csv(sess + "_sensor.csv")
        gt_sequences = load_ground_truth_sequences(sess)
        if not gt_sequences:
            continue
        pred_onsets_ns = detect_onsets_in_stream(
            sensor,
            onset_model,
            onset_means,
            onset_stds,
            device,
            onset_window_ms=onset_window_ms,
            onset_stride_ms=onset_stride_ms,
            target_rate_hz=target_rate_hz,
            threshold=threshold,
            nms_radius_ms=nms_radius_ms,
        )

        for seq in gt_sequences:
            ref = seq["reference"]
            gt_onsets = seq["gt_onset_times_ns"]
            if not ref:
                continue
            total_chars += len(ref)
            total_seqs += 1

            pw_start = min(gt_onsets) - 200_000_000
            pw_end = max(gt_onsets) + 500_000_000
            seq_pred_onsets = [t for t in pred_onsets_ns if pw_start <= t <= pw_end]
            windows = cut_classifier_windows(sensor, seq_pred_onsets, target_rate_hz=target_rate_hz)
            prob_vecs = classify_windows(windows, classifier, cls_means, cls_stds, device)
            valid_probs = [p for p in prob_vecs if p is not None]
            hyp = "".join(cls_classes[int(np.argmax(p))] for p in valid_probs)

            missed_chars += max(0, len(gt_onsets) - len(seq_pred_onsets))
            extra_chars += max(0, len(seq_pred_onsets) - len(gt_onsets))

            for i, ref_ch in enumerate(ref):
                if i >= len(valid_probs):
                    break
                ranked = np.argsort(-valid_probs[i])
                ranked_classes = [cls_classes[r] for r in ranked]
                for k in (1, 3, 5):
                    if ref_ch in ranked_classes[:k]:
                        topk_correct[k] += 1

            total_edits += levenshtein(ref, hyp)
            if valid_probs:
                try:
                    candidate_strings = [c["candidate"] for c in topk_strings_from_prob_vectors(np.stack(valid_probs), cls_classes, branch_topk=5, beam_width=max(SEQ_HIT_CUTOFFS))]
                    for cutoff in SEQ_HIT_CUTOFFS:
                        if ref in candidate_strings[:cutoff]:
                            seq_hits[cutoff] += 1
                except Exception:
                    pass

    def safe_div(a, b):
        return a / max(b, 1)

    results = {
        "total_sequences": total_seqs,
        "total_chars": total_chars,
        "char_top1": safe_div(topk_correct[1], total_chars),
        "char_top3": safe_div(topk_correct[3], total_chars),
        "char_top5": safe_div(topk_correct[5], total_chars),
        "cer": safe_div(total_edits, total_chars),
        "missed_characters": missed_chars,
        "extra_characters": extra_chars,
    }
    for cutoff in SEQ_HIT_CUTOFFS:
        results[f"sequence_top{cutoff}"] = safe_div(seq_hits[cutoff], total_seqs)

    print(f"\n{'='*60}")
    print("  PATH A: PASSWORD SESSIONS")
    print(f"{'='*60}")
    print(f"  Sequences: {total_seqs}  |  Characters: {total_chars}")
    print(f"  char_top1: {results['char_top1']:.1%}")
    print(f"  char_top3: {results['char_top3']:.1%}")
    print(f"  char_top5: {results['char_top5']:.1%}")
    for cutoff in SEQ_HIT_CUTOFFS:
        print(f"  seq_top{cutoff}: {results[f'sequence_top{cutoff}']:.1%}")
    print(f"  CER:       {results['cer']:.1%}")
    print(f"  missed / extra chars: {missed_chars} / {extra_chars}")
    print(f"{'='*60}")
    return results


# ── Path B eval ──────────────────────────────────────────────

def eval_e2e_on_mixed2(
    onset_model,
    onset_means,
    onset_stds,
    onset_ckpt,
    boundary_model,
    boundary_means,
    boundary_stds,
    boundary_ckpt,
    classifier,
    cls_classes,
    cls_means,
    cls_stds,
    device,
    mixed2_dirs,
    onset_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    boundary_gap_s: float = 0.60,
    nms_radius_ms: float = 100.0,
) -> dict:
    sessions = []
    for d in mixed2_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith("_sensor.csv") and "mixed2" in f:
                sessions.append(os.path.join(d, f.replace("_sensor.csv", "")))
    sessions = sorted(set(sessions))
    if not sessions:
        print("  ⚠ No mixed2 sessions found")
        return {}

    onset_window_ms = int(onset_ckpt.get("window_ms", DEFAULT_WINDOW_MS))
    onset_stride_ms = int(onset_ckpt.get("stride_ms", DEFAULT_STRIDE_MS))
    target_rate_hz = int(onset_ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ))
    boundary_window_ms = int(boundary_ckpt.get("window_ms", PASSWORD_BOUNDARY_WINDOW_MS))
    boundary_stride_ms = int(boundary_ckpt.get("stride_ms", PASSWORD_BOUNDARY_STRIDE_MS))

    tags = ["e2e_full", "e2e_gt_seg", "e2e_gt_aligned", "gt_baseline"]
    stats = {
        "topk_correct": {tag: {1: 0, 3: 0, 5: 0} for tag in tags},
        "seq_hits": {tag: {cutoff: 0 for cutoff in SEQ_HIT_CUTOFFS} for tag in tags},
        "total_edits": {tag: 0 for tag in tags},
    }
    total_seqs = 0
    total_chars = 0
    episode_results = []

    for sess in sessions:
        activity_log_path = sess + "_activity_log.csv"
        events_path = sess + "_events.csv"
        if not os.path.exists(activity_log_path):
            continue

        sensor = load_sensor_csv(sess + "_sensor.csv")
        activity_segments = load_activity_log(activity_log_path)
        events = load_events_csv(events_path, press_only=True) if os.path.exists(events_path) else np.array([], dtype=np.int64)
        gt_password_segs = refine_password_segments_with_events(activity_segments, events)
        if not gt_password_segs:
            continue

        gt_passwords = []
        for seg in gt_password_segs:
            gt_passwords.extend([p for p in seg.get("prompts", []) if p])
        if not gt_passwords:
            continue

        gt_episode_objs = [Episode(start_s=seg["start_time_ns"] / 1e9, end_s=seg["end_time_ns"] / 1e9, label="password") for seg in gt_password_segs]

        # Step 1: predicted password boundary -> episode(s)
        boundary_probs, boundary_times = run_password_boundary_inference(
            boundary_model,
            sensor,
            boundary_means,
            boundary_stds,
            device,
            window_ms=boundary_window_ms,
            stride_ms=boundary_stride_ms,
            target_rate_hz=target_rate_hz,
        )
        pred_password_eps = decode_password_boundary_predictions(
            boundary_probs,
            boundary_times,
            password_threshold=boundary_threshold,
            start_end_threshold=0.30,
            min_duration_s=0.40,
            merge_gap_s=boundary_gap_s,
        )
        ep_match = match_episodes(pred_password_eps, gt_episode_objs, min_iou=0.3)
        episode_results.append(ep_match)

        # Step 2: onset detector across full stream
        all_onsets_ns = detect_onsets_in_stream(
            sensor,
            onset_model,
            onset_means,
            onset_stds,
            device,
            onset_window_ms=onset_window_ms,
            onset_stride_ms=onset_stride_ms,
            target_rate_hz=target_rate_hz,
            threshold=onset_threshold,
            nms_radius_ms=nms_radius_ms,
        )

        # Collect onsets for each path using refined password episodes.
        #
        # e2e_full and e2e_gt_seg are intentionally kept GT-free at the grouping
        # stage: we keep the auto-grouped onset lists in temporal order and score
        # them against references by index only. No GT group count / matching is
        # used there anymore.
        gt_groups = [events[(events >= int(seg["start_time_ns"])) & (events <= int(seg["end_time_ns"]))].tolist() for seg in gt_password_segs]
        full_groups = collect_onsets_inside_episodes(all_onsets_ns, pred_password_eps)
        gtseg_groups = collect_onsets_inside_episodes(all_onsets_ns, gt_episode_objs)

        # Explicit oracle baseline: refined GT boundary + predicted onset, then
        # per-password groups are recovered from the GT event ranges. This is the
        # only path allowed to use GT-assisted grouping/alignment semantics.
        gt_aligned_groups = []
        for seg in gt_password_segs:
            start_ns = int(seg["start_time_ns"])
            end_ns = int(seg["end_time_ns"])
            gt_aligned_groups.append([t for t in all_onsets_ns if start_ns <= t <= end_ns])

        for pw_idx, ref in enumerate(gt_passwords):
            total_seqs += 1
            total_chars += len(ref)
            path_onsets = {
                "e2e_full": full_groups[pw_idx] if pw_idx < len(full_groups) else [],
                "e2e_gt_seg": gtseg_groups[pw_idx] if pw_idx < len(gtseg_groups) else [],
                "e2e_gt_aligned": gt_aligned_groups[pw_idx] if pw_idx < len(gt_aligned_groups) else [],
                "gt_baseline": gt_groups[pw_idx] if pw_idx < len(gt_groups) else [],
            }
            score_sequence_paths(
                path_onsets,
                ref,
                sensor,
                classifier,
                cls_classes,
                cls_means,
                cls_stds,
                device,
                target_rate_hz,
                stats,
            )

    def safe_div(a, b):
        return a / max(b, 1)

    results = {"total_sequences": total_seqs, "total_chars": total_chars}
    for tag in tags:
        m = {
            "char_top1": safe_div(stats["topk_correct"][tag][1], total_chars),
            "char_top3": safe_div(stats["topk_correct"][tag][3], total_chars),
            "char_top5": safe_div(stats["topk_correct"][tag][5], total_chars),
            "cer": safe_div(stats["total_edits"][tag], total_chars),
        }
        for cutoff in SEQ_HIT_CUTOFFS:
            m[f"sequence_top{cutoff}"] = safe_div(stats["seq_hits"][tag][cutoff], total_seqs)
        results[tag] = m

    results["delta_full_vs_gt"] = {k: results["e2e_full"][k] - results["gt_baseline"][k] for k in results["gt_baseline"]}
    results["delta_gtseg_vs_gt"] = {k: results["e2e_gt_seg"][k] - results["gt_baseline"][k] for k in results["gt_baseline"]}
    results["delta_gt_aligned_vs_gt"] = {k: results["e2e_gt_aligned"][k] - results["gt_baseline"][k] for k in results["gt_baseline"]}

    if episode_results:
        all_ious = [iou for r in episode_results for iou in r.ious]
        all_start = [e for r in episode_results for e in r.start_errors_s]
        all_end = [e for r in episode_results for e in r.end_errors_s]
        results["episode_metrics"] = {
            "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "mean_start_error_ms": float(np.mean(all_start)) * 1000.0 if all_start else float("inf"),
            "mean_end_error_ms": float(np.mean(all_end)) * 1000.0 if all_end else float("inf"),
            "episode_precision": sum(r.n_matched for r in episode_results) / max(sum(r.n_pred for r in episode_results), 1),
            "episode_recall": sum(r.n_matched for r in episode_results) / max(sum(r.n_gt for r in episode_results), 1),
        }

    print(f"\n{'='*60}")
    print("  PATH B: MIXED2 PASSWORD BOUNDARY -> ONSET -> CLASSIFIER")
    print(f"{'='*60}")
    print(f"  Sequences: {total_seqs}  |  Chars: {total_chars}")
    for label, tag in [
        ("Full E2E (pred boundary -> onset -> per-episode grouping, no GT alignment -> classify)", "e2e_full"),
        ("GT segment (refined GT boundary -> onset -> per-episode grouping, no GT alignment -> classify)", "e2e_gt_seg"),
        ("GT aligned oracle (GT boundary -> onset -> GT-assisted grouping -> classify)", "e2e_gt_aligned"),
        ("GT baseline (GT onsets -> classify)", "gt_baseline"),
    ]:
        m = results[tag]
        print(f"\n  {label}:")
        print(f"    char_top1: {m['char_top1']:.1%}")
        print(f"    char_top3: {m['char_top3']:.1%}")
        print(f"    char_top5: {m['char_top5']:.1%}")
        for cutoff in SEQ_HIT_CUTOFFS:
            print(f"    seq_top{cutoff}: {m[f'sequence_top{cutoff}']:.1%}")
        print(f"    CER:       {m['cer']:.1%}")

    if "episode_metrics" in results:
        em = results["episode_metrics"]
        print(f"\n  Boundary metrics:")
        print(f"    episode_precision: {em['episode_precision']:.3f}")
        print(f"    episode_recall:    {em['episode_recall']:.3f}")
        print(f"    mean_iou:          {em['mean_iou']:.3f}")
        print(f"    start_error:       {em['mean_start_error_ms']:.1f}ms")
        print(f"    end_error:         {em['mean_end_error_ms']:.1f}ms")

    return results


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end onset / password-boundary evaluation")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--onset-checkpoint", default="results/onset_detector.pt")
    parser.add_argument("--onset-scaler", default="results/onset_scaler.npz")
    parser.add_argument("--boundary-checkpoint", default="results/password_boundary_detector.pt")
    parser.add_argument("--boundary-scaler", default="results/password_boundary_scaler.npz")
    parser.add_argument("--classifier-checkpoint", default="results/inception_password_final.pt")
    parser.add_argument("--classifier-scaler", default="results/inception_password_scaler.npz")
    parser.add_argument("--password-dirs", nargs="+", default=["data/raw/password/len_8"])
    parser.add_argument("--test-parts", nargs="+", type=int, default=[17, 18, 19, 20])
    parser.add_argument("--mixed2-dirs", nargs="*", default=[])
    parser.add_argument("--report", default="results/onset_e2e_report.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5, help="Onset threshold")
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--boundary-gap-ms", type=float, default=600.0, help="Bridge brief internal pauses inside one password episode.")
    parser.add_argument("--nms-radius-ms", type=float, default=100.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.project_root:
        _setup_project_imports(args.project_root)
        root = os.path.abspath(args.project_root)
        for attr in ("onset_checkpoint", "onset_scaler", "boundary_checkpoint", "boundary_scaler", "classifier_checkpoint", "classifier_scaler", "report"):
            val = getattr(args, attr)
            if not os.path.isabs(val):
                setattr(args, attr, os.path.join(root, val))
        args.password_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.password_dirs]
        args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d for d in args.mixed2_dirs]

    onset_model, onset_means, onset_stds, onset_ckpt = load_detector(args.onset_checkpoint, args.onset_scaler, device)
    print(f"Onset detector: {onset_ckpt.get('model_name', 'cnn')}")

    boundary_model, boundary_means, boundary_stds, boundary_ckpt = load_detector(args.boundary_checkpoint, args.boundary_scaler, device)
    print(f"Password-boundary detector: {boundary_ckpt.get('model_name', 'password_boundary_cnn')}")

    classifier, cls_classes, cls_means, cls_stds = load_final_inception(args.classifier_checkpoint, args.classifier_scaler, device)
    print(f"Classifier: InceptionTime ({len(cls_classes)} classes)")

    results = {}

    all_sessions = discover_password_sessions(args.password_dirs)
    sessions = [s for s in all_sessions if parse_part(s) in set(args.test_parts)]
    if sessions:
        results["path_a_password"] = eval_e2e_on_sessions(
            onset_model,
            onset_means,
            onset_stds,
            onset_ckpt,
            classifier,
            cls_classes,
            cls_means,
            cls_stds,
            device,
            sessions,
            threshold=args.threshold,
            nms_radius_ms=args.nms_radius_ms,
        )

    if args.mixed2_dirs:
        results["path_b_mixed2"] = eval_e2e_on_mixed2(
            onset_model,
            onset_means,
            onset_stds,
            onset_ckpt,
            boundary_model,
            boundary_means,
            boundary_stds,
            boundary_ckpt,
            classifier,
            cls_classes,
            cls_means,
            cls_stds,
            device,
            args.mixed2_dirs,
            onset_threshold=args.threshold,
            boundary_threshold=args.boundary_threshold,
            boundary_gap_s=args.boundary_gap_ms / 1000.0,
            nms_radius_ms=args.nms_radius_ms,
        )

    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.report}")


if __name__ == "__main__":
    main()
