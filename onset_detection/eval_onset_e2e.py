"""
End-to-End Onset → Password Classifier Pipeline
=================================================
Full attack chain demonstration with TWO paths:

  Path A (original): password session → onset detect → classifier → top-k
  Path B (new):      mixed2 stream → activity segment → find typing_2 →
                     onset detect within typing_2 → gap-group → classify

Path B compares FOUR baselines (from most realistic to most oracle):

  e2e_full:       predicted segment → predicted typing_2 → onset detect →
                  gap-based password grouping → classify
                  (zero GT information)

  e2e_gt_seg:     GT segment boundary → onset detect → gap-based grouping
                  → classify  (GT boundary only)

  e2e_gt_aligned: GT boundary → onset detect → GT-onset-assisted alignment
                  → classify  (GT boundary + GT onset timing)

  gt_baseline:    GT onset times directly → classify  (full oracle)

This decomposition isolates degradation sources:
  e2e_full − gt_baseline     = total pipeline degradation
  e2e_gt_seg − gt_baseline   = onset detection + grouping error
  e2e_gt_aligned − gt_baseline = onset detection error only

Run:
  # Path A: test on password sessions
  python3 eval_onset_e2e.py \\
    --onset-checkpoint results/onset_detector.pt \\
    --classifier-checkpoint results/inception_password_final.pt \\
    --password-dirs data/raw/password/len_8 \\
    --test-parts 17 18 19 20

  # Path B: test on mixed2 streams (full segmentation pipeline)
  python3 eval_onset_e2e.py \\
    --onset-checkpoint results/onset_detector.pt \\
    --activity-checkpoint results/activity_detector.pt \\
    --classifier-checkpoint results/inception_password_final.pt \\
    --mixed2-dirs data/raw/onset_mixed2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

# ── Imports from onset detection module ──────────────────────

from onset_model import build_onset_model
from onset_preprocessor import (
    load_sensor_csv,
    load_events_csv,
    load_activity_log,
    resample_window,
    extract_sliding_windows,
    window_samples,
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_WINDOW_MS,
    DEFAULT_STRIDE_MS,
    DEFAULT_LABEL_RADIUS_MS,
    ACTIVITY_WINDOW_MS,
    ACTIVITY_STRIDE_MS,
)
from onset_utils import (
    detect_peaks, nms_1d, match_events,
    Episode, extract_episodes_from_activity_curve, classify_episodes_by_density,
    match_episodes, format_episode_match_result, group_onsets_by_gap,
)
from eval_onset import run_activity_inference

# ── Imports from password classifier (parent project) ────────

_PROJECT_ROOT = None


def _setup_project_imports(project_root: str = ""):
    global _PROJECT_ROOT
    if project_root:
        root = os.path.abspath(project_root)
    else:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _PROJECT_ROOT = root
    phase3 = os.path.join(root, "phase3_password_inception")
    for p in [root, phase3]:
        if p not in sys.path:
            sys.path.insert(0, p)


_setup_project_imports()

try:
    from run_password_closure_inception import (
        InceptionTimeClassifier,
        load_final_inception,
        WindowConfig as ClassifierWindowConfig,
        supported_key,
        normalize_sequence,
        topk_strings_from_prob_vectors,
    )
except ImportError:
    print("⚠ Could not import password classifier. Use --project-root to point "
          "at the main project directory, or run from the project root.")
    sys.exit(1)


# ── Device ───────────────────────────────────────────────────

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


# ── Load models ──────────────────────────────────────────────

def load_onset_detector(checkpoint_path: str, scaler_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_onset_model(
        ckpt.get("model_name", "cnn"),
        n_channels=ckpt.get("n_channels", 6),
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
    onset_window_ms: int = 150,
    onset_stride_ms: int = 25,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    threshold: float = 0.5,
    nms_radius_ms: float = 100.0,
) -> list[float]:
    """
    Run onset detector on continuous sensor data.
    Returns list of predicted onset timestamps in nanoseconds.
    """
    dummy_events = np.array([], dtype=np.int64)
    result = extract_sliding_windows(
        sensor, dummy_events,
        window_ms=onset_window_ms,
        stride_ms=onset_stride_ms,
        label_radius_ms=DEFAULT_LABEL_RADIUS_MS,
        target_rate_hz=target_rate_hz,
    )

    if len(result["windows"]) == 0:
        return []

    windows = result["windows"].astype(np.float32)
    for ch in range(windows.shape[-1]):
        windows[:, :, ch] = (windows[:, :, ch] - onset_means[ch]) / max(onset_stds[ch], 1e-10)

    onset_model.eval()
    all_probs = []
    batch_size = 256
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i:i+batch_size]).to(device)
        with torch.no_grad():
            logits = onset_model(batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs)
    times_s = result["times_s"]

    peaks = detect_peaks(probs, times_s, threshold=threshold, smooth_n=3)
    peaks = nms_1d(peaks, radius_s=nms_radius_ms / 1000.0)

    onset_times_ns = [int(p["time_s"] * 1e9) for p in peaks]
    return onset_times_ns


def detect_onsets_in_time_range(
    sensor: np.ndarray,
    onset_model: torch.nn.Module,
    onset_means: np.ndarray,
    onset_stds: np.ndarray,
    device: torch.device,
    start_s: float,
    end_s: float,
    **kwargs,
) -> list[int]:
    """Detect onsets only within a specific time range of the sensor data."""
    ts_ns = sensor[:, 0]
    start_ns = int(start_s * 1e9)
    end_ns = int(end_s * 1e9)

    idx_start = np.searchsorted(ts_ns, start_ns, side="left")
    idx_end = np.searchsorted(ts_ns, end_ns, side="right")

    if idx_end - idx_start < 10:
        return []

    sub_sensor = sensor[idx_start:idx_end]
    return detect_onsets_in_stream(
        sub_sensor, onset_model, onset_means, onset_stds, device, **kwargs,
    )


# ── Classifier windowing ────────────────────────────────────

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
    min_samples = 4

    windows = []
    for onset_ns in onset_times_ns:
        w_start = onset_ns - pre_ms * 1_000_000
        w_end = onset_ns + post_ms * 1_000_000

        idx_start = np.searchsorted(ts, w_start, side="left")
        idx_end = np.searchsorted(ts, w_end, side="right")

        if idx_end - idx_start < min_samples:
            windows.append(None)
            continue

        chunk = vals[idx_start:idx_end]
        win = resample_window(chunk, target_len)
        windows.append(win)

    return windows


# ── Classify windows ─────────────────────────────────────────

def classify_windows(
    windows: list[Optional[np.ndarray]],
    classifier: torch.nn.Module,
    classes: np.ndarray,
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
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = classifier(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    results: list[Optional[np.ndarray]] = [None] * len(windows)
    for batch_idx, orig_idx in enumerate(valid_indices):
        results[orig_idx] = probs[batch_idx]

    return results


# ── Cluster onsets into typing episodes ──────────────────────

def cluster_episodes(
    onset_times_ns: list[int],
    max_gap_ms: float = 1500.0,
) -> list[list[int]]:
    if not onset_times_ns:
        return []

    gap_ns = max_gap_ms * 1_000_000
    episodes = [[onset_times_ns[0]]]

    for t in onset_times_ns[1:]:
        if t - episodes[-1][-1] > gap_ns:
            episodes.append([t])
        else:
            episodes[-1].append(t)

    return episodes


# ── Levenshtein distance ─────────────────────────────────────

def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if c1 == c2 else 1),
            ))
        prev = curr
    return prev[-1]


# ── Full end-to-end eval on password sessions (Path A) ───────

PART_RE = re.compile(r"_part(\d+)_")


def discover_password_sessions(dirs: list[str]) -> list[str]:
    sessions = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith(".") or f.startswith("._"):
                continue
            if f.endswith("_sensor.csv") and "_free_type_" in f:
                prefix = os.path.join(d, f.replace("_sensor.csv", ""))
                if os.path.exists(prefix + "_events.csv"):
                    sessions.append(prefix)
    return sessions


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
            if key in {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
                       "left", "right", "up", "down", "delete"}:
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
        gt_times_ns = [e["timestamp_ns"] for e in seq_events]
        out.append({
            "reference": ref,
            "gt_onset_times_ns": gt_times_ns,
            "events": seq_events,
        })
    return out


SEQ_HIT_CUTOFFS = (10, 50, 100)


def eval_e2e_on_sessions(
    onset_model, onset_means, onset_stds, onset_ckpt,
    classifier, cls_classes, cls_means, cls_stds,
    device, sessions,
    threshold=0.5, nms_radius_ms=100.0,
) -> dict:
    """Full E2E evaluation: onset → classifier → metrics (Path A)."""
    onset_window_ms = onset_ckpt.get("window_ms", 150)
    onset_stride_ms = onset_ckpt.get("stride_ms", 25)
    target_rate_hz = onset_ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)

    total_chars = 0
    total_seqs = 0
    missed_chars = 0
    extra_chars = 0

    topk_correct = {"e2e": defaultdict(int), "gt": defaultdict(int)}
    total_edits = {"e2e": 0, "gt": 0}
    seq_hits = {"e2e": defaultdict(int), "gt": defaultdict(int)}

    for sess in sessions:
        sensor = load_sensor_csv(sess + "_sensor.csv")
        gt_sequences = load_ground_truth_sequences(sess)

        if not gt_sequences:
            continue

        all_onset_ns = detect_onsets_in_stream(
            sensor, onset_model, onset_means, onset_stds, device,
            onset_window_ms=onset_window_ms,
            onset_stride_ms=onset_stride_ms,
            target_rate_hz=target_rate_hz,
            threshold=threshold,
            nms_radius_ms=nms_radius_ms,
        )

        for seq in gt_sequences:
            ref = seq["reference"]
            gt_times = seq["gt_onset_times_ns"]
            total_chars += len(ref)
            total_seqs += 1

            seq_start = min(gt_times) - 200_000_000
            seq_end = max(gt_times) + 500_000_000
            e2e_onsets = [t for t in all_onset_ns if seq_start <= t <= seq_end]

            gt_match = match_events(
                [t / 1e9 for t in e2e_onsets],
                [t / 1e9 for t in gt_times],
                tolerance_s=0.100,
            )
            missed_chars += gt_match.fn
            extra_chars += gt_match.fp

            for tag, onsets in [("e2e", e2e_onsets), ("gt", gt_times)]:
                windows = cut_classifier_windows(sensor, onsets, target_rate_hz=target_rate_hz)
                prob_vecs = classify_windows(
                    windows, classifier, cls_classes, cls_means, cls_stds, device,
                )

                valid_probs = [p for p in prob_vecs if p is not None]
                if not valid_probs:
                    total_edits[tag] += len(ref)
                    continue

                prob_matrix = np.stack(valid_probs)
                hyp_chars = [cls_classes[int(np.argmax(p))] for p in valid_probs]
                hyp = "".join(hyp_chars)

                for i, ref_ch in enumerate(ref):
                    if i >= len(valid_probs):
                        break
                    ranked = np.argsort(-valid_probs[i])
                    ranked_classes = [cls_classes[r] for r in ranked]
                    for k in (1, 3, 5):
                        if ref_ch in ranked_classes[:k]:
                            topk_correct[tag][k] += 1

                total_edits[tag] += levenshtein(ref, hyp)

                try:
                    candidates = topk_strings_from_prob_vectors(
                        prob_matrix, cls_classes,
                        branch_topk=5,
                        beam_width=max(SEQ_HIT_CUTOFFS),
                    )
                    candidate_strings = [c["candidate"] for c in candidates]
                    for cutoff in SEQ_HIT_CUTOFFS:
                        if ref in candidate_strings[:cutoff]:
                            seq_hits[tag][cutoff] += 1
                except Exception:
                    pass

    def safe_div(a, b):
        return a / max(b, 1)

    def build_metrics(tag):
        m = {
            "char_top1": safe_div(topk_correct[tag][1], total_chars),
            "char_top3": safe_div(topk_correct[tag][3], total_chars),
            "char_top5": safe_div(topk_correct[tag][5], total_chars),
            "cer": safe_div(total_edits[tag], total_chars),
        }
        for cutoff in SEQ_HIT_CUTOFFS:
            m[f"sequence_top{cutoff}"] = safe_div(seq_hits[tag][cutoff], total_seqs)
        return m

    e2e_m = build_metrics("e2e")
    gt_m = build_metrics("gt")
    delta = {k: e2e_m[k] - gt_m[k] for k in e2e_m}

    results = {
        "total_sequences": total_seqs,
        "total_chars": total_chars,
        "e2e": {**e2e_m, "missed_chars": missed_chars, "extra_chars": extra_chars},
        "gt_baseline": gt_m,
        "delta": delta,
    }

    print(f"\n{'='*60}")
    print(f"  END-TO-END ATTACK CHAIN RESULTS (Path A: password sessions)")
    print(f"{'='*60}")
    print(f"  Sequences: {total_seqs}  |  Chars: {total_chars}")
    print(f"  Missed chars (onset FN): {missed_chars}  |  Extra chars (onset FP): {extra_chars}")
    for label, tag in [("E2E (onset → classifier)", "e2e"),
                       ("GT-onset baseline", "gt_baseline")]:
        m = results[tag]
        print(f"\n  {label}:")
        print(f"    char_top1: {m['char_top1']:.1%}")
        print(f"    char_top3: {m['char_top3']:.1%}")
        print(f"    char_top5: {m['char_top5']:.1%}")
        for cutoff in SEQ_HIT_CUTOFFS:
            print(f"    seq_top{cutoff}: {m[f'sequence_top{cutoff}']:.1%}")
        print(f"    CER:       {m['cer']:.1%}")
    print(f"\n  Degradation (E2E − GT):")
    for k in ["char_top1", "char_top3", "char_top5", "cer"] + \
             [f"sequence_top{c}" for c in SEQ_HIT_CUTOFFS]:
        print(f"    Δ {k}: {delta[k]:+.1%}")
    print(f"{'='*60}")

    return results


# ══════════════════════════════════════════════════════════════
# Path B: Mixed2 stream → activity segment → typing_2 → classify
# ══════════════════════════════════════════════════════════════

def eval_e2e_on_mixed2(
    onset_model, onset_means, onset_stds, onset_ckpt,
    activity_model, act_means, act_stds, act_ckpt,
    classifier, cls_classes, cls_means, cls_stds,
    device,
    mixed2_dirs: list[str],
    onset_threshold: float = 0.5,
    activity_threshold: float = 0.5,
    nms_radius_ms: float = 100.0,
) -> dict:
    """
    Full segmentation E2E on mixed2 streams.

    Four comparison paths (from most realistic to most oracle):

      e2e_full:       activity segment → typing_2 → onset detect →
                      gap-based password grouping → classify
                      (NO ground-truth information used)

      e2e_gt_seg:     GT typing_2 segment boundary → onset detect →
                      gap-based password grouping → classify
                      (GT segment boundaries, but predicted onsets & grouping)

      e2e_gt_aligned: GT typing_2 segment → onset detect →
                      GT-onset-assisted per-password alignment → classify
                      (GT boundaries + GT onset timing for alignment)

      gt_baseline:    GT onset times directly → classify
                      (full oracle)

    The key difference from the previous version: e2e_full now uses
    predicted-onset gap-based grouping with n_groups = len(passwords),
    NOT GT onset times for per-password alignment.
    """
    from onset_preprocessor import discover_sessions

    sessions = discover_sessions(mixed2_dirs, mode_filter="", dedup=False)
    if not sessions:
        print("  ⚠ No mixed2 sessions found")
        return {}

    onset_window_ms = onset_ckpt.get("window_ms", 150)
    onset_stride_ms = onset_ckpt.get("stride_ms", 25)
    act_window_ms = act_ckpt.get("window_ms", ACTIVITY_WINDOW_MS)
    act_stride_ms = act_ckpt.get("stride_ms", ACTIVITY_STRIDE_MS)
    target_rate_hz = onset_ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)

    # Accumulators for 4 paths
    tags = ["e2e_full", "e2e_gt_seg", "e2e_gt_aligned", "gt_baseline"]
    total_chars = 0
    total_seqs = 0
    topk_correct = {t: defaultdict(int) for t in tags}
    total_edits = {t: 0 for t in tags}
    seq_hits = {t: defaultdict(int) for t in tags}

    episode_results = []

    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        events_path = sess + "_events.csv"
        activity_log_path = sess + "_activity_log.csv"

        if not os.path.exists(activity_log_path):
            continue

        sensor = load_sensor_csv(sensor_path)
        activity_segments = load_activity_log(activity_log_path)

        if os.path.exists(events_path):
            events = load_events_csv(events_path, press_only=True)
        else:
            events = np.array([], dtype=np.int64)

        # Find ground-truth typing_2 segments
        gt_typing2_segs = [
            seg for seg in activity_segments
            if seg.get("label", "") == "typing_2"
        ]

        if not gt_typing2_segs:
            continue

        # Get GT passwords from protocol
        gt_passwords = []
        for seg in gt_typing2_segs:
            gt_passwords.extend(seg.get("prompts", []))
        gt_passwords = [p for p in gt_passwords if p]

        if not gt_passwords:
            continue

        n_passwords = len(gt_passwords)

        # ── Step 1: Activity segmentation ──
        act_probs, act_times = run_activity_inference(
            activity_model, sensor, act_means, act_stds, device,
            window_ms=act_window_ms,
            stride_ms=act_stride_ms,
            target_rate_hz=target_rate_hz,
        )

        pred_episodes = []
        if len(act_probs) > 0:
            pred_episodes = extract_episodes_from_activity_curve(
                act_probs, act_times,
                threshold=activity_threshold,
                min_duration_s=0.5,
                merge_gap_s=0.8,
            )

        # ── Step 2: Detect onsets across entire stream ──
        all_onset_ns = detect_onsets_in_stream(
            sensor, onset_model, onset_means, onset_stds, device,
            onset_window_ms=onset_window_ms,
            onset_stride_ms=onset_stride_ms,
            target_rate_hz=target_rate_hz,
            threshold=onset_threshold,
            nms_radius_ms=nms_radius_ms,
        )
        onset_times_s = [t / 1e9 for t in all_onset_ns]

        # Classify predicted episodes (demo-protocol heuristic)
        pred_episodes = classify_episodes_by_density(pred_episodes, onset_times_s)

        # Find predicted typing_2 episodes
        pred_typing2 = [ep for ep in pred_episodes if ep.label == "typing_2"]

        # Episode matching for boundary metrics
        gt_episodes = []
        for seg in activity_segments:
            if seg["activity"] == "keyboard":
                gt_episodes.append(Episode(
                    start_s=seg["start_time_ns"] / 1e9,
                    end_s=seg["end_time_ns"] / 1e9,
                    label=seg.get("label", "keyboard"),
                ))
        ep_match = match_episodes(pred_episodes, gt_episodes, min_iou=0.3)
        episode_results.append(ep_match)

        # ── Collect onsets for each path ──

        # e2e_full: onsets within PREDICTED typing_2 episodes
        e2e_full_onsets_ns = []
        for ep in pred_typing2:
            for t_ns in all_onset_ns:
                t_s = t_ns / 1e9
                if ep.start_s <= t_s <= ep.end_s:
                    e2e_full_onsets_ns.append(t_ns)

        # e2e_gt_seg / e2e_gt_aligned: onsets within GT typing_2 segments
        e2e_gtseg_onsets_ns = []
        for seg in gt_typing2_segs:
            for t_ns in all_onset_ns:
                if seg["start_time_ns"] <= t_ns <= seg["end_time_ns"]:
                    e2e_gtseg_onsets_ns.append(t_ns)

        # GT onset times within typing_2 segments, grouped per-password
        gt_onset_in_typing2 = []
        for seg in gt_typing2_segs:
            gt_mask = (events >= seg["start_time_ns"]) & (events <= seg["end_time_ns"])
            gt_onset_in_typing2.extend(events[gt_mask].tolist())
        gt_onset_in_typing2.sort()

        gt_groups = group_onsets_by_gap(gt_onset_in_typing2, n_groups=n_passwords)

        # ── Gap-based grouping of predicted onsets (NO GT info) ──
        e2e_full_groups = group_onsets_by_gap(
            e2e_full_onsets_ns, n_groups=n_passwords,
        )
        e2e_gtseg_groups = group_onsets_by_gap(
            e2e_gtseg_onsets_ns, n_groups=n_passwords,
        )

        # ── GT-assisted grouping for the oracle-aligned baseline ──
        # Uses GT onset group time ranges to select predicted onsets
        e2e_gt_aligned_groups = []
        for gt_grp in gt_groups:
            if not gt_grp:
                e2e_gt_aligned_groups.append([])
                continue
            pw_start = min(gt_grp) - 200_000_000
            pw_end = max(gt_grp) + 500_000_000
            aligned = [t for t in e2e_gtseg_onsets_ns if pw_start <= t <= pw_end]
            e2e_gt_aligned_groups.append(aligned)

        # ── Score each password across 4 paths ──
        for pw_idx, ref in enumerate(gt_passwords):
            if not ref:
                continue
            total_chars += len(ref)
            total_seqs += 1

            # Select onset group per path
            path_onsets = {
                "e2e_full": e2e_full_groups[pw_idx] if pw_idx < len(e2e_full_groups) else [],
                "e2e_gt_seg": e2e_gtseg_groups[pw_idx] if pw_idx < len(e2e_gtseg_groups) else [],
                "e2e_gt_aligned": e2e_gt_aligned_groups[pw_idx] if pw_idx < len(e2e_gt_aligned_groups) else [],
                "gt_baseline": gt_groups[pw_idx] if pw_idx < len(gt_groups) else [],
            }

            for tag in tags:
                onsets_for_pw = path_onsets[tag]

                windows = cut_classifier_windows(
                    sensor, onsets_for_pw, target_rate_hz=target_rate_hz,
                )
                prob_vecs = classify_windows(
                    windows, classifier, cls_classes, cls_means, cls_stds, device,
                )

                valid_probs = [p for p in prob_vecs if p is not None]
                if not valid_probs:
                    total_edits[tag] += len(ref)
                    continue

                prob_matrix = np.stack(valid_probs)
                hyp_chars = [cls_classes[int(np.argmax(p))] for p in valid_probs]
                hyp = "".join(hyp_chars)

                for i, ref_ch in enumerate(ref):
                    if i >= len(valid_probs):
                        break
                    ranked = np.argsort(-valid_probs[i])
                    ranked_classes = [cls_classes[r] for r in ranked]
                    for k in (1, 3, 5):
                        if ref_ch in ranked_classes[:k]:
                            topk_correct[tag][k] += 1

                total_edits[tag] += levenshtein(ref, hyp)

                try:
                    candidates = topk_strings_from_prob_vectors(
                        prob_matrix, cls_classes,
                        branch_topk=5,
                        beam_width=max(SEQ_HIT_CUTOFFS),
                    )
                    candidate_strings = [c["candidate"] for c in candidates]
                    for cutoff in SEQ_HIT_CUTOFFS:
                        if ref in candidate_strings[:cutoff]:
                            seq_hits[tag][cutoff] += 1
                except Exception:
                    pass

    # ── Compile results ──
    def safe_div(a, b):
        return a / max(b, 1)

    def build_metrics(tag):
        m = {
            "char_top1": safe_div(topk_correct[tag][1], total_chars),
            "char_top3": safe_div(topk_correct[tag][3], total_chars),
            "char_top5": safe_div(topk_correct[tag][5], total_chars),
            "cer": safe_div(total_edits[tag], total_chars),
        }
        for cutoff in SEQ_HIT_CUTOFFS:
            m[f"sequence_top{cutoff}"] = safe_div(seq_hits[tag][cutoff], total_seqs)
        return m

    results = {
        "total_sequences": total_seqs,
        "total_chars": total_chars,
    }

    for tag in tags:
        results[tag] = build_metrics(tag)

    # Degradation tables
    results["delta_full_vs_gt"] = {
        k: results["e2e_full"][k] - results["gt_baseline"][k]
        for k in results["gt_baseline"]
    }
    results["delta_gtseg_vs_gt"] = {
        k: results["e2e_gt_seg"][k] - results["gt_baseline"][k]
        for k in results["gt_baseline"]
    }
    results["delta_gt_aligned_vs_gt"] = {
        k: results["e2e_gt_aligned"][k] - results["gt_baseline"][k]
        for k in results["gt_baseline"]
    }

    # Episode boundary metrics
    if episode_results:
        all_ious = [iou for r in episode_results for iou in r.ious]
        all_start = [e for r in episode_results for e in r.start_errors_s]
        all_end = [e for r in episode_results for e in r.end_errors_s]
        n_sep = sum(1 for r in episode_results if r.correctly_separated)
        n_sep_total = sum(1 for r in episode_results if r.n_gt >= 2)

        results["episode_metrics"] = {
            "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "mean_start_error_ms": float(np.mean(all_start)) * 1000 if all_start else float("inf"),
            "mean_end_error_ms": float(np.mean(all_end)) * 1000 if all_end else float("inf"),
            "separation_accuracy": n_sep / max(n_sep_total, 1),
        }

    # Print
    print(f"\n{'='*60}")
    print(f"  END-TO-END MIXED2 RESULTS (Path B: segment → typing_2 → classify)")
    print(f"{'='*60}")
    print(f"  Sequences: {total_seqs}  |  Chars: {total_chars}")

    for label, tag in [
        ("Full E2E (segment → typing_2 → onset → gap-group → classify)", "e2e_full"),
        ("GT-segment (GT boundary → onset → gap-group → classify)", "e2e_gt_seg"),
        ("GT-aligned (GT boundary → onset → GT-assisted alignment → classify)", "e2e_gt_aligned"),
        ("GT-onset baseline (GT onsets → classify)", "gt_baseline"),
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
        print(f"\n  Episode boundary metrics:")
        print(f"    Mean IoU: {em['mean_iou']:.3f}")
        print(f"    Start error: {em['mean_start_error_ms']:.1f}ms")
        print(f"    End error:   {em['mean_end_error_ms']:.1f}ms")
        print(f"    Separation:  {em['separation_accuracy']:.1%}")

    print(f"\n  Degradation (Full E2E − GT baseline):")
    for k in ["char_top1", "char_top3", "char_top5", "cer"]:
        print(f"    Δ {k}: {results['delta_full_vs_gt'][k]:+.1%}")

    print(f"\n  Degradation (GT-segment − GT baseline):")
    for k in ["char_top1", "char_top3", "char_top5", "cer"]:
        print(f"    Δ {k}: {results['delta_gtseg_vs_gt'][k]:+.1%}")

    print(f"\n  Degradation (GT-aligned − GT baseline):")
    for k in ["char_top1", "char_top3", "char_top5", "cer"]:
        print(f"    Δ {k}: {results['delta_gt_aligned_vs_gt'][k]:+.1%}")

    print(f"{'='*60}")
    return results


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end onset → classifier evaluation")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--onset-checkpoint", default="results/onset_detector.pt")
    parser.add_argument("--onset-scaler", default="results/onset_scaler.npz")
    parser.add_argument("--activity-checkpoint", default="results/activity_detector.pt",
                        help="Activity segmenter checkpoint for Path B")
    parser.add_argument("--activity-scaler", default="results/activity_scaler.npz")
    parser.add_argument("--classifier-checkpoint", default="results/inception_password_final.pt")
    parser.add_argument("--classifier-scaler", default="results/inception_password_scaler.npz")
    parser.add_argument("--password-dirs", nargs="+", default=["data/raw/password/len_8"])
    parser.add_argument("--test-parts", nargs="+", type=int, default=[17, 18, 19, 20])
    parser.add_argument("--mixed2-dirs", nargs="*", default=[],
                        help="mixed2 stream dirs for Path B evaluation")
    parser.add_argument("--report", default="results/onset_e2e_report.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius-ms", type=float, default=100.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    if args.project_root:
        _setup_project_imports(args.project_root)
        root = os.path.abspath(args.project_root)
        for attr in ("onset_checkpoint", "onset_scaler",
                     "activity_checkpoint", "activity_scaler",
                     "classifier_checkpoint", "classifier_scaler", "report"):
            val = getattr(args, attr)
            if not os.path.isabs(val):
                setattr(args, attr, os.path.join(root, val))
        args.password_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                              for d in args.password_dirs]
        if args.mixed2_dirs:
            args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                                for d in args.mixed2_dirs]

    # Load onset detector
    onset_model, onset_means, onset_stds, onset_ckpt = load_onset_detector(
        args.onset_checkpoint, args.onset_scaler, device,
    )
    print(f"Onset detector: {onset_ckpt.get('model_name', 'cnn')}")

    # Load password classifier
    classifier, cls_classes, cls_means, cls_stds = load_final_inception(
        args.classifier_checkpoint, args.classifier_scaler, device,
    )
    print(f"Classifier: InceptionTime, {len(cls_classes)} classes")

    all_results = {}

    # ── Path A: password sessions ──
    all_sessions = discover_password_sessions(args.password_dirs)
    test_parts = set(args.test_parts)
    sessions = [s for s in all_sessions if parse_part(s) in test_parts]

    if sessions:
        print(f"\nPath A: {len(sessions)} password sessions (parts: {sorted(test_parts)})")
        path_a = eval_e2e_on_sessions(
            onset_model, onset_means, onset_stds, onset_ckpt,
            classifier, cls_classes, cls_means, cls_stds,
            device, sessions,
            threshold=args.threshold,
            nms_radius_ms=args.nms_radius_ms,
        )
        all_results["path_a_password"] = path_a

    # ── Path B: mixed2 streams ──
    if args.mixed2_dirs:
        # Load activity segmenter
        if os.path.exists(args.activity_checkpoint):
            act_model, act_means, act_stds, act_ckpt = load_onset_detector(
                args.activity_checkpoint, args.activity_scaler, device,
            )
            print(f"\nActivity segmenter: {act_ckpt.get('model_name', 'activity_cnn')}")

            path_b = eval_e2e_on_mixed2(
                onset_model, onset_means, onset_stds, onset_ckpt,
                act_model, act_means, act_stds, act_ckpt,
                classifier, cls_classes, cls_means, cls_stds,
                device,
                mixed2_dirs=args.mixed2_dirs,
                onset_threshold=args.threshold,
                activity_threshold=args.activity_threshold,
                nms_radius_ms=args.nms_radius_ms,
            )
            all_results["path_b_mixed2"] = path_b
        else:
            print(f"\n⚠ Activity checkpoint not found: {args.activity_checkpoint}")
            print(f"  Skipping Path B. Train with: python3 train_onset.py --task activity")

    # Save
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {args.report}")


if __name__ == "__main__":
    main()
