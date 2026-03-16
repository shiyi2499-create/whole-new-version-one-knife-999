"""
End-to-End Onset → Password Classifier Pipeline
=================================================
The full attack chain demonstration:

  1. Load a continuous IMU stream (mixed-stream or password session)
  2. Run onset detector → candidate onset timestamps
  3. At each candidate onset, cut a 300ms classifier window
     (100ms pre + 200ms post, matching the password classifier protocol)
  4. Run the password classifier on each window → per-position top-k
  5. Cluster consecutive onsets into "typing episodes"
  6. Within each episode, assemble character predictions → sequence
  7. Report char_top1/3/5 + sequence_top10/50/100 + CER
  8. Compare with ground-truth-onset baseline to measure degradation

This script does NOT retrain anything. It loads:
  - A trained onset detector checkpoint
  - A trained password classifier (InceptionTime) checkpoint

Run:
  python3 eval_onset_e2e.py \\
    --onset-checkpoint results/onset_detector.pt \\
    --onset-scaler results/onset_scaler.npz \\
    --classifier-checkpoint results/inception_password_final.pt \\
    --classifier-scaler results/inception_password_scaler.npz \\
    --password-dirs data/raw/password/len_8 \\
    --test-parts 17 18 19 20
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
    resample_window,
    window_samples,
    DEFAULT_TARGET_RATE_HZ,
)
from onset_utils import detect_peaks, nms_1d, match_events

# ── Imports from password classifier (parent project) ────────
# Resolved at runtime via --project-root or best-effort __file__ detection.

_PROJECT_ROOT = None  # set by CLI

def _setup_project_imports(project_root: str = ""):
    """Add project root and phase3 subdir to sys.path for classifier imports."""
    global _PROJECT_ROOT
    if project_root:
        root = os.path.abspath(project_root)
    else:
        # Best effort: assume onset_detection/ is a direct child of project root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _PROJECT_ROOT = root
    phase3 = os.path.join(root, "phase3_password_inception")
    for p in [root, phase3]:
        if p not in sys.path:
            sys.path.insert(0, p)

# Default setup for import-time (overridden by CLI --project-root)
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
    from onset_preprocessor import extract_sliding_windows, DEFAULT_LABEL_RADIUS_MS

    # We don't need labels, just windows + timestamps
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

    # Batch inference
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

    # Peak detection + NMS
    peaks = detect_peaks(probs, times_s, threshold=threshold, smooth_n=3)
    peaks = nms_1d(peaks, radius_s=nms_radius_ms / 1000.0)

    # Convert to nanoseconds for compatibility with classifier windowing
    onset_times_ns = [int(p["time_s"] * 1e9) for p in peaks]
    return onset_times_ns


# ── Classifier windowing ────────────────────────────────────

def cut_classifier_windows(
    sensor: np.ndarray,
    onset_times_ns: list[int],
    pre_ms: int = 100,
    post_ms: int = 200,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> list[Optional[np.ndarray]]:
    """
    For each onset time, cut a classifier-compatible window.
    Returns list of (target_len, 6) arrays or None if too few samples.
    """
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
    """
    Run password classifier on a list of windows.

    Returns per-window softmax probability vectors (n_classes,), or None
    for invalid windows.  These are consumed by both top-k character
    scoring and beam-search sequence scoring.
    """
    valid_indices = [i for i, w in enumerate(windows) if w is not None]
    if not valid_indices:
        return [None] * len(windows)

    # Stack valid windows
    X = np.stack([windows[i] for i in valid_indices]).astype(np.float32)

    # Normalise with classifier scaler
    for ch in range(X.shape[2]):
        X[:, :, ch] = (X[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)

    # Inference
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
    """
    Group consecutive onsets into episodes.
    A new episode starts when the gap exceeds max_gap_ms.
    """
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


# ── Full end-to-end eval on password sessions ────────────────

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
    """Load ground truth password sequences from events + attempts."""
    events_path = session_prefix + "_events.csv"
    attempts_path = session_prefix + "_attempts.csv"

    # Load attempts for reference strings
    attempts = []
    if os.path.exists(attempts_path):
        with open(attempts_path) as f:
            attempts = list(csv.DictReader(f))

    # Parse events into sequences (split by enter)
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


def eval_e2e_on_sessions(
    onset_model: torch.nn.Module,
    onset_means: np.ndarray,
    onset_stds: np.ndarray,
    onset_ckpt: dict,
    classifier: torch.nn.Module,
    cls_classes: np.ndarray,
    cls_means: np.ndarray,
    cls_stds: np.ndarray,
    device: torch.device,
    sessions: list[str],
    threshold: float = 0.5,
    nms_radius_ms: float = 100.0,
) -> dict:
    """
    Full E2E evaluation: onset → classifier → metrics.
    Also runs ground-truth onset baseline for comparison.
    """
    onset_window_ms = onset_ckpt.get("window_ms", 150)
    onset_stride_ms = onset_ckpt.get("stride_ms", 25)
    target_rate_hz = onset_ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)

    # Accumulators
    total_chars = 0
    total_seqs = 0
    missed_chars = 0
    extra_chars = 0
    SEQ_HIT_CUTOFFS = (10, 50, 100)

    topk_correct = {"e2e": defaultdict(int), "gt": defaultdict(int)}
    total_edits = {"e2e": 0, "gt": 0}
    seq_hits = {"e2e": defaultdict(int), "gt": defaultdict(int)}

    class_set = set(cls_classes.tolist())

    for sess in sessions:
        sensor = load_sensor_csv(sess + "_sensor.csv")
        gt_sequences = load_ground_truth_sequences(sess)

        if not gt_sequences:
            continue

        # Detect onsets in the full stream
        all_onset_ns = detect_onsets_in_stream(
            sensor, onset_model, onset_means, onset_stds, device,
            onset_window_ms=onset_window_ms,
            onset_stride_ms=onset_stride_ms,
            target_rate_hz=target_rate_hz,
            threshold=threshold,
            nms_radius_ms=nms_radius_ms,
        )

        # Process each ground-truth sequence
        for seq in gt_sequences:
            ref = seq["reference"]
            gt_times = seq["gt_onset_times_ns"]
            total_chars += len(ref)
            total_seqs += 1

            # ── E2E path: use detected onsets ──
            seq_start = min(gt_times) - 200_000_000  # 200ms margin
            seq_end = max(gt_times) + 500_000_000    # 500ms margin
            e2e_onsets = [t for t in all_onset_ns if seq_start <= t <= seq_end]

            # Count missed / extra
            gt_match = match_events(
                [t / 1e9 for t in e2e_onsets],
                [t / 1e9 for t in gt_times],
                tolerance_s=0.100,
            )
            missed_chars += gt_match.fn
            extra_chars += gt_match.fp

            # Classify both paths and score
            for tag, onsets in [("e2e", e2e_onsets), ("gt", gt_times)]:
                windows = cut_classifier_windows(sensor, onsets, target_rate_hz=target_rate_hz)
                prob_vecs = classify_windows(
                    windows, classifier, cls_classes, cls_means, cls_stds, device,
                )

                # Collect valid probability vectors
                valid_probs = [p for p in prob_vecs if p is not None]
                if not valid_probs:
                    total_edits[tag] += len(ref)
                    continue

                prob_matrix = np.stack(valid_probs)  # (n_detected, n_classes)

                # Top-1 hypothesis
                hyp_chars = [cls_classes[int(np.argmax(p))] for p in valid_probs]
                hyp = "".join(hyp_chars)

                # Character-level top-k scoring
                for i, ref_ch in enumerate(ref):
                    if i >= len(valid_probs):
                        break
                    ranked = np.argsort(-valid_probs[i])
                    ranked_classes = [cls_classes[r] for r in ranked]
                    for k in (1, 3, 5):
                        if ref_ch in ranked_classes[:k]:
                            topk_correct[tag][k] += 1

                # CER
                total_edits[tag] += levenshtein(ref, hyp)

                # Sequence-level beam search (same as password route)
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
                    pass  # graceful fallback if beam search fails

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
    print(f"  END-TO-END ATTACK CHAIN RESULTS")
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


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end onset → classifier evaluation")
    parser.add_argument("--project-root", default="",
                        help="Project root directory for resolving imports and relative paths.")
    parser.add_argument("--onset-checkpoint", default="results/onset_detector.pt")
    parser.add_argument("--onset-scaler", default="results/onset_scaler.npz")
    parser.add_argument("--classifier-checkpoint", default="results/inception_password_final.pt")
    parser.add_argument("--classifier-scaler", default="results/inception_password_scaler.npz")
    parser.add_argument("--password-dirs", nargs="+", default=["data/raw/password/len_8"])
    parser.add_argument("--test-parts", nargs="+", type=int, default=[17, 18, 19, 20],
                        help="Password parts to use for testing")
    parser.add_argument("--report", default="results/onset_e2e_report.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius-ms", type=float, default=100.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    # Re-setup imports if project root was explicitly provided
    if args.project_root:
        _setup_project_imports(args.project_root)
        root = os.path.abspath(args.project_root)
        # Resolve relative paths from project root
        for attr in ("onset_checkpoint", "onset_scaler",
                     "classifier_checkpoint", "classifier_scaler", "report"):
            val = getattr(args, attr)
            if not os.path.isabs(val):
                setattr(args, attr, os.path.join(root, val))
        args.password_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                              for d in args.password_dirs]

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

    # Discover and filter sessions
    all_sessions = discover_password_sessions(args.password_dirs)
    test_parts = set(args.test_parts)
    sessions = [s for s in all_sessions if parse_part(s) in test_parts]
    print(f"Test sessions: {len(sessions)} (parts: {sorted(test_parts)})")

    if not sessions:
        print("❌ No test sessions found!")
        sys.exit(1)

    # Run E2E eval
    results = eval_e2e_on_sessions(
        onset_model, onset_means, onset_stds, onset_ckpt,
        classifier, cls_classes, cls_means, cls_stds,
        device, sessions,
        threshold=args.threshold,
        nms_radius_ms=args.nms_radius_ms,
    )

    # Save
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {args.report}")


if __name__ == "__main__":
    main()
