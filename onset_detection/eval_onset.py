"""
Onset Detection Evaluation
===========================
Evaluate a trained onset detector on:
  1. Segment-level (held-out sliding windows) → AUC, P/R/F1
  2. Event-level on continuous streams → Event P/R/F1, timing error, FA/min
  3. Tolerance sweep at ±25/50/75/100 ms

Can process:
  - The test split of onset_dataset.npz (segment-level)
  - Raw mixed-stream sessions from onset_collector.py (event-level)

Run:
  python3 eval_onset.py
  python3 eval_onset.py --mixed-dirs data/raw/onset_mixed
  python3 eval_onset.py --threshold 0.6 --nms-radius-ms 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import numpy as np
import torch

from onset_model import build_onset_model
from onset_preprocessor import (
    load_sensor_csv,
    load_events_csv,
    extract_sliding_windows,
    discover_sessions,
    window_samples,
    DEFAULT_WINDOW_MS,
    DEFAULT_STRIDE_MS,
    DEFAULT_LABEL_RADIUS_MS,
    DEFAULT_TARGET_RATE_HZ,
)
from onset_dataset import load_onset_dataset, session_split, OnsetWindowDataset
from onset_utils import (
    detect_peaks,
    nms_1d,
    match_events,
    false_alarms_per_minute,
    tolerance_sweep,
    format_match_result,
)


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


def load_onset_detector(checkpoint_path: str, scaler_path: str, device: torch.device):
    """Load trained onset detector + normalisation stats."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_onset_model(
        ckpt.get("model_name", "cnn"),
        n_channels=ckpt.get("n_channels", 6),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = np.load(scaler_path)
    return model, scaler["means"], scaler["stds"], ckpt


# ── Segment-level eval ───────────────────────────────────────

def eval_segment_level(
    model: torch.nn.Module,
    dataset_path: str,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Evaluate on the test split of onset_dataset.npz."""
    from train_onset import compute_binary_metrics

    data = load_onset_dataset(dataset_path)
    _, _, test_idx = session_split(data["sessions"], data["labels"], seed=seed)

    if len(test_idx) == 0:
        print("  ⚠ No test samples available")
        return {}

    ds = OnsetWindowDataset(
        data["windows"][test_idx], data["labels"][test_idx],
        augment=False, normalize=True, means=means, stds=stds,
    )

    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    metrics = compute_binary_metrics(probs, labels)

    print(f"\n  SEGMENT-LEVEL TEST METRICS (N={len(labels)})")
    print(f"    AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}  "
          f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
          f"Acc={metrics['accuracy']:.3f}")
    print(f"    pos={metrics['n_pos']}  neg={metrics['n_neg']}")

    return metrics


# ── Event-level eval on continuous streams ────────────────────

def eval_stream(
    model: torch.nn.Module,
    sensor: np.ndarray,
    event_times_ns: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    window_ms: int = DEFAULT_WINDOW_MS,
    stride_ms: int = DEFAULT_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
    threshold: float = 0.5,
    nms_radius_ms: float = 100.0,
) -> dict:
    """
    Run onset detector on a single continuous sensor stream.
    Returns predicted onset times and event-level metrics.
    """
    # Extract sliding windows (label_radius doesn't matter for inference)
    result = extract_sliding_windows(
        sensor, event_times_ns,
        window_ms=window_ms,
        stride_ms=stride_ms,
        label_radius_ms=DEFAULT_LABEL_RADIUS_MS,
        target_rate_hz=target_rate_hz,
    )

    if len(result["windows"]) == 0:
        return {"error": "no windows extracted"}

    # Normalise
    windows = result["windows"].astype(np.float32)
    for ch in range(windows.shape[-1]):
        windows[:, :, ch] = (windows[:, :, ch] - means[ch]) / max(stds[ch], 1e-10)

    # Inference
    model.eval()
    all_probs = []
    batch_size = 256
    for i in range(0, len(windows), batch_size):
        batch = torch.from_numpy(windows[i:i+batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs)
    times = result["times_s"]

    # Peak detection + NMS
    peaks = detect_peaks(probs, times, threshold=threshold, smooth_n=3)
    peaks = nms_1d(peaks, radius_s=nms_radius_ms / 1000.0)
    predicted_times = [p["time_s"] for p in peaks]

    # Ground truth
    gt_times = event_times_ns.astype(np.float64) / 1e9

    # Duration
    duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    # Event matching at multiple tolerances
    sweep = tolerance_sweep(predicted_times, gt_times, (25, 50, 75, 100))

    # Primary metric at 50ms
    m50 = match_events(predicted_times, gt_times, tolerance_s=0.050)

    return {
        "n_predicted": len(predicted_times),
        "n_ground_truth": len(gt_times),
        "duration_s": duration_s,
        "tolerance_sweep": sweep,
        "event_f1_50ms": m50.f1,
        "event_precision_50ms": m50.precision,
        "event_recall_50ms": m50.recall,
        "timing_error_mean_ms": m50.timing_error_mean * 1000,
        "timing_error_median_ms": m50.timing_error_median * 1000,
        "false_alarms_per_min": false_alarms_per_minute(m50.fp, duration_s),
        "predicted_times_s": predicted_times,
        "gt_times_s": gt_times.tolist(),
        "probs": probs.tolist(),
        "times_s": times.tolist(),
    }


def eval_mixed_streams(
    model: torch.nn.Module,
    mixed_dirs: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    ckpt: dict,
    threshold: float = 0.5,
    nms_radius_ms: float = 100.0,
) -> dict:
    """Evaluate onset detector on mixed-stream evaluation sessions."""
    window_ms = ckpt.get("window_ms", DEFAULT_WINDOW_MS)
    stride_ms = ckpt.get("stride_ms", DEFAULT_STRIDE_MS)
    target_rate_hz = ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)

    sessions = discover_sessions(mixed_dirs, mode_filter="")
    if not sessions:
        print("  ⚠ No mixed-stream sessions found")
        return {}

    print(f"\n  Found {len(sessions)} mixed-stream sessions")

    all_pred_times = []
    all_gt_times = []
    total_duration = 0.0
    per_session = []

    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        events_path = sess + "_events.csv"

        sensor = load_sensor_csv(sensor_path)
        if os.path.exists(events_path):
            events = load_events_csv(events_path, press_only=True)
        else:
            events = np.array([], dtype=np.int64)

        result = eval_stream(
            model, sensor, events, means, stds, device,
            window_ms=window_ms,
            stride_ms=stride_ms,
            target_rate_hz=target_rate_hz,
            threshold=threshold,
            nms_radius_ms=nms_radius_ms,
        )

        if "error" in result:
            continue

        sess_id = os.path.basename(sess)
        print(f"    {sess_id}: pred={result['n_predicted']} gt={result['n_ground_truth']} "
              f"F1@50ms={result['event_f1_50ms']:.3f}")

        all_pred_times.extend(result["predicted_times_s"])
        all_gt_times.extend(result["gt_times_s"])
        total_duration += result["duration_s"]
        per_session.append({
            "session": sess_id,
            "n_predicted": result["n_predicted"],
            "n_ground_truth": result["n_ground_truth"],
            "event_f1_50ms": result["event_f1_50ms"],
            "event_precision_50ms": result["event_precision_50ms"],
            "event_recall_50ms": result["event_recall_50ms"],
            "timing_error_mean_ms": result["timing_error_mean_ms"],
            "false_alarms_per_min": result["false_alarms_per_min"],
        })

    # Aggregate
    if not all_pred_times and not all_gt_times:
        return {"per_session": per_session}

    sweep = tolerance_sweep(all_pred_times, all_gt_times, (25, 50, 75, 100))
    m50 = match_events(all_pred_times, all_gt_times, tolerance_s=0.050)

    aggregate = {
        "total_streams": len(per_session),
        "total_predicted": len(all_pred_times),
        "total_ground_truth": len(all_gt_times),
        "total_duration_s": total_duration,
        "aggregate_event_f1_50ms": m50.f1,
        "aggregate_event_precision_50ms": m50.precision,
        "aggregate_event_recall_50ms": m50.recall,
        "aggregate_timing_error_mean_ms": m50.timing_error_mean * 1000,
        "aggregate_timing_error_median_ms": m50.timing_error_median * 1000,
        "aggregate_fa_per_min": false_alarms_per_minute(m50.fp, total_duration),
        "tolerance_sweep": sweep,
        "per_session": per_session,
    }

    print(f"\n  AGGREGATE MIXED-STREAM EVENT METRICS")
    print(f"    Streams: {len(per_session)}  Duration: {total_duration:.1f}s")
    print(f"    Predicted: {len(all_pred_times)}  Ground truth: {len(all_gt_times)}")
    print(format_match_result(m50, "@ ±50ms"))
    print(f"    FA/min: {aggregate['aggregate_fa_per_min']:.2f}")

    print(f"\n  Tolerance sweep:")
    for tol_ms, metrics in sweep.items():
        print(f"    ±{tol_ms}ms: P={metrics['precision']:.3f} "
              f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    return aggregate


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate onset detector")
    parser.add_argument("--project-root", default="",
                        help="Project root directory. Relative paths resolve from here.")
    parser.add_argument("--checkpoint", default="results/onset_detector.pt")
    parser.add_argument("--scaler", default="results/onset_scaler.npz")
    parser.add_argument("--dataset", default="data/processed/onset_dataset.npz",
                        help="Onset dataset for segment-level eval")
    parser.add_argument("--mixed-dirs", nargs="*", default=[],
                        help="Directories with mixed-stream sessions for event-level eval")
    parser.add_argument("--report", default="results/onset_eval_report.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve relative paths from project root
    if args.project_root:
        root = os.path.abspath(args.project_root)
        for attr in ("checkpoint", "scaler", "dataset", "report"):
            val = getattr(args, attr)
            if not os.path.isabs(val):
                setattr(args, attr, os.path.join(root, val))
        if args.mixed_dirs:
            args.mixed_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                               for d in args.mixed_dirs]

    device = resolve_device(args.device)
    model, means, stds, ckpt = load_onset_detector(args.checkpoint, args.scaler, device)
    print(f"Loaded onset detector: {ckpt.get('model_name', 'cnn')}")
    print(f"Device: {device}")

    report = {"threshold": args.threshold, "nms_radius_ms": args.nms_radius_ms}

    # 1. Segment-level
    if os.path.exists(args.dataset):
        seg_metrics = eval_segment_level(model, args.dataset, means, stds, device, args.seed)
        report["segment_level"] = seg_metrics

    # 2. Event-level on mixed streams
    if args.mixed_dirs:
        event_metrics = eval_mixed_streams(
            model, args.mixed_dirs, means, stds, device, ckpt,
            threshold=args.threshold,
            nms_radius_ms=args.nms_radius_ms,
        )
        report["event_level"] = event_metrics

    # Save
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved → {args.report}")


if __name__ == "__main__":
    main()
