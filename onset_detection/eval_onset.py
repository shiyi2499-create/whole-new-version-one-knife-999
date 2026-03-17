"""
Onset Detection & Activity Segmentation Evaluation
====================================================
Evaluate:
  1. Segment-level (held-out windows) → AUC, P/R/F1
  2. Event-level on continuous streams → Event P/R/F1, timing error, FA/min
  3. Tolerance sweep at ±25/50/75/100 ms
  4. Episode boundary metrics → start/end error, IoU, separation (NEW)
  5. Activity segmentation accuracy on mixed2 streams (NEW)

Run:
  # Onset evaluation
  python3 eval_onset.py --checkpoint results/onset_detector.pt

  # Activity segmentation evaluation on mixed2 data
  python3 eval_onset.py --task activity \\
    --checkpoint results/activity_detector.pt \\
    --scaler results/activity_scaler.npz \\
    --mixed2-dirs data/raw/onset_mixed2
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
    load_activity_log,
    extract_sliding_windows,
    extract_activity_windows,
    discover_sessions,
    window_samples,
    DEFAULT_WINDOW_MS,
    DEFAULT_STRIDE_MS,
    DEFAULT_LABEL_RADIUS_MS,
    DEFAULT_TARGET_RATE_HZ,
    ACTIVITY_WINDOW_MS,
    ACTIVITY_STRIDE_MS,
)
from onset_dataset import load_onset_dataset, session_split, OnsetWindowDataset
from onset_utils import (
    detect_peaks,
    nms_1d,
    match_events,
    false_alarms_per_minute,
    tolerance_sweep,
    format_match_result,
    Episode,
    extract_episodes_from_activity_curve,
    classify_episodes_by_density,
    episode_iou,
    match_episodes,
    format_episode_match_result,
    EpisodeMatchResult,
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
    """Load trained onset/activity detector + normalisation stats."""
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
    """Evaluate on the test split of onset/activity dataset."""
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

    task = data.get("task", "onset")
    label = "ACTIVITY" if task == "activity" else "ONSET"
    print(f"\n  SEGMENT-LEVEL {label} TEST METRICS (N={len(labels)})")
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
    """Run onset detector on a single continuous sensor stream."""
    result = extract_sliding_windows(
        sensor, event_times_ns,
        window_ms=window_ms,
        stride_ms=stride_ms,
        label_radius_ms=DEFAULT_LABEL_RADIUS_MS,
        target_rate_hz=target_rate_hz,
    )

    if len(result["windows"]) == 0:
        return {"error": "no windows extracted"}

    windows = result["windows"].astype(np.float32)
    for ch in range(windows.shape[-1]):
        windows[:, :, ch] = (windows[:, :, ch] - means[ch]) / max(stds[ch], 1e-10)

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

    peaks = detect_peaks(probs, times, threshold=threshold, smooth_n=3)
    peaks = nms_1d(peaks, radius_s=nms_radius_ms / 1000.0)
    predicted_times = [p["time_s"] for p in peaks]

    gt_times = event_times_ns.astype(np.float64) / 1e9
    duration_s = float(times[-1] - times[0]) if len(times) > 1 else 0.0

    sweep = tolerance_sweep(predicted_times, gt_times, (25, 50, 75, 100))
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


# ══════════════════════════════════════════════════════════════
# Activity segmentation evaluation on mixed2 streams (NEW)
# ══════════════════════════════════════════════════════════════

def run_activity_inference(
    model: torch.nn.Module,
    sensor: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    window_ms: int = ACTIVITY_WINDOW_MS,
    stride_ms: int = ACTIVITY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run activity segmenter on continuous sensor data.
    Returns (probs, timestamps) arrays.
    """
    result = extract_activity_windows(
        sensor, [],  # empty segments – we just want the windows
        window_ms=window_ms,
        stride_ms=stride_ms,
        target_rate_hz=target_rate_hz,
    )

    if len(result["windows"]) == 0:
        return np.array([]), np.array([])

    windows = result["windows"].astype(np.float32)
    for ch in range(windows.shape[-1]):
        windows[:, :, ch] = (windows[:, :, ch] - means[ch]) / max(stds[ch], 1e-10)

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
    return probs, times


def eval_mixed2_activity(
    activity_model: torch.nn.Module,
    mixed2_dirs: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    ckpt: dict,
    onset_model: torch.nn.Module = None,
    onset_means: np.ndarray = None,
    onset_stds: np.ndarray = None,
    onset_ckpt: dict = None,
    activity_threshold: float = 0.5,
    onset_threshold: float = 0.5,
    onset_nms_radius_ms: float = 100.0,
) -> dict:
    """
    Evaluate activity segmentation on mixed2 sessions.

    For each session:
      1. Run activity segmenter → keyboard-active probability curve
      2. Extract episodes from the curve
      3. Compare predicted episodes with ground-truth from activity_log.csv
      4. Compute: boundary errors, IoU, episode separation
      5. If onset model is provided, also classify typing_1 vs typing_2
         using the demo-protocol IKI/rate heuristic (not a learned classifier)
    """
    window_ms = ckpt.get("window_ms", ACTIVITY_WINDOW_MS)
    stride_ms = ckpt.get("stride_ms", ACTIVITY_STRIDE_MS)
    target_rate_hz = ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ)

    sessions = discover_sessions(mixed2_dirs, mode_filter="", dedup=False)
    if not sessions:
        print("  ⚠ No mixed2 sessions found")
        return {}

    print(f"\n  Found {len(sessions)} mixed2 sessions")

    all_episode_results: list[EpisodeMatchResult] = []
    per_session = []

    for sess in sessions:
        sensor_path = sess + "_sensor.csv"
        activity_log_path = sess + "_activity_log.csv"

        if not os.path.exists(activity_log_path):
            continue

        sensor = load_sensor_csv(sensor_path)
        activity_segments = load_activity_log(activity_log_path)

        # Build ground-truth episodes
        gt_episodes = []
        for seg in activity_segments:
            if seg["activity"] == "keyboard":
                gt_episodes.append(Episode(
                    start_s=seg["start_time_ns"] / 1e9,
                    end_s=seg["end_time_ns"] / 1e9,
                    label=seg.get("label", "keyboard"),
                ))

        # Run activity segmenter
        probs, times = run_activity_inference(
            activity_model, sensor, means, stds, device,
            window_ms=window_ms,
            stride_ms=stride_ms,
            target_rate_hz=target_rate_hz,
        )

        if len(probs) == 0:
            continue

        # Extract predicted episodes
        pred_episodes = extract_episodes_from_activity_curve(
            probs, times,
            threshold=activity_threshold,
            min_duration_s=0.5,
            merge_gap_s=0.8,
        )

        # If onset model available, classify episodes
        if onset_model is not None and onset_means is not None:
            events_path = sess + "_events.csv"
            if os.path.exists(events_path):
                events = load_events_csv(events_path, press_only=True)
            else:
                events = np.array([], dtype=np.int64)

            # Detect onsets within predicted episodes
            from onset_preprocessor import extract_sliding_windows
            onset_window_ms = onset_ckpt.get("window_ms", DEFAULT_WINDOW_MS) if onset_ckpt else DEFAULT_WINDOW_MS
            onset_stride_ms = onset_ckpt.get("stride_ms", DEFAULT_STRIDE_MS) if onset_ckpt else DEFAULT_STRIDE_MS

            onset_result = extract_sliding_windows(
                sensor, events,
                window_ms=onset_window_ms,
                stride_ms=onset_stride_ms,
                label_radius_ms=DEFAULT_LABEL_RADIUS_MS,
                target_rate_hz=target_rate_hz,
            )

            if len(onset_result["windows"]) > 0:
                onset_windows = onset_result["windows"].astype(np.float32)
                for ch in range(onset_windows.shape[-1]):
                    onset_windows[:, :, ch] = (
                        (onset_windows[:, :, ch] - onset_means[ch]) /
                        max(onset_stds[ch], 1e-10)
                    )

                onset_model.eval()
                onset_probs_list = []
                for i in range(0, len(onset_windows), 256):
                    batch = torch.from_numpy(onset_windows[i:i+256]).to(device)
                    with torch.no_grad():
                        logits = onset_model(batch)
                        p = torch.sigmoid(logits.squeeze(-1))
                        onset_probs_list.append(p.cpu().numpy())
                onset_probs = np.concatenate(onset_probs_list)
                onset_times = onset_result["times_s"]

                peaks = detect_peaks(onset_probs, onset_times,
                                     threshold=onset_threshold, smooth_n=3)
                peaks = nms_1d(peaks, radius_s=onset_nms_radius_ms / 1000.0)
                onset_times_s = [pk["time_s"] for pk in peaks]

                pred_episodes = classify_episodes_by_density(
                    pred_episodes, onset_times_s,
                )

        # Match predicted to ground-truth episodes
        ep_result = match_episodes(pred_episodes, gt_episodes, min_iou=0.3)
        all_episode_results.append(ep_result)

        sess_id = os.path.basename(sess)
        print(f"    {sess_id}: GT={len(gt_episodes)} Pred={len(pred_episodes)} "
              f"IoU={ep_result.mean_iou:.3f} Sep={'✓' if ep_result.correctly_separated else '✗'}")

        per_session.append({
            "session": sess_id,
            "n_gt_episodes": len(gt_episodes),
            "n_pred_episodes": len(pred_episodes),
            "n_matched": ep_result.n_matched,
            "mean_iou": ep_result.mean_iou,
            "mean_start_error_ms": ep_result.mean_start_error_ms,
            "mean_end_error_ms": ep_result.mean_end_error_ms,
            "correctly_separated": ep_result.correctly_separated,
            "pred_labels": [ep.label for ep in pred_episodes],
            "gt_labels": [ep.label for ep in gt_episodes],
        })

    # Aggregate episode metrics
    if all_episode_results:
        all_ious = [iou for r in all_episode_results for iou in r.ious]
        all_start_errs = [e for r in all_episode_results for e in r.start_errors_s]
        all_end_errs = [e for r in all_episode_results for e in r.end_errors_s]
        n_sep_correct = sum(1 for r in all_episode_results if r.correctly_separated)
        n_sep_total = sum(1 for r in all_episode_results if r.n_gt >= 2)

        aggregate = {
            "total_sessions": len(per_session),
            "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
            "mean_start_error_ms": float(np.mean(all_start_errs)) * 1000 if all_start_errs else float("inf"),
            "mean_end_error_ms": float(np.mean(all_end_errs)) * 1000 if all_end_errs else float("inf"),
            "median_start_error_ms": float(np.median(all_start_errs)) * 1000 if all_start_errs else float("inf"),
            "median_end_error_ms": float(np.median(all_end_errs)) * 1000 if all_end_errs else float("inf"),
            "separation_accuracy": n_sep_correct / max(n_sep_total, 1),
            "per_session": per_session,
        }

        print(f"\n  AGGREGATE EPISODE BOUNDARY METRICS")
        print(f"    Sessions: {len(per_session)}")
        print(f"    Mean IoU: {aggregate['mean_iou']:.3f}")
        print(f"    Start boundary error: mean={aggregate['mean_start_error_ms']:.1f}ms  "
              f"median={aggregate['median_start_error_ms']:.1f}ms")
        print(f"    End boundary error:   mean={aggregate['mean_end_error_ms']:.1f}ms  "
              f"median={aggregate['median_end_error_ms']:.1f}ms")
        print(f"    2-episode separation: {n_sep_correct}/{n_sep_total} "
              f"({aggregate['separation_accuracy']:.1%})")

        return aggregate

    return {"per_session": per_session}


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate onset detector / activity segmenter")
    parser.add_argument("--project-root", default="")
    parser.add_argument("--task", choices=["onset", "activity", "both"], default="onset",
                        help="onset: onset detection eval. activity: activity segmentation eval. "
                             "both: run both with separate checkpoints.")
    parser.add_argument("--checkpoint", default="results/onset_detector.pt")
    parser.add_argument("--scaler", default="results/onset_scaler.npz")
    parser.add_argument("--activity-checkpoint", default="results/activity_detector.pt",
                        help="Activity segmenter checkpoint (for --task activity or both)")
    parser.add_argument("--activity-scaler", default="results/activity_scaler.npz")
    parser.add_argument("--dataset", default="data/processed/onset_dataset.npz")
    parser.add_argument("--mixed-dirs", nargs="*", default=[])
    parser.add_argument("--mixed2-dirs", nargs="*", default=[],
                        help="Directories with mixed2 sessions for episode-level eval")
    parser.add_argument("--report", default="results/onset_eval_report.json")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--activity-threshold", type=float, default=0.5)
    parser.add_argument("--nms-radius-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Auto-adjust for activity task
    if args.task == "activity":
        if args.checkpoint == "results/onset_detector.pt":
            args.checkpoint = args.activity_checkpoint
        if args.scaler == "results/onset_scaler.npz":
            args.scaler = args.activity_scaler
        if args.dataset == "data/processed/onset_dataset.npz":
            args.dataset = "data/processed/activity_dataset.npz"

    if args.project_root:
        root = os.path.abspath(args.project_root)
        for attr in ("checkpoint", "scaler", "activity_checkpoint", "activity_scaler",
                      "dataset", "report"):
            val = getattr(args, attr)
            if not os.path.isabs(val):
                setattr(args, attr, os.path.join(root, val))
        if args.mixed_dirs:
            args.mixed_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                               for d in args.mixed_dirs]
        if args.mixed2_dirs:
            args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                                for d in args.mixed2_dirs]

    device = resolve_device(args.device)

    report = {"threshold": args.threshold, "nms_radius_ms": args.nms_radius_ms}

    # ── Onset evaluation ──
    if args.task in ("onset", "both"):
        model, means, stds, ckpt = load_onset_detector(args.checkpoint, args.scaler, device)
        print(f"Loaded onset detector: {ckpt.get('model_name', 'cnn')}")
        print(f"Device: {device}")

        if os.path.exists(args.dataset):
            seg_metrics = eval_segment_level(model, args.dataset, means, stds, device, args.seed)
            report["segment_level"] = seg_metrics

        if args.mixed_dirs:
            event_metrics = eval_mixed_streams(
                model, args.mixed_dirs, means, stds, device, ckpt,
                threshold=args.threshold,
                nms_radius_ms=args.nms_radius_ms,
            )
            report["event_level"] = event_metrics

    # ── Activity segmentation evaluation ──
    if args.task in ("activity", "both"):
        act_ckpt_path = args.activity_checkpoint if args.task == "both" else args.checkpoint
        act_scaler_path = args.activity_scaler if args.task == "both" else args.scaler

        act_model, act_means, act_stds, act_ckpt = load_onset_detector(
            act_ckpt_path, act_scaler_path, device,
        )
        print(f"\nLoaded activity segmenter: {act_ckpt.get('model_name', 'activity_cnn')}")

        # Segment-level eval on activity dataset
        act_dataset = args.dataset
        if args.task == "both":
            # For 'both' mode, use separate activity dataset
            act_dataset = args.dataset.replace("onset_dataset", "activity_dataset")
        if os.path.exists(act_dataset):
            act_seg_metrics = eval_segment_level(
                act_model, act_dataset, act_means, act_stds, device, args.seed,
            )
            report["activity_segment_level"] = act_seg_metrics

        # Episode-level eval on mixed2 streams
        if args.mixed2_dirs:
            # Optionally load onset detector for episode classification
            onset_model = None
            onset_means = onset_stds = onset_ckpt = None
            if args.task == "both":
                onset_model, onset_means, onset_stds, onset_ckpt = (
                    load_onset_detector(args.checkpoint, args.scaler, device)
                )

            episode_metrics = eval_mixed2_activity(
                act_model, args.mixed2_dirs, act_means, act_stds, device, act_ckpt,
                onset_model=onset_model,
                onset_means=onset_means,
                onset_stds=onset_stds,
                onset_ckpt=onset_ckpt,
                activity_threshold=args.activity_threshold,
                onset_threshold=args.threshold,
                onset_nms_radius_ms=args.nms_radius_ms,
            )
            report["episode_level"] = episode_metrics

    # Save
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved → {args.report}")


if __name__ == "__main__":
    main()
