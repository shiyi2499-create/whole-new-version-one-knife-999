"""
Onset / Password-Boundary Evaluation
====================================

Supports:
  - segment-level held-out evaluation
  - onset event-level evaluation on continuous streams
  - mixed2 password episode boundary evaluation

New main task:
  `--task password_boundary`
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch

from onset_dataset import OnsetWindowDataset, load_onset_dataset, session_split
from onset_model import build_onset_model
from onset_preprocessor import (
    ACTIVITY_STRIDE_MS,
    ACTIVITY_WINDOW_MS,
    DEFAULT_LABEL_RADIUS_MS,
    DEFAULT_STRIDE_MS,
    DEFAULT_TARGET_RATE_HZ,
    DEFAULT_WINDOW_MS,
    PASSWORD_BOUNDARY_STRIDE_MS,
    PASSWORD_BOUNDARY_WINDOW_MS,
    discover_sessions,
    extract_password_boundary_windows,
    extract_sliding_windows,
    get_password_segments_from_activity_log,
    refine_password_segments_with_events,
    load_activity_log,
    load_events_csv,
    load_sensor_csv,
)
from onset_utils import (
    Episode,
    PASSWORD_BOUNDARY_LABELS,
    decode_password_boundary_predictions,
    detect_peaks,
    false_alarms_per_minute,
    format_episode_match_result,
    format_match_result,
    match_episodes,
    match_events,
    nms_1d,
    tolerance_sweep,
)
from train_onset import compute_binary_metrics, compute_multiclass_metrics


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


# ── Generic batched inference ────────────────────────────────

def infer_windows(model, windows: np.ndarray, means: np.ndarray, stds: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    X = windows.astype(np.float32).copy()
    for ch in range(X.shape[-1]):
        X[:, :, ch] = (X[:, :, ch] - means[ch]) / max(stds[ch], 1e-10)
    outputs = []
    for i in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[i:i + batch_size]).to(device)
        with torch.no_grad():
            logits = model(batch)
            if logits.ndim == 2 and logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = torch.sigmoid(logits.squeeze(-1))
        outputs.append(probs.cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([])


# ── Segment-level evaluation ─────────────────────────────────

def eval_segment_level(
    model: torch.nn.Module,
    dataset_path: str,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    seed: int = 42,
) -> dict:
    data = load_onset_dataset(dataset_path)
    _, _, test_idx = session_split(data["sessions"], data["labels"], seed=seed)
    if len(test_idx) == 0:
        print("  ⚠ No test split available")
        return {}

    ds = OnsetWindowDataset(
        data["windows"][test_idx],
        data["labels"][test_idx],
        augment=False,
        normalize=True,
        means=means,
        stds=stds,
        n_classes=data["n_classes"],
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

    probs_parts = []
    labels_parts = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            if logits.ndim == 2 and logits.shape[1] > 1:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = torch.sigmoid(logits.squeeze(-1))
            probs_parts.append(probs.cpu().numpy())
            labels_parts.append(yb.numpy())

    probs = np.concatenate(probs_parts) if probs_parts else np.array([])
    labels = np.concatenate(labels_parts) if labels_parts else np.array([])

    if data["n_classes"] > 2:
        metrics = compute_multiclass_metrics(probs, labels, data["label_names"])
        print(f"\n  SEGMENT-LEVEL {data['task'].upper()} TEST METRICS (N={len(labels)})")
        print(f"    macroF1={metrics['macro_f1']:.3f}  weightedF1={metrics['weighted_f1']:.3f}  Acc={metrics['accuracy']:.3f}")
        for cls_name, cls_m in metrics["per_class"].items():
            print(f"    {cls_name:16s} P={cls_m['precision']:.3f}  R={cls_m['recall']:.3f}  F1={cls_m['f1']:.3f}  n={cls_m['support']}")
    else:
        metrics = compute_binary_metrics(probs, labels)
        print(f"\n  SEGMENT-LEVEL {data['task'].upper()} TEST METRICS (N={len(labels)})")
        print(
            f"    AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}  "
            f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  Acc={metrics['accuracy']:.3f}"
        )
    return metrics


# ── Onset event-level continuous-stream evaluation ───────────

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
    result = extract_sliding_windows(
        sensor,
        np.array([], dtype=np.int64),
        window_ms=window_ms,
        stride_ms=stride_ms,
        label_radius_ms=DEFAULT_LABEL_RADIUS_MS,
        target_rate_hz=target_rate_hz,
    )
    if len(result["windows"]) == 0:
        return {"error": "no windows extracted"}

    probs = infer_windows(model, result["windows"], means, stds, device)
    times = result["times_s"]
    peaks = detect_peaks(np.asarray(probs).reshape(-1), times, threshold=threshold, smooth_n=3)
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
        "timing_error_mean_ms": m50.timing_error_mean * 1000.0,
        "timing_error_median_ms": m50.timing_error_median * 1000.0,
        "false_alarms_per_min": false_alarms_per_minute(m50.fp, duration_s),
        "predicted_times_s": predicted_times,
        "gt_times_s": gt_times.tolist(),
    }


# ── Password-boundary inference / mixed2 eval ────────────────

def run_password_boundary_inference(
    boundary_model: torch.nn.Module,
    sensor: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    window_ms: int = PASSWORD_BOUNDARY_WINDOW_MS,
    stride_ms: int = PASSWORD_BOUNDARY_STRIDE_MS,
    target_rate_hz: int = DEFAULT_TARGET_RATE_HZ,
):
    result = extract_sliding_windows(
        sensor,
        np.array([], dtype=np.int64),
        window_ms=window_ms,
        stride_ms=stride_ms,
        label_radius_ms=0,
        target_rate_hz=target_rate_hz,
    )
    if len(result["windows"]) == 0:
        return np.zeros((0, len(PASSWORD_BOUNDARY_LABELS)), dtype=np.float32), np.array([], dtype=np.float64)
    probs = infer_windows(boundary_model, result["windows"], means, stds, device)
    return np.asarray(probs), result["times_s"]


# legacy alias kept so older code still imports successfully
run_activity_inference = run_password_boundary_inference



def eval_mixed2_password_boundary(
    boundary_model: torch.nn.Module,
    mixed2_dirs: list[str],
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    ckpt: dict,
    password_threshold: float = 0.5,
    start_end_threshold: float = 0.3,
    min_duration_s: float = 0.4,
    merge_gap_s: float = 0.35,
) -> dict:
    sessions = discover_sessions(mixed2_dirs, mode_filter="mixed2", dedup=False) or discover_sessions(mixed2_dirs, mode_filter="", dedup=False)
    if not sessions:
        print("  ⚠ No mixed2 sessions found")
        return {}

    window_ms = int(ckpt.get("window_ms", PASSWORD_BOUNDARY_WINDOW_MS))
    stride_ms = int(ckpt.get("stride_ms", PASSWORD_BOUNDARY_STRIDE_MS))
    target_rate_hz = int(ckpt.get("target_rate_hz", DEFAULT_TARGET_RATE_HZ))

    all_matches: list = []
    per_session = []

    print(f"\n  Found {len(sessions)} mixed2 sessions")
    for sess in sessions:
        activity_log_path = sess + "_activity_log.csv"
        if not os.path.exists(activity_log_path):
            continue
        events_path = sess + "_events.csv"
        if not os.path.exists(events_path):
            continue
        sensor = load_sensor_csv(sess + "_sensor.csv")
        activity_segments = load_activity_log(activity_log_path)
        event_times_ns = load_events_csv(events_path, press_only=True)
        gt_password_segs = refine_password_segments_with_events(activity_segments, event_times_ns)
        gt_episodes = [Episode(start_s=seg["start_time_ns"] / 1e9, end_s=seg["end_time_ns"] / 1e9, label="password") for seg in gt_password_segs]
        if not gt_episodes:
            continue

        probs, times = run_password_boundary_inference(
            boundary_model,
            sensor,
            means,
            stds,
            device,
            window_ms=window_ms,
            stride_ms=stride_ms,
            target_rate_hz=target_rate_hz,
        )
        pred_episodes = decode_password_boundary_predictions(
            probs,
            times,
            password_threshold=password_threshold,
            start_end_threshold=start_end_threshold,
            min_duration_s=min_duration_s,
            merge_gap_s=merge_gap_s,
        )
        match = match_episodes(pred_episodes, gt_episodes, min_iou=0.3)
        all_matches.append(match)
        sess_id = os.path.basename(sess)
        print(format_episode_match_result(match, label=sess_id))
        per_session.append({
            "session": sess_id,
            "n_gt": match.n_gt,
            "n_pred": match.n_pred,
            "n_matched": match.n_matched,
            "mean_iou": match.mean_iou,
            "mean_start_error_ms": match.mean_start_error_ms,
            "mean_end_error_ms": match.mean_end_error_ms,
        })

    if not all_matches:
        return {"per_session": per_session}

    all_ious = [v for m in all_matches for v in m.ious]
    all_start = [v for m in all_matches for v in m.start_errors_s]
    all_end = [v for m in all_matches for v in m.end_errors_s]
    n_gt = sum(m.n_gt for m in all_matches)
    n_pred = sum(m.n_pred for m in all_matches)
    n_matched = sum(m.n_matched for m in all_matches)

    summary = {
        "total_streams": len(all_matches),
        "episode_precision": n_matched / max(n_pred, 1),
        "episode_recall": n_matched / max(n_gt, 1),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "mean_start_error_ms": float(np.mean(all_start)) * 1000.0 if all_start else float("inf"),
        "mean_end_error_ms": float(np.mean(all_end)) * 1000.0 if all_end else float("inf"),
        "per_session": per_session,
    }

    print(f"\n{'='*60}")
    print("  MIXED2 PASSWORD EPISODE BOUNDARY METRICS")
    print(f"{'='*60}")
    print(f"  Streams:          {summary['total_streams']}")
    print(f"  Episode precision {summary['episode_precision']:.3f}")
    print(f"  Episode recall    {summary['episode_recall']:.3f}")
    print(f"  Mean IoU          {summary['mean_iou']:.3f}")
    print(f"  Start error       {summary['mean_start_error_ms']:.1f}ms")
    print(f"  End error         {summary['mean_end_error_ms']:.1f}ms")
    print(f"{'='*60}")
    return summary


# ── CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate onset / password-boundary models")
    parser.add_argument("--task", choices=["onset", "password_boundary", "activity", "both"], default="onset")
    parser.add_argument("--checkpoint", default="results/onset_detector.pt")
    parser.add_argument("--scaler", default="results/onset_scaler.npz")
    parser.add_argument("--boundary-checkpoint", default="results/password_boundary_detector.pt")
    parser.add_argument("--boundary-scaler", default="results/password_boundary_scaler.npz")
    parser.add_argument("--dataset", default="data/processed/onset_dataset.npz")
    parser.add_argument("--mixed2-dirs", nargs="*", default=[])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--boundary-gap-ms", type=float, default=600.0, help="Allow brief internal pauses inside one password episode.")
    parser.add_argument("--nms-radius-ms", type=float, default=100.0)
    parser.add_argument("--report", default="results/onset_eval_report.json")
    args = parser.parse_args()

    device = resolve_device(args.device)
    report = {}

    if args.task in ("onset", "both"):
        onset_model, onset_means, onset_stds, onset_ckpt = load_detector(args.checkpoint, args.scaler, device)
        print(f"Loaded onset model: {onset_ckpt.get('model_name', 'cnn')}")
        if os.path.exists(args.dataset):
            report["onset_segment_level"] = eval_segment_level(onset_model, args.dataset, onset_means, onset_stds, device)

    if args.task in ("password_boundary", "activity", "both"):
        if args.task == "password_boundary":
            boundary_ckpt_path = args.checkpoint
            boundary_scaler_path = args.scaler
            if args.dataset == "data/processed/onset_dataset.npz":
                args.dataset = "data/processed/password_boundary_dataset.npz"
        else:
            boundary_ckpt_path = args.boundary_checkpoint
            boundary_scaler_path = args.boundary_scaler
        boundary_model, boundary_means, boundary_stds, boundary_ckpt = load_detector(boundary_ckpt_path, boundary_scaler_path, device)
        print(f"Loaded password-boundary model: {boundary_ckpt.get('model_name', 'password_boundary_cnn')}")
        if os.path.exists(args.dataset):
            report["password_boundary_segment_level"] = eval_segment_level(boundary_model, args.dataset, boundary_means, boundary_stds, device)
        if args.mixed2_dirs:
            report["mixed2_password_boundary"] = eval_mixed2_password_boundary(
                boundary_model,
                args.mixed2_dirs,
                boundary_means,
                boundary_stds,
                device,
                boundary_ckpt,
                password_threshold=args.threshold,
                merge_gap_s=args.boundary_gap_ms / 1000.0,
            )

    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved → {args.report}")


if __name__ == "__main__":
    main()
