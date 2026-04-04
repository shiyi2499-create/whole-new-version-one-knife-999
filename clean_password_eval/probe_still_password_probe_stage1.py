from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "demo_inference_api") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "demo_inference_api"))

from inference.pipeline_inference import _load_manifest, _load_stage1_model, _resolve_ckpt_path, _resolve_device
from inference.preprocess import csv_to_array, estimate_sample_rate, extract_timestamps
from onset_detection.stage2_segmental.scripts.train_eval_stage1_dense_labeling import (
    _build_dense_features,
    _infer_probs_chunked,
    extract_segments,
)


def _load_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "key": row["key"],
                    "event_type": row["event_type"],
                }
            )
    return rows


def _password_press_timestamps(events: list[dict[str, Any]]) -> list[int]:
    out: list[int] = []
    for e in events:
        if e["event_type"] != "press":
            continue
        key = str(e["key"]).lower()
        if key in {"enter", "backspace"}:
            continue
        out.append(int(e["timestamp_ns"]))
    return out


def _best_segment(segments: list[tuple[int, int, float]]) -> tuple[int, int, float] | None:
    if not segments:
        return None
    return max(segments, key=lambda s: (float(s[2]), int(s[1]) - int(s[0])))


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe still-password-still Stage1 posthoc behavior.")
    ap.add_argument("--dataset-root", default="data/raw/800hz/clean_password_probe")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.55, 0.6])
    ap.add_argument("--smooth-windows", nargs="+", type=int, default=[1, 31, 81])
    ap.add_argument("--chunk-len", type=int, default=32768)
    ap.add_argument("--chunk-overlap", type=int, default=4096)
    args = ap.parse_args()

    dataset_root = (REPO_ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    checkpoint_dir = (REPO_ROOT / args.checkpoint_dir).resolve() if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(str(checkpoint_dir))
    device = _resolve_device(manifest.get("runtime", {}).get("device", "auto"))
    cfg = dict(manifest["stage1"]["model_config"])
    stage1 = _load_stage1_model(_resolve_ckpt_path(str(checkpoint_dir), manifest["stage1"]), device, cfg)
    trained_posthoc = dict(manifest.get("stage1_posthoc", {}).get("params", {}))

    protocol_paths = sorted(dataset_root.glob("*_protocol.json"))
    if not protocol_paths:
        raise SystemExit(f"No protocol files found in {dataset_root}")

    rows: list[dict[str, Any]] = []
    for proto_path in protocol_paths:
        protocol = json.loads(proto_path.read_text(encoding="utf-8"))
        sensor_csv = Path(protocol["paths"]["sensor_csv"])
        events_csv = Path(protocol["paths"]["events_csv"])
        truth = str(protocol["reference_text"])

        csv_string = sensor_csv.read_text(encoding="utf-8")
        timestamps = extract_timestamps(csv_string)
        imu = csv_to_array(csv_string)
        sr = float(estimate_sample_rate(imu, timestamps))
        features = _build_dense_features(imu, sr, feature_mode=str(cfg["feature_mode"]))
        probs = _infer_probs_chunked(
            stage1,
            features,
            device,
            chunk_len=args.chunk_len,
            chunk_overlap=args.chunk_overlap,
        )

        events = _load_events(events_csv)
        presses = _password_press_timestamps(events)
        gt_duration_sec = None
        if presses:
            start_ns = int(presses[0] - int(round(220.0 * 1e6)))
            end_ns = int(presses[-1] + int(round(380.0 * 1e6)))
            start_frame = int(np.searchsorted(timestamps, start_ns, side="left"))
            end_frame = int(np.searchsorted(timestamps, end_ns, side="right"))
            gt_duration_sec = float(max(0, end_frame - start_frame) / max(sr, 1e-6))

        sweep_rows: list[dict[str, Any]] = []
        for smooth_w in args.smooth_windows:
            for threshold in args.thresholds:
                segs = extract_segments(
                    probs,
                    threshold=float(threshold),
                    min_length=max(1, int(round(float(trained_posthoc.get("min_segment_s", 0.5)) * sr))),
                    merge_gap=max(0, int(round(float(trained_posthoc.get("merge_gap_s", 0.0)) * sr))),
                    prob_smooth_window=int(smooth_w),
                    valley_merge_threshold=float(trained_posthoc.get("valley_merge_threshold", 0.0)),
                    valley_merge_max_gap=max(0, int(round(float(trained_posthoc.get("valley_merge_gap_s", 0.0)) * sr))),
                )
                best = _best_segment(segs)
                sweep_rows.append(
                    {
                        "smooth_window": int(smooth_w),
                        "threshold": float(threshold),
                        "num_segments": int(len(segs)),
                        "best_confidence": None if best is None else float(best[2]),
                        "best_duration_sec": None if best is None else float((best[1] - best[0]) / max(sr, 1e-6)),
                    }
                )

        trained_segments = extract_segments(
            probs,
            threshold=float(trained_posthoc["threshold"]),
            min_length=max(1, int(round(float(trained_posthoc.get("min_segment_s", 0.5)) * sr))),
            merge_gap=max(0, int(round(float(trained_posthoc.get("merge_gap_s", 0.0)) * sr))),
            prob_smooth_window=int(trained_posthoc.get("prob_smooth_window", 1)),
            valley_merge_threshold=float(trained_posthoc.get("valley_merge_threshold", 0.0)),
            valley_merge_max_gap=max(0, int(round(float(trained_posthoc.get("valley_merge_gap_s", 0.0)) * sr))),
        )
        trained_best = _best_segment(trained_segments)

        rows.append(
            {
                "session_prefix": protocol["session_prefix"],
                "truth": truth,
                "sample_rate_est": sr,
                "probs_max": float(np.max(probs)),
                "probs_mean": float(np.mean(probs)),
                "probs_q99": float(np.quantile(probs, 0.99)),
                "gt_duration_sec": gt_duration_sec,
                "trained_posthoc": {
                    "params": trained_posthoc,
                    "num_segments": int(len(trained_segments)),
                    "best_duration_sec": None if trained_best is None else float((trained_best[1] - trained_best[0]) / max(sr, 1e-6)),
                    "best_confidence": None if trained_best is None else float(trained_best[2]),
                },
                "sweep": sweep_rows,
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "checkpoint_dir": str(checkpoint_dir),
        "n_trials": len(rows),
        "mean_probs_max": float(np.mean([r["probs_max"] for r in rows])),
        "mean_probs_q99": float(np.mean([r["probs_q99"] for r in rows])),
        "trained_posthoc_detected": int(sum(r["trained_posthoc"]["num_segments"] > 0 for r in rows)),
        "trained_posthoc_mean_best_duration_sec": float(
            np.mean([r["trained_posthoc"]["best_duration_sec"] for r in rows if r["trained_posthoc"]["best_duration_sec"] is not None])
        ) if any(r["trained_posthoc"]["best_duration_sec"] is not None for r in rows) else None,
    }

    (output_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows = [
        {
            "session_prefix": r["session_prefix"],
            "truth": r["truth"],
            "sample_rate_est": r["sample_rate_est"],
            "probs_max": r["probs_max"],
            "probs_mean": r["probs_mean"],
            "probs_q99": r["probs_q99"],
            "gt_duration_sec": r["gt_duration_sec"],
            "trained_num_segments": r["trained_posthoc"]["num_segments"],
            "trained_best_duration_sec": r["trained_posthoc"]["best_duration_sec"],
            "trained_best_confidence": r["trained_posthoc"]["best_confidence"],
        }
        for r in rows
    ]
    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
