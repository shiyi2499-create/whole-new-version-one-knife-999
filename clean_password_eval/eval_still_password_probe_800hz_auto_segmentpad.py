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

load_all_models = None
run_ctc = None
run_pipeline_stage23 = None
run_stage1 = None
csv_to_array = None
estimate_sample_rate = None
extract_timestamps = None


def levenshtein(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / float(len(ref))


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


def _choose_stage1_segment(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not segments:
        return None
    return max(segments, key=lambda s: (float(s.get("confidence", 0.0)), int(s["end"]) - int(s["start"])))


def _event_timing(events: list[dict[str, Any]], reference_text: str) -> dict[str, Any]:
    presses = [e for e in events if e["event_type"] == "press" and e["key"] not in {"enter", "backspace"}]
    typed = []
    for e in presses:
        if e["key"] == "space":
            typed.append(" ")
        elif len(e["key"]) == 1:
            typed.append(e["key"])
    press_ts = [int(e["timestamp_ns"]) for e in presses]
    intervals_ms = [
        (press_ts[i + 1] - press_ts[i]) / 1_000_000.0
        for i in range(len(press_ts) - 1)
    ]
    return {
        "typed_press_text": "".join(typed),
        "n_key_presses_ex_enter": len(press_ts),
        "reference_length": len(reference_text),
        "mean_inter_key_ms": None if not intervals_ms else float(np.mean(intervals_ms)),
        "median_inter_key_ms": None if not intervals_ms else float(np.median(intervals_ms)),
    }


def _pack_prediction(pipe: dict[str, Any], ctc: dict[str, Any], truth: str) -> dict[str, Any]:
    pipe_pred = str(pipe.get("char_top1", ""))
    ctc_pred = str(ctc.get("prediction", ""))
    return {
        "pipeline_prediction": pipe_pred,
        "pipeline_num_keys": int(pipe.get("num_keys", 0)),
        "pipeline_cer": cer(truth, pipe_pred),
        "pipeline_top_candidate": "" if not pipe.get("top_candidates") else str(pipe["top_candidates"][0]["password"]),
        "pipeline_top100_hit": bool(any(str(x.get("password", "")) == truth for x in pipe.get("top_candidates", [])[:100])),
        "ctc_prediction": ctc_pred,
        "ctc_cer": cer(truth, ctc_pred),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate still probe with post-Stage1 segment padding.")
    parser.add_argument("--dataset-root", default="data/raw/800hz/clean_password_probe")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--segment-pad-sec", type=float, default=0.0)
    args = parser.parse_args()

    global load_all_models, run_ctc, run_pipeline_stage23, run_stage1
    global csv_to_array, estimate_sample_rate, extract_timestamps
    from inference.pipeline_inference_800hz_demo import load_all_models, run_ctc, run_pipeline_stage23, run_stage1
    from inference.preprocess import csv_to_array, estimate_sample_rate, extract_timestamps

    dataset_root = (REPO_ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    checkpoint_dir = (REPO_ROOT / args.checkpoint_dir).resolve() if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_paths = sorted(dataset_root.glob("*_protocol.json"))
    if not protocol_paths:
        raise SystemExit(f"No protocol files found in {dataset_root}")

    models = load_all_models(str(checkpoint_dir))
    rows: list[dict[str, Any]] = []

    for proto_path in protocol_paths:
        protocol = json.loads(proto_path.read_text(encoding="utf-8"))
        truth = str(protocol["reference_text"])
        sensor_csv = Path(protocol["paths"]["sensor_csv"])
        events_csv = Path(protocol["paths"]["events_csv"])

        csv_string = sensor_csv.read_text(encoding="utf-8")
        timestamps = extract_timestamps(csv_string)
        imu = csv_to_array(csv_string)
        orig_hz = float(estimate_sample_rate(imu, timestamps))
        events = _load_events(events_csv)
        timing = _event_timing(events, truth)

        segments = run_stage1(imu, models, source_hz=orig_hz)
        stage1_best = _choose_stage1_segment(segments)
        auto_pack: dict[str, Any] | None = None

        if stage1_best is not None:
            pad_frames = int(round(float(args.segment_pad_sec) * orig_hz))
            lo = max(0, int(stage1_best["start"]) - pad_frames)
            hi = min(len(imu), int(stage1_best["end"]) + pad_frames)
            seg = imu[lo:hi]
            pipe = run_pipeline_stage23(seg, models, beam_width=args.beam_width, source_hz=orig_hz)
            ctc = run_ctc(seg, models, source_hz=orig_hz)
            auto_pack = {
                "segment_start": lo,
                "segment_end": hi,
                "segment_duration_sec": float((hi - lo) / max(orig_hz, 1e-6)),
                **_pack_prediction(pipe, ctc, truth),
            }

        rows.append(
            {
                "session_prefix": protocol["session_prefix"],
                "truth": truth,
                "sample_rate_est": orig_hz,
                **timing,
                "stage1_num_segments": len(segments),
                "stage1_best": stage1_best,
                "segment_pad_sec": float(args.segment_pad_sec),
                "auto_fullsample": auto_pack,
            }
        )

    def _mean_from(path: tuple[str, ...]) -> float | None:
        vals = []
        for row in rows:
            cur: Any = row
            ok = True
            for key in path:
                if not isinstance(cur, dict) or key not in cur or cur[key] is None:
                    ok = False
                    break
                cur = cur[key]
            if ok:
                vals.append(float(cur))
        return None if not vals else float(sum(vals) / len(vals))

    summary = {
        "dataset_root": str(dataset_root),
        "checkpoint_dir": str(checkpoint_dir),
        "segment_pad_sec": float(args.segment_pad_sec),
        "n_trials": len(rows),
        "auto_pipeline_mean_cer": _mean_from(("auto_fullsample", "pipeline_cer")),
        "auto_ctc_mean_cer": _mean_from(("auto_fullsample", "ctc_cer")),
        "auto_pipeline_exact": int(sum((r.get("auto_fullsample") or {}).get("pipeline_prediction", "") == r["truth"] for r in rows)),
        "auto_ctc_exact": int(sum((r.get("auto_fullsample") or {}).get("ctc_prediction", "") == r["truth"] for r in rows)),
        "auto_pipeline_top100_hit": int(sum(bool((r.get("auto_fullsample") or {}).get("pipeline_top100_hit")) for r in rows)),
        "mean_segment_duration_sec": _mean_from(("auto_fullsample", "segment_duration_sec")),
    }

    (output_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        flat_rows = []
        for row in rows:
            block = row.get("auto_fullsample") or {}
            flat_rows.append(
                {
                    "session_prefix": row["session_prefix"],
                    "truth": row["truth"],
                    "stage1_num_segments": row["stage1_num_segments"],
                    "segment_pad_sec": row["segment_pad_sec"],
                    "segment_duration_sec": block.get("segment_duration_sec"),
                    "pipeline_prediction": block.get("pipeline_prediction"),
                    "pipeline_num_keys": block.get("pipeline_num_keys"),
                    "pipeline_cer": block.get("pipeline_cer"),
                    "pipeline_top100_hit": block.get("pipeline_top100_hit"),
                }
            )
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
