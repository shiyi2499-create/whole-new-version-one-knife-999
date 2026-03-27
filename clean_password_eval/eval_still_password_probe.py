from __future__ import annotations

import argparse
import csv
import json
import math
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
resample_to_190hz = None


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
    rows = []
    with path.open("r", newline="") as f:
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


def _eval_segment(seg: np.ndarray, models: dict, beam_width: int) -> dict[str, Any]:
    pipe = run_pipeline_stage23(seg, models, beam_width=beam_width)
    ctc = run_ctc(seg, models)
    return {"pipeline": pipe, "ctc": ctc}


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
        "min_inter_key_ms": None if not intervals_ms else float(np.min(intervals_ms)),
        "max_inter_key_ms": None if not intervals_ms else float(np.max(intervals_ms)),
        "inter_key_ms": intervals_ms,
    }


def _event_window_indices(
    timestamps_ns: np.ndarray,
    events: list[dict[str, Any]],
    target_hz: float = 190.0,
    tail_margin_sec: float = 0.2,
) -> tuple[int | None, int | None]:
    press_ts = [e["timestamp_ns"] for e in events if e["event_type"] == "press" and e["key"] != "enter"]
    if not press_ts:
        return None, None
    start_ns = min(press_ts)
    end_ns = max(press_ts)
    # Keep a small tail after the last real keypress so we preserve release decay,
    # but do not let Enter dominate the evaluated typing window.
    end_ns = int(end_ns + tail_margin_sec * 1_000_000_000.0)
    lo = int(np.searchsorted(timestamps_ns, start_ns, side="left"))
    hi = int(np.searchsorted(timestamps_ns, end_ns, side="right"))
    orig_hz = estimate_sample_rate(np.empty((len(timestamps_ns), 6), dtype=np.float32), timestamps_ns)
    scale = float(target_hz) / float(orig_hz) if orig_hz and orig_hz > 1e-6 else 1.0
    return int(round(lo * scale)), int(round(hi * scale))


def _tight_segment_from_keyness(seg190: np.ndarray, models: dict, beam_width: int, prob_threshold: float, margin_sec: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    probe = run_pipeline_stage23(seg190, models, beam_width=beam_width)
    debug = dict(probe.get("anchor_debug") or {})
    peak_frames = [int(x) for x in debug.get("peak_frames", [])]
    peak_probs = [float(x) for x in debug.get("peak_probs", [])]
    high = [f for f, p in zip(peak_frames, peak_probs) if p >= prob_threshold]
    info = {
        "anchor_debug": debug,
        "high_conf_peak_frames": high,
        "high_conf_threshold": float(prob_threshold),
        "margin_sec": float(margin_sec),
    }
    if not high:
        return None, info
    sr = float(models["stage1_config"].get("sample_rate_hz", 190.0))
    margin_frames = int(round(margin_sec * sr))
    lo = max(0, min(high) - margin_frames)
    hi = min(len(seg190), max(high) + margin_frames)
    if hi - lo < 3:
        return None, info
    info["tight_start_rel"] = int(lo)
    info["tight_end_rel"] = int(hi)
    info["tight_duration_sec"] = float((hi - lo) / sr)
    return seg190[lo:hi], info


def _pack_prediction(block: dict[str, Any], truth: str) -> dict[str, Any]:
    pipe = block.get("pipeline") or {}
    ctc = block.get("ctc") or {}
    return {
        "pipeline_prediction": str(pipe.get("char_top1", "")),
        "pipeline_num_keys": int(pipe.get("num_keys", 0)),
        "pipeline_cer": cer(truth, str(pipe.get("char_top1", ""))),
        "pipeline_top_candidate": "" if not pipe.get("top_candidates") else str(pipe["top_candidates"][0]["password"]),
        "ctc_prediction": str(ctc.get("prediction", "")),
        "ctc_cer": cer(truth, str(ctc.get("prediction", ""))),
        "ctc_top_candidate": "" if not ctc.get("beam_candidates") else str(ctc["beam_candidates"][0]["password"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate still-password-still probe data with auto, event-window, and tight-burst variants.")
    parser.add_argument("--dataset-root", default="data/raw/clean_password_probe")
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--output-dir", default="results/still_password_probe_eval")
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--tight-threshold", type=float, default=0.7)
    parser.add_argument("--tight-margin-sec", type=float, default=0.5)
    parser.add_argument("--coarse-merge-gap-sec", type=float, default=0.0)
    args = parser.parse_args()

    global load_all_models, run_ctc, run_pipeline_stage23, run_stage1
    global csv_to_array, estimate_sample_rate, extract_timestamps, resample_to_190hz
    from inference.pipeline_inference import load_all_models, run_ctc, run_pipeline_stage23, run_stage1
    from inference.preprocess import csv_to_array, estimate_sample_rate, extract_timestamps, resample_to_190hz

    dataset_root = (REPO_ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or str(REPO_ROOT)

    protocol_paths = sorted(dataset_root.glob("*_protocol.json"))
    if not protocol_paths:
        raise SystemExit(f"No protocol files found in {dataset_root}")

    models = load_all_models(checkpoint_dir)
    models.setdefault("pipeline_defaults", {})["stage1_coarse_merge_gap_s"] = float(args.coarse_merge_gap_sec)
    rows: list[dict[str, Any]] = []

    for proto_path in protocol_paths:
        protocol = json.loads(proto_path.read_text(encoding="utf-8"))
        truth = str(protocol["reference_text"])
        sensor_csv = Path(protocol["paths"]["sensor_csv"])
        events_csv = Path(protocol["paths"]["events_csv"])

        csv_string = sensor_csv.read_text(encoding="utf-8")
        timestamps = extract_timestamps(csv_string)
        imu = csv_to_array(csv_string)
        orig_hz = estimate_sample_rate(imu, timestamps)
        imu190 = resample_to_190hz(imu, orig_hz)
        events = _load_events(events_csv)
        timing = _event_timing(events, truth)

        segments = run_stage1(imu190, models)
        stage1_best = _choose_stage1_segment(segments)

        auto_pack: dict[str, Any] | None = None
        tight_pack: dict[str, Any] | None = None
        tight_info: dict[str, Any] | None = None

        if stage1_best is not None:
            lo = max(0, int(stage1_best["start"]))
            hi = min(len(imu190), int(stage1_best["end"]))
            stage1_seg = imu190[lo:hi]
            auto_eval = _eval_segment(stage1_seg, models, beam_width=args.beam_width)
            auto_pack = {
                "segment_start": lo,
                "segment_end": hi,
                "segment_duration_sec": (hi - lo) / 190.0,
                **_pack_prediction(auto_eval, truth),
            }

            tight_seg, tight_info = _tight_segment_from_keyness(
                stage1_seg,
                models,
                beam_width=args.beam_width,
                prob_threshold=args.tight_threshold,
                margin_sec=args.tight_margin_sec,
            )
            if tight_seg is not None:
                tight_eval = _eval_segment(tight_seg, models, beam_width=args.beam_width)
                rel_lo = int(tight_info["tight_start_rel"])
                rel_hi = int(tight_info["tight_end_rel"])
                tight_pack = {
                    "segment_start": lo + rel_lo,
                    "segment_end": lo + rel_hi,
                    "segment_duration_sec": (rel_hi - rel_lo) / 190.0,
                    **_pack_prediction(tight_eval, truth),
                }

        event_lo, event_hi = _event_window_indices(timestamps, events, target_hz=190.0)
        event_pack: dict[str, Any] | None = None
        if event_lo is not None and event_hi is not None and event_hi > event_lo:
            seg = imu190[max(0, event_lo): min(len(imu190), event_hi)]
            event_eval = _eval_segment(seg, models, beam_width=args.beam_width)
            event_pack = {
                "segment_start": int(event_lo),
                "segment_end": int(event_hi),
                "segment_duration_sec": (int(event_hi) - int(event_lo)) / 190.0,
                **_pack_prediction(event_eval, truth),
            }

        row = {
            "session_prefix": protocol["session_prefix"],
            "truth": truth,
            "sample_rate_est": float(orig_hz),
            "typed_matches_reference": bool(protocol.get("typed_matches_reference", False)),
            **timing,
            "stage1_num_segments": len(segments),
            "stage1_segments": segments,
            "stage1_best": stage1_best,
            "auto_fullsample": auto_pack,
            "event_window": event_pack,
            "tight_burst": tight_pack,
            "tight_burst_debug": tight_info,
        }
        rows.append(row)

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
        return None if not vals else sum(vals) / len(vals)

    summary = {
        "dataset_root": str(dataset_root),
        "n_trials": len(rows),
        "mean_inter_key_ms": _mean_from(("mean_inter_key_ms",)),
        "auto_pipeline_mean_cer": _mean_from(("auto_fullsample", "pipeline_cer")),
        "auto_ctc_mean_cer": _mean_from(("auto_fullsample", "ctc_cer")),
        "event_pipeline_mean_cer": _mean_from(("event_window", "pipeline_cer")),
        "event_ctc_mean_cer": _mean_from(("event_window", "ctc_cer")),
        "tight_pipeline_mean_cer": _mean_from(("tight_burst", "pipeline_cer")),
        "tight_ctc_mean_cer": _mean_from(("tight_burst", "ctc_cer")),
        "tight_success_rate": sum(1 for r in rows if r.get("tight_burst") is not None) / len(rows) if rows else None,
        "coarse_merge_gap_sec": float(args.coarse_merge_gap_sec),
    }

    (output_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows = []
    for row in rows:
        base = {
            "session_prefix": row["session_prefix"],
            "truth": row["truth"],
            "mean_inter_key_ms": row["mean_inter_key_ms"],
            "median_inter_key_ms": row["median_inter_key_ms"],
            "stage1_num_segments": row["stage1_num_segments"],
        }
        for prefix in ("auto_fullsample", "event_window", "tight_burst"):
            block = row.get(prefix) or {}
            base[f"{prefix}_pipeline_prediction"] = block.get("pipeline_prediction")
            base[f"{prefix}_pipeline_cer"] = block.get("pipeline_cer")
            base[f"{prefix}_pipeline_num_keys"] = block.get("pipeline_num_keys")
            base[f"{prefix}_ctc_prediction"] = block.get("ctc_prediction")
            base[f"{prefix}_ctc_cer"] = block.get("ctc_cer")
            base[f"{prefix}_segment_duration_sec"] = block.get("segment_duration_sec")
        flat_rows.append(base)

    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output_dir / 'report.json'}")
    print(f"Wrote: {output_dir / 'rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
