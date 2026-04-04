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


def _align_presses_to_reference(events: list[dict[str, Any]], reference: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    presses = [e for e in events if e["event_type"] == "press" and e["key"] not in {"enter", "return", "backspace"}]
    obs = "".join(e["key"] if len(e["key"]) == 1 else " " for e in presses)
    matched = []
    j = 0
    skipped = []
    for idx, e in enumerate(presses):
        if j < len(reference) and e["key"] == reference[j]:
            matched.append((idx, e))
            j += 1
            if j == len(reference):
                skipped.extend(list(range(idx + 1, len(presses))))
                break
        else:
            skipped.append(idx)
    debug = {
        "observed_press_text": obs,
        "reference_text": reference,
        "observed_press_count": len(presses),
        "matched_count": len(matched),
        "matched_press_indices": [i for i, _ in matched],
        "skipped_press_indices": skipped,
        "alignment_ok": j == len(reference),
    }
    return [e for _, e in matched], debug


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


def _find_bounds(timestamps: np.ndarray, matched_presses: list[dict[str, Any]], sample_rate_hz: float, pre_margin_sec: float, post_margin_sec: float) -> tuple[int, int]:
    if not matched_presses:
        return 0, len(timestamps)
    first_ts = int(matched_presses[0]["timestamp_ns"])
    last_ts = int(matched_presses[-1]["timestamp_ns"])
    first_idx = int(np.searchsorted(timestamps, first_ts, side="left"))
    last_idx = int(np.searchsorted(timestamps, last_ts, side="left"))
    pre = int(round(pre_margin_sec * sample_rate_hz))
    post = int(round(post_margin_sec * sample_rate_hz))
    lo = max(0, first_idx - pre)
    hi = min(len(timestamps), last_idx + post)
    return lo, hi


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
        "ctc_top_candidate": "" if not ctc.get("beam_candidates") else str(ctc["beam_candidates"][0]["password"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate workstation 800Hz still-password-still with GT-assisted segment + Stage2/Stage3.")
    parser.add_argument("--dataset-root", default="data/raw/800hz/clean_password_probe")
    parser.add_argument("--checkpoint-dir", default="demo_inference_api/inference/checkpoints_800hz_fullauto_stage12")
    parser.add_argument("--output-dir", default="results/800hz_stage12_baseline/still_probe_stage2_gtassist_eval_20260401")
    parser.add_argument("--beam-width", type=int, default=500)
    parser.add_argument("--pre-margin-sec", type=float, default=0.20)
    # Stage2 peak proposal needs substantial post-typing context on these
    # workstation still-password-still captures; short crops systematically
    # undercount trailing keys.
    parser.add_argument("--post-margin-sec", type=float, default=4.0)
    args = parser.parse_args()

    global load_all_models, run_ctc, run_pipeline_stage23
    global csv_to_array, estimate_sample_rate, extract_timestamps
    from inference.pipeline_inference_800hz_demo import load_all_models, run_ctc, run_pipeline_stage23
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
        matched_presses, align_debug = _align_presses_to_reference(events, truth)
        lo, hi = _find_bounds(
            timestamps,
            matched_presses,
            orig_hz,
            pre_margin_sec=args.pre_margin_sec,
            post_margin_sec=args.post_margin_sec,
        )
        seg = imu[lo:hi]

        pipe = run_pipeline_stage23(seg, models, beam_width=args.beam_width, source_hz=orig_hz)
        ctc = run_ctc(seg, models, source_hz=orig_hz)

        rows.append(
            {
                "session_prefix": protocol["session_prefix"],
                "truth": truth,
                "sample_rate_est": orig_hz,
                "typed_matches_reference": bool(protocol.get("typed_matches_reference", False)),
                **timing,
                "alignment_debug": align_debug,
                "gtassist_segment": {
                    "start": int(lo),
                    "end": int(hi),
                    "duration_sec": float((hi - lo) / max(orig_hz, 1e-6)),
                    "pre_margin_sec": float(args.pre_margin_sec),
                    "post_margin_sec": float(args.post_margin_sec),
                },
                "stage2_stage3": {
                    **_pack_prediction(pipe, ctc, truth),
                    "selected_frames": pipe.get("selected_frames"),
                    "selected_frames_190hz": pipe.get("selected_frames_190hz"),
                    "selected_frames_800hz": pipe.get("selected_frames_800hz"),
                    "mode_used": pipe.get("mode_used"),
                    "anchor_debug": pipe.get("anchor_debug"),
                    "selection_debug": pipe.get("selection_debug"),
                },
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
        "n_trials": len(rows),
        "mean_inter_key_ms": _mean_from(("mean_inter_key_ms",)),
        "pipeline_mean_cer": _mean_from(("stage2_stage3", "pipeline_cer")),
        "ctc_mean_cer": _mean_from(("stage2_stage3", "ctc_cer")),
        "pipeline_exact": int(sum((r["stage2_stage3"]["pipeline_prediction"] == r["truth"]) for r in rows)),
        "ctc_exact": int(sum((r["stage2_stage3"]["ctc_prediction"] == r["truth"]) for r in rows)),
        "pipeline_top100_hit": int(sum(bool(r["stage2_stage3"]["pipeline_top100_hit"]) for r in rows)),
        "mean_num_keys": _mean_from(("stage2_stage3", "pipeline_num_keys")),
        "mean_truth_len": float(np.mean([len(r["truth"]) for r in rows])) if rows else None,
    }

    (output_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows = []
    for row in rows:
        block = row["stage2_stage3"]
        flat_rows.append(
            {
                "session_prefix": row["session_prefix"],
                "truth": row["truth"],
                "truth_len": len(row["truth"]),
                "sample_rate_est": row["sample_rate_est"],
                "pipeline_prediction": block["pipeline_prediction"],
                "pipeline_num_keys": block["pipeline_num_keys"],
                "pipeline_cer": block["pipeline_cer"],
                "pipeline_top100_hit": block["pipeline_top100_hit"],
                "ctc_prediction": block["ctc_prediction"],
                "ctc_cer": block["ctc_cer"],
                "segment_duration_sec": row["gtassist_segment"]["duration_sec"],
            }
        )

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
