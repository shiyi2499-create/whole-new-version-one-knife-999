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
if str(REPO_ROOT / "onset_detection") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "onset_detection"))

from inference.pipeline_inference import load_all_models, run_stage1, run_pipeline_stage23
from inference.preprocess import csv_to_array, estimate_sample_rate, extract_timestamps
from onset_detection.stage2_segmental.scripts.eval_stage123_end_to_end_strongstage2 import _run_stage3_fixed


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
    with path.open("r", newline="", encoding="utf-8") as f:
        return [
            {
                "timestamp_ns": int(r["timestamp_ns"]),
                "key": r["key"],
                "event_type": r["event_type"],
            }
            for r in csv.DictReader(f)
        ]


def _align_presses_to_reference(events: list[dict[str, Any]], reference: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    presses = [e for e in events if e["event_type"] == "press" and e["key"] not in {"enter", "return", "backspace"}]
    matched: list[dict[str, Any]] = []
    j = 0
    skipped: list[int] = []
    for idx, e in enumerate(presses):
        if j < len(reference) and e["key"] == reference[j]:
            matched.append(e)
            j += 1
            if j == len(reference):
                skipped.extend(list(range(idx + 1, len(presses))))
                break
        else:
            skipped.append(idx)
    debug = {
        "observed_press_count": len(presses),
        "matched_count": len(matched),
        "alignment_ok": j == len(reference),
        "skipped_press_indices": skipped,
    }
    return matched, debug


def _choose_stage1_segment(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not segments:
        return None
    return max(segments, key=lambda s: (float(s.get("confidence", 0.0)), int(s["end"]) - int(s["start"])))


def _result_block_from_pipeline(pipe: dict[str, Any], truth: str) -> dict[str, Any]:
    pred = str(pipe.get("char_top1", ""))
    top = pipe.get("top_candidates", []) or []
    truth_rank = None
    for i, cand in enumerate(top, 1):
        if str(cand.get("password", "")) == truth:
            truth_rank = i
            break
    return {
        "prediction": pred,
        "cer": cer(truth, pred),
        "num_keys": int(pipe.get("num_keys", 0)),
        "truth_in_top100": truth_rank is not None and truth_rank <= 100,
        "truth_rank": truth_rank,
        "top_candidates": top[:10],
    }


def _result_block_stage3(res: dict[str, Any] | None, truth: str, gt_key_count: int) -> dict[str, Any] | None:
    if res is None:
        return None
    pred = str(res.get("prediction", ""))
    cands = res.get("top_sequence_candidates", []) or []
    truth_rank = None
    for i, c in enumerate(cands, 1):
        if str(c.get("candidate", "")) == truth:
            truth_rank = i
            break
    return {
        "prediction": pred,
        "cer": cer(truth, pred),
        "num_keys": int(gt_key_count),
        "truth_in_top100": truth_rank is not None and truth_rank <= 100,
        "truth_rank": truth_rank,
        "top_candidates": [
            {"password": str(c["candidate"]), "score": float(c["log_prob"])}
            for c in cands[:10]
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Same Stage1 segment: current Stage2 vs GT-stage2, same Stage3.")
    ap.add_argument("--dataset-root", default="data/raw/800hz/clean_password_probe")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--beam-width", type=int, default=500)
    args = ap.parse_args()

    dataset_root = (REPO_ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    checkpoint_dir = (REPO_ROOT / args.checkpoint_dir).resolve() if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = load_all_models(str(checkpoint_dir))
    protocol_paths = sorted(dataset_root.glob("*_protocol.json"))
    if not protocol_paths:
        raise SystemExit(f"No protocol files found in {dataset_root}")

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
        matched_presses, align_debug = _align_presses_to_reference(events, truth)

        segments = run_stage1(imu, models)
        best = _choose_stage1_segment(segments)
        if best is None:
            rows.append(
                {
                    "session_prefix": protocol["session_prefix"],
                    "truth": truth,
                    "sample_rate_est": orig_hz,
                    "stage1_num_segments": 0,
                    "stage1_best": None,
                    "alignment_debug": align_debug,
                    "current_stage2_same_stage1": None,
                    "gt_stage2_same_stage1": None,
                }
            )
            continue

        lo = max(0, int(best["start"]))
        hi = min(len(imu), int(best["end"]))
        seg = imu[lo:hi]

        cur_pipe = run_pipeline_stage23(seg, models, beam_width=args.beam_width)

        gt_global_frames = np.searchsorted(
            timestamps,
            np.asarray([int(e["timestamp_ns"]) for e in matched_presses], dtype=np.int64),
            side="left",
        ).astype(np.int64)
        gt_local_frames = gt_global_frames - lo
        inside = (gt_local_frames >= 0) & (gt_local_frames < (hi - lo))
        gt_local_frames = gt_local_frames[inside]

        gt_fixed = _run_stage3_fixed(
            models["stage3_model"],
            models["stage3_target_len"],
            models["stage3_classes"],
            models["stage3_means"],
            models["stage3_stds"],
            models["device"],
            seg,
            orig_hz,
            gt_local_frames,
            ref=truth,
            beam_width=args.beam_width,
            branch_topk=int(models.get("pipeline_defaults", {}).get("branch_topk", 5)),
            sequence_hit_cutoff=int(models.get("pipeline_defaults", {}).get("sequence_hit_cutoff", max(100, args.beam_width))),
            pre_ms=float(models.get("stage3_pre_ms", 100.0)),
            post_ms=float(models.get("stage3_post_ms", 200.0)),
            norm_mode=str(models.get("stage3_norm_mode", "global")),
            use_diff_channels=bool(models.get("stage3_use_diff_channels", False)),
        )

        rows.append(
            {
                "session_prefix": protocol["session_prefix"],
                "truth": truth,
                "sample_rate_est": orig_hz,
                "stage1_num_segments": len(segments),
                "stage1_best": {
                    "start": lo,
                    "end": hi,
                    "duration_sec": float((hi - lo) / max(orig_hz, 1e-6)),
                    "confidence": float(best.get("confidence", 0.0)),
                },
                "alignment_debug": align_debug,
                "gt_keyframes_inside_stage1": int(len(gt_local_frames)),
                "truth_len": len(truth),
                "current_stage2_same_stage1": _result_block_from_pipeline(cur_pipe, truth),
                "gt_stage2_same_stage1": _result_block_stage3(gt_fixed, truth, int(len(gt_local_frames))),
            }
        )

    def _mean(path: tuple[str, str]) -> float | None:
        vals = []
        for row in rows:
            blk = row.get(path[0]) or {}
            v = blk.get(path[1])
            if v is not None:
                vals.append(float(v))
        return None if not vals else float(np.mean(vals))

    summary = {
        "dataset_root": str(dataset_root),
        "checkpoint_dir": str(checkpoint_dir),
        "n_trials": len(rows),
        "same_stage1_current_stage2_mean_cer": _mean(("current_stage2_same_stage1", "cer")),
        "same_stage1_gt_stage2_mean_cer": _mean(("gt_stage2_same_stage1", "cer")),
        "same_stage1_current_stage2_exact": int(sum((r.get("current_stage2_same_stage1") or {}).get("prediction", "") == r["truth"] for r in rows)),
        "same_stage1_gt_stage2_exact": int(sum((r.get("gt_stage2_same_stage1") or {}).get("prediction", "") == r["truth"] for r in rows)),
        "same_stage1_current_stage2_top100": int(sum(bool((r.get("current_stage2_same_stage1") or {}).get("truth_in_top100")) for r in rows)),
        "same_stage1_gt_stage2_top100": int(sum(bool((r.get("gt_stage2_same_stage1") or {}).get("truth_in_top100")) for r in rows)),
        "mean_stage1_duration_sec": float(np.mean([r["stage1_best"]["duration_sec"] for r in rows if r.get("stage1_best")])),
    }

    (output_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_rows = []
    for row in rows:
        cur = row.get("current_stage2_same_stage1") or {}
        gt = row.get("gt_stage2_same_stage1") or {}
        s1 = row.get("stage1_best") or {}
        flat_rows.append(
            {
                "session_prefix": row["session_prefix"],
                "truth": row["truth"],
                "stage1_duration_sec": s1.get("duration_sec"),
                "gt_keyframes_inside_stage1": row.get("gt_keyframes_inside_stage1"),
                "cur_pred": cur.get("prediction"),
                "cur_num_keys": cur.get("num_keys"),
                "cur_cer": cur.get("cer"),
                "gt2_pred": gt.get("prediction"),
                "gt2_num_keys": gt.get("num_keys"),
                "gt2_cer": gt.get("cer"),
            }
        )
    with (output_dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote: {output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
