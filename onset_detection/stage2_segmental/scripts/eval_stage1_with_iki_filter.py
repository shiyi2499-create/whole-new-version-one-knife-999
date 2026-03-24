#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from glob import glob
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader
from onset_detection.stage2_segmental.scripts.train_eval_stage1_dense_labeling import (
    UNet1D,
    _build_session_records,
    compute_iou,
    compute_key_recall,
    extract_segments,
    resolve_device,
)


@dataclass
class IKIPrior:
    median_s: float
    mean_s: float
    std_s: float
    p10_s: float
    p90_s: float
    n: int


@dataclass
class IKISegment:
    start_frame: int
    end_frame: int
    dense_confidence: float
    iki_score: float
    combined_score: float
    n_keystrokes: int
    iki_median_s: float
    iki_std_s: float
    key_frames_rel: list[int]


def _read_events(path: str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out.append(
                {
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "event_type": row["event_type"],
                }
            )
    return out


def _read_attempts(path: str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out.append(
                {
                    "attempt_start_ns": int(row["attempt_start_ns"]),
                    "submit_ns": int(row["submit_ns"]),
                    "match": row["match"],
                }
            )
    return out


def build_iki_prior_from_password_dirs(password_dirs: list[str]) -> IKIPrior:
    all_ikis: list[float] = []
    for pw_dir in password_dirs:
        if not os.path.isdir(pw_dir):
            continue
        for attempts_path in sorted(glob(os.path.join(pw_dir, "*_attempts.csv"))):
            events_path = attempts_path.replace("_attempts.csv", "_events.csv")
            if not os.path.exists(events_path):
                continue
            presses = [
                row["timestamp_ns"]
                for row in _read_events(events_path)
                if row["event_type"] == "press"
            ]
            for att in _read_attempts(attempts_path):
                if att["match"] != "YES":
                    continue
                seg_presses = [t for t in presses if att["attempt_start_ns"] <= t <= att["submit_ns"]]
                if len(seg_presses) >= 2:
                    all_ikis.extend((np.diff(seg_presses) / 1e9).tolist())

    if not all_ikis:
        return IKIPrior(1.5, 1.5, 0.5, 0.8, 2.5, 0)
    arr = np.asarray(all_ikis, dtype=np.float64)
    return IKIPrior(
        median_s=float(np.median(arr)),
        mean_s=float(np.mean(arr)),
        std_s=float(np.std(arr)),
        p10_s=float(np.percentile(arr, 10)),
        p90_s=float(np.percentile(arr, 90)),
        n=int(len(arr)),
    )


def detect_keystrokes_in_segment(imu_segment: np.ndarray, sr: float) -> np.ndarray:
    if len(imu_segment) < 10:
        return np.array([], dtype=np.int64)
    win = max(3, int(round(sr * 0.04)))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    activity = np.zeros(len(imu_segment), dtype=np.float64)
    for ch in range(min(6, imu_segment.shape[1])):
        col = imu_segment[:, ch].astype(np.float64)
        mu = np.convolve(col, kernel, mode="same")
        mu2 = np.convolve(col ** 2, kernel, mode="same")
        activity += np.maximum(mu2 - mu ** 2, 0.0)
    gyro_mag = np.sqrt(np.sum(imu_segment[:, 3:6].astype(np.float64) ** 2, axis=1))
    combined = np.log1p(activity) + 0.5 * gyro_mag
    if len(combined) < 20:
        return np.array([], dtype=np.int64)
    std = float(np.std(combined))
    mean = float(np.mean(combined))
    peaks, props = find_peaks(
        combined,
        distance=max(3, int(round(sr * 0.08))),
        prominence=max(0.05, std * 0.3),
        height=mean + std * 0.3,
    )
    return peaks.astype(np.int64)


def collapse_nearby_peaks(
    peak_frames: np.ndarray,
    peak_scores: np.ndarray,
    sr: float,
    merge_s: float,
) -> np.ndarray:
    peak_frames = np.asarray(peak_frames, dtype=np.int64)
    peak_scores = np.asarray(peak_scores, dtype=np.float64)
    if len(peak_frames) <= 1 or merge_s <= 0:
        return peak_frames
    order = np.argsort(peak_frames)
    peak_frames = peak_frames[order]
    peak_scores = peak_scores[order]
    merge_frames = max(1, int(round(float(merge_s) * float(sr))))
    kept = []
    cur_frames = [int(peak_frames[0])]
    cur_scores = [float(peak_scores[0])]
    for fr, sc in zip(peak_frames[1:], peak_scores[1:]):
        if int(fr) - cur_frames[-1] <= merge_frames:
            cur_frames.append(int(fr))
            cur_scores.append(float(sc))
        else:
            best_idx = int(np.argmax(cur_scores))
            kept.append(cur_frames[best_idx])
            cur_frames = [int(fr)]
            cur_scores = [float(sc)]
    best_idx = int(np.argmax(cur_scores))
    kept.append(cur_frames[best_idx])
    return np.asarray(kept, dtype=np.int64)


def compute_segment_iki_score(
    keystroke_frames: np.ndarray,
    sr: float,
    iki_prior: IKIPrior,
    count_weight: float,
) -> tuple[float, dict]:
    n = int(len(keystroke_frames))
    details = {
        "n_keystrokes": n,
        "iki_median_s": 0.0,
        "iki_std_s": 0.0,
    }
    if n < 2:
        return -5.0, details
    ikis_s = np.diff(keystroke_frames.astype(np.float64)) / float(sr)
    iki_med = float(np.median(ikis_s))
    iki_std = float(np.std(ikis_s))
    details["iki_median_s"] = iki_med
    details["iki_std_s"] = iki_std

    score = 0.0

    if count_weight > 0.0:
        if 7 <= n <= 13:
            score += 3.0 * count_weight
        elif 5 <= n <= 15:
            score += 1.0 * count_weight
        elif n < 4:
            score -= 2.0 * count_weight
        elif n > 25:
            score -= 3.0 * count_weight

    if iki_med < 0.25:
        score -= 5.0
    elif iki_med < 0.5:
        score -= 2.0
    elif 0.5 <= iki_med <= 4.0:
        dist = abs(iki_med - iki_prior.median_s)
        score += max(0.0, 3.0 - dist / max(0.1, iki_prior.std_s))
    else:
        score -= 1.0

    if iki_med > 0.3 and iki_std > 0:
        cv = iki_std / iki_med
        if cv < 0.4:
            score += 1.0
        elif cv < 0.7:
            score += 0.5
        elif cv > 1.5:
            score -= 1.0

    return float(score), details


def iki_filter_segments(
    candidate_segments: list[tuple[int, int, float]],
    imu: np.ndarray,
    sr: float,
    iki_prior: IKIPrior,
    dense_weight: float,
    iki_weight: float,
    count_weight: float,
    iki_score_threshold: float,
    key_merge_s: float,
) -> list[IKISegment]:
    results: list[IKISegment] = []
    for start, end, conf in candidate_segments:
        seg_imu = imu[start:end]
        key_frames = detect_keystrokes_in_segment(seg_imu, sr)
        if len(key_frames):
            # Re-score the detected peaks with the same local signal and collapse
            # overly-dense duplicates before computing IKI.
            win = max(3, int(round(sr * 0.04)))
            kernel = np.ones(win, dtype=np.float64) / float(win)
            activity = np.zeros(len(seg_imu), dtype=np.float64)
            for ch in range(min(6, seg_imu.shape[1])):
                col = seg_imu[:, ch].astype(np.float64)
                mu = np.convolve(col, kernel, mode="same")
                mu2 = np.convolve(col ** 2, kernel, mode="same")
                activity += np.maximum(mu2 - mu ** 2, 0.0)
            gyro_mag = np.sqrt(np.sum(seg_imu[:, 3:6].astype(np.float64) ** 2, axis=1))
            combined = np.log1p(activity) + 0.5 * gyro_mag
            key_scores = combined[np.clip(key_frames, 0, len(combined) - 1)]
            key_frames = collapse_nearby_peaks(key_frames, key_scores, sr, merge_s=key_merge_s)
        iki_score, details = compute_segment_iki_score(key_frames, sr, iki_prior, count_weight=count_weight)
        combined = float(dense_weight * conf + iki_weight * iki_score)
        results.append(
            IKISegment(
                start_frame=int(start),
                end_frame=int(end),
                dense_confidence=float(conf),
                iki_score=float(iki_score),
                combined_score=combined,
                n_keystrokes=int(details["n_keystrokes"]),
                iki_median_s=float(details["iki_median_s"]),
                iki_std_s=float(details["iki_std_s"]),
                key_frames_rel=key_frames.astype(np.int64).tolist(),
            )
        )
    results.sort(key=lambda x: -x.combined_score)
    return [x for x in results if x.iki_score >= float(iki_score_threshold)]


def _load_dense_model(
    ckpt_path: str,
    in_channels: int,
    device: torch.device,
    base_filters: int,
    depth: int,
    kernel_size: int,
    dropout: float,
    use_attention: bool,
) -> UNet1D:
    model = UNet1D(
        in_channels=in_channels,
        base_filters=base_filters,
        depth=depth,
        kernel_size=kernel_size,
        dropout=dropout,
        use_attention=use_attention,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _best_match_for_gt(pred_segments: list[tuple[int, int, float]], gt) -> tuple[float, float, tuple[int, int, float] | None]:
    best_iou = 0.0
    best_key_recall = 0.0
    best_pred = None
    for pred in pred_segments:
        iou = compute_iou(pred[0], pred[1], gt.start_frame, gt.end_frame)
        if iou > best_iou:
            best_iou = iou
            best_key_recall = compute_key_recall(pred[0], pred[1], gt.key_frames)
            best_pred = pred
    return best_iou, best_key_recall, best_pred


def _summarize_session_rows(rows: list[dict]) -> dict:
    all_gt_best_ious = []
    all_gt_best_key_recalls = []
    all_gt_complete = []
    single_top1_ious = []
    single_top1_key_recall = []
    single_top1_complete = []
    pred_counts = []
    for row in rows:
        pred_counts.append(int(row["num_pred_segments"]))
        if int(row["num_gt_segments"]) == 1:
            gt = row["gt_rows"][0]
            single_top1_ious.append(float(gt["best_iou"]))
            single_top1_key_recall.append(float(gt["best_key_recall"]))
            single_top1_complete.append(float(gt["best_key_recall"] >= 0.999))
        for gt in row["gt_rows"]:
            all_gt_best_ious.append(float(gt["best_iou"]))
            all_gt_best_key_recalls.append(float(gt["best_key_recall"]))
            all_gt_complete.append(float(gt["best_key_recall"] >= 0.999))
    return {
        "n_sessions": int(len(rows)),
        "mean_pred_segments": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "all_gt_oracle": {
            "mean_best_iou": float(np.mean(all_gt_best_ious)) if all_gt_best_ious else 0.0,
            "iou_ge_0.7": float(np.mean([x >= 0.7 for x in all_gt_best_ious])) if all_gt_best_ious else 0.0,
            "iou_ge_0.5": float(np.mean([x >= 0.5 for x in all_gt_best_ious])) if all_gt_best_ious else 0.0,
            "mean_best_key_recall": float(np.mean(all_gt_best_key_recalls)) if all_gt_best_key_recalls else 0.0,
            "complete_hit_rate": float(np.mean(all_gt_complete)) if all_gt_complete else 0.0,
        },
        "single_session_top1": {
            "n_sessions": int(len(single_top1_ious)),
            "mean_iou": float(np.mean(single_top1_ious)) if single_top1_ious else 0.0,
            "iou_ge_0.7": float(np.mean([x >= 0.7 for x in single_top1_ious])) if single_top1_ious else 0.0,
            "iou_ge_0.5": float(np.mean([x >= 0.5 for x in single_top1_ious])) if single_top1_ious else 0.0,
            "mean_key_recall": float(np.mean(single_top1_key_recall)) if single_top1_key_recall else 0.0,
            "complete_hit_rate": float(np.mean(single_top1_complete)) if single_top1_complete else 0.0,
        },
    }


def evaluate_dense_plus_iki(
    model: UNet1D,
    records,
    device: torch.device,
    threshold: float,
    min_segment_frames: int,
    merge_gap_frames: int,
    iki_prior: IKIPrior,
    dense_weight: float,
    iki_weight: float,
    count_weight: float,
    iki_score_threshold: float,
    key_merge_s: float,
    top_k_keep: int,
) -> tuple[dict, list[dict], dict, list[dict]]:
    model.eval()
    baseline_rows = []
    filtered_rows = []
    with torch.no_grad():
        for rec in records:
            x = torch.from_numpy(rec.features).float().unsqueeze(0).to(device)
            probs = torch.sigmoid(model(x)).squeeze().detach().cpu().numpy()
            baseline = extract_segments(
                probs,
                threshold=threshold,
                min_length=min_segment_frames,
                merge_gap=merge_gap_frames,
            )
            loader = SessionLoader(rec.session_path)
            _, imu = loader.get_imu()
            filtered = iki_filter_segments(
                candidate_segments=baseline,
                imu=imu,
                sr=rec.sample_rate_hz,
                iki_prior=iki_prior,
                dense_weight=dense_weight,
                iki_weight=iki_weight,
                count_weight=count_weight,
                iki_score_threshold=iki_score_threshold,
                key_merge_s=key_merge_s,
            )
            if top_k_keep > 0:
                filtered = filtered[:top_k_keep]
            filtered_segments = [(s.start_frame, s.end_frame, s.combined_score) for s in filtered]

            row_base = {
                "session_id": rec.session_id,
                "source": Path(rec.session_path).parent.name,
                "num_gt_segments": len(rec.gt_segments),
                "num_pred_segments": len(baseline),
                "pred_segments_top5": [
                    {"start_frame": int(p[0]), "end_frame": int(p[1]), "confidence": float(p[2])}
                    for p in baseline[:5]
                ],
                "gt_rows": [],
            }
            row_filt = {
                "session_id": rec.session_id,
                "source": Path(rec.session_path).parent.name,
                "num_gt_segments": len(rec.gt_segments),
                "num_pred_segments": len(filtered_segments),
                "pred_segments_top5": [
                    {
                        "start_frame": int(s.start_frame),
                        "end_frame": int(s.end_frame),
                        "dense_confidence": float(s.dense_confidence),
                        "iki_score": float(s.iki_score),
                        "combined_score": float(s.combined_score),
                        "n_keystrokes": int(s.n_keystrokes),
                        "iki_median_s": float(s.iki_median_s),
                        "iki_std_s": float(s.iki_std_s),
                    }
                    for s in filtered[:5]
                ],
                "gt_rows": [],
            }
            for gt in rec.gt_segments:
                biou, bkr, bpred = _best_match_for_gt(baseline, gt)
                fiou, fkr, fpred = _best_match_for_gt(filtered_segments, gt)
                row_base["gt_rows"].append(
                    {
                        "episode_id": gt.episode_id,
                        "password": gt.password,
                        "best_iou": float(biou),
                        "best_key_recall": float(bkr),
                        "best_pred": None if bpred is None else {
                            "start_frame": int(bpred[0]),
                            "end_frame": int(bpred[1]),
                            "confidence": float(bpred[2]),
                        },
                    }
                )
                row_filt["gt_rows"].append(
                    {
                        "episode_id": gt.episode_id,
                        "password": gt.password,
                        "best_iou": float(fiou),
                        "best_key_recall": float(fkr),
                        "best_pred": None if fpred is None else {
                            "start_frame": int(fpred[0]),
                            "end_frame": int(fpred[1]),
                            "confidence": float(fpred[2]),
                        },
                    }
                )
            baseline_rows.append(row_base)
            filtered_rows.append(row_filt)

    return (
        _summarize_session_rows(baseline_rows),
        baseline_rows,
        _summarize_session_rows(filtered_rows),
        filtered_rows,
    )


def _subset_rows(rows: list[dict], sources: set[str]) -> list[dict]:
    return [r for r in rows if r["source"] in sources]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_ckpt", required=True)
    ap.add_argument("--eval_dirs", nargs="+", required=True)
    ap.add_argument("--password_dirs", nargs="+", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--feature_mode", default="raw6_energy_activity_pulse")
    ap.add_argument("--label_pre_pad_ms", type=float, default=220.0)
    ap.add_argument("--label_post_pad_ms", type=float, default=380.0)
    ap.add_argument("--min_password_len", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min_segment_s", type=float, default=0.5)
    ap.add_argument("--merge_gap_s", type=float, default=1.0)
    ap.add_argument("--dense_weight", type=float, default=1.0)
    ap.add_argument("--iki_weight", type=float, default=1.0)
    ap.add_argument("--count_weight", type=float, default=0.0)
    ap.add_argument("--iki_score_threshold", type=float, default=0.0)
    ap.add_argument("--key_merge_s", type=float, default=0.45)
    ap.add_argument("--top_k_keep", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--base_filters", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--kernel_size", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_attention", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    records = _build_session_records(
        roots=args.eval_dirs,
        pre_pad_ms=args.label_pre_pad_ms,
        post_pad_ms=args.label_post_pad_ms,
        feature_mode=args.feature_mode,
        min_password_len=args.min_password_len,
    )
    if not records:
        raise RuntimeError("No eval records loaded.")
    in_channels = int(records[0].features.shape[0])
    model = _load_dense_model(
        ckpt_path=args.dense_ckpt,
        in_channels=in_channels,
        device=device,
        base_filters=args.base_filters,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_attention=bool(args.use_attention),
    )
    iki_prior = build_iki_prior_from_password_dirs(args.password_dirs)
    median_sr = float(np.median([r.sample_rate_hz for r in records]))
    min_segment_frames = int(round(args.min_segment_s * median_sr))
    merge_gap_frames = int(round(args.merge_gap_s * median_sr))

    baseline_report, baseline_details, filtered_report, filtered_details = evaluate_dense_plus_iki(
        model=model,
        records=records,
        device=device,
        threshold=args.threshold,
        min_segment_frames=min_segment_frames,
        merge_gap_frames=merge_gap_frames,
        iki_prior=iki_prior,
        dense_weight=args.dense_weight,
                iki_weight=args.iki_weight,
                count_weight=args.count_weight,
                iki_score_threshold=args.iki_score_threshold,
                key_merge_s=args.key_merge_s,
                top_k_keep=args.top_k_keep,
            )

    subsets = {
        "single": {"mixed_single_training", "mixed_single_len9"},
        "retry": {"mixed_retry_training"},
        "retry_len9": {"mixed_retry_len9"},
    }
    summary = {
        "config": vars(args),
        "iki_prior": asdict(iki_prior),
        "baseline": baseline_report,
        "filtered": filtered_report,
        "by_subset": {},
    }
    for name, source_set in subsets.items():
        b_rows = _subset_rows(baseline_details, source_set)
        f_rows = _subset_rows(filtered_details, source_set)
        if b_rows:
            summary["by_subset"][name] = {
                "baseline": _summarize_session_rows(b_rows),
                "filtered": _summarize_session_rows(f_rows),
            }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "baseline_details.json", "w", encoding="utf-8") as f:
        json.dump(baseline_details, f, ensure_ascii=False, indent=2)
    with open(out_dir / "filtered_details.json", "w", encoding="utf-8") as f:
        json.dump(filtered_details, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
