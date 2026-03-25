#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader
from onset_detection.stage2_segmental.data import build_password_episodes
from onset_detection.stage2_segmental.metrics import char_topk_from_logits, levenshtein
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from phase3_password_inception.run_password_closure_inception import (
    load_final_inception,
    topk_strings_from_prob_vectors,
)
from train_eval_peak_keyness import (
    _build_dataset,
    _load_mixed_episodes,
    _load_password_attempt_episodes,
    _peak_feature_vector,
    _propose_peaks,
    _select_k_peaks,
)


def resolve_device(name: str) -> torch.device:
    req = (name or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"
    return torch.device(req)


def cer(a: str, b: str) -> float:
    return levenshtein(a, b) / max(len(a), 1)


def _extract_window_from_signal(
    signal: np.ndarray,
    center_frame: int,
    sample_rate_hz: float,
    target_len: int,
    pre_ms: float = 100.0,
    post_ms: float = 200.0,
) -> np.ndarray | None:
    from scipy.signal import resample

    pre_frames = int(round(pre_ms / 1000.0 * sample_rate_hz))
    post_frames = int(round(post_ms / 1000.0 * sample_rate_hz))
    lo = max(0, int(center_frame) - pre_frames)
    hi = min(len(signal), int(center_frame) + post_frames)
    if hi - lo < 3:
        return None
    out = resample(signal[lo:hi], target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def _session_prefix_map(root: str) -> dict[str, str]:
    root_p = Path(root)
    mapping = {}
    for sensor_path in sorted(root_p.glob("*_sensor.csv")):
        session_id = sensor_path.name[: -len("_sensor.csv")]
        mapping[session_id] = str(root_p / session_id)
    return mapping


def _decode_logits(
    logits: np.ndarray,
    classes: np.ndarray,
    ref: str | None = None,
    beam_width: int = 100,
    branch_topk: int = 5,
) -> dict:
    probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1).cpu().numpy()
    pred_idx = probs.argmax(axis=1)
    pred_text = "".join(str(classes[int(i)]) for i in pred_idx.tolist())
    ranks = np.argsort(probs, axis=1)[:, ::-1]
    max_prob = probs.max(axis=1)
    top2 = np.partition(probs, -2, axis=1)[:, -2] if probs.shape[1] > 1 else np.zeros_like(max_prob)
    margin = np.clip(max_prob - top2, 0.0, 1.0)
    seq_candidates = topk_strings_from_prob_vectors(
        [p for p in probs],
        classes,
        branch_topk=branch_topk,
        beam_width=beam_width,
    )
    out = {
        "prediction": pred_text,
        "mean_max_prob": float(np.mean(max_prob)) if len(max_prob) else 0.0,
        "mean_margin": float(np.mean(margin)) if len(margin) else 0.0,
        "avg_log_prob": float(np.mean([c["log_prob"] for c in seq_candidates[:1]])) if seq_candidates else -1e9,
        "top_sequence_candidates": seq_candidates[:10],
        "topk_per_pos": [[str(classes[int(i)]) for i in row[:5]] for row in ranks.tolist()],
    }
    if ref is not None:
        label_map = {str(c): i for i, c in enumerate(classes.tolist())}
        labels = np.asarray([label_map[ch] for ch in ref if ch in label_map], dtype=np.int64)
        out.update(char_topk_from_logits(logits, labels))
        out["reference"] = ref
        out["cer"] = float(cer(ref, pred_text))
        out["exact_match"] = float(ref == pred_text)
        top100 = [x["candidate"] for x in seq_candidates[:100]]
        out["sequence_top100_hit"] = float(ref in top100)
    return out


def _run_stage3_fixed(
    classifier,
    target_len: int,
    classes: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    crop_imu: np.ndarray,
    sample_rate_hz: float,
    local_frames: np.ndarray,
    ref: str | None,
) -> dict | None:
    windows = []
    for frame in local_frames.tolist():
        win = _extract_window_from_signal(crop_imu, int(frame), sample_rate_hz, target_len)
        if win is None:
            return None
        windows.append(win)
    xb = np.stack(windows).astype(np.float32)
    xb = (xb - means[None, None, :]) / (stds[None, None, :] + 1e-6)
    with torch.no_grad():
        logits = classifier(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
    out = _decode_logits(logits, classes, ref=ref)
    out["mode"] = "fixed"
    out["windows_n"] = int(len(windows))
    return out


def _run_stage3_overlap(
    overlap_model,
    classifier,
    classes: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    crop_imu: np.ndarray,
    sample_rate_hz: float,
    local_frames: np.ndarray,
    ref: str | None,
) -> dict | None:
    if len(local_frames) == 0:
        return None
    with torch.no_grad():
        imu = torch.tensor(crop_imu, dtype=torch.float32, device=device)
        key_frames = torch.tensor(local_frames, dtype=torch.long, device=device)
        out = overlap_model.forward_episode(imu, key_frames, sample_rate_hz)
        windows = out["windows"].detach().cpu().numpy()
    xb = (windows - means[None, None, :]) / (stds[None, None, :] + 1e-6)
    with torch.no_grad():
        logits = classifier(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
    dec = _decode_logits(logits, classes, ref=ref)
    dec["mode"] = "overlap"
    dec["windows_n"] = int(len(windows))
    dec["overlap_starts"] = [float(x) for x in out["starts"].detach().cpu().tolist()]
    dec["overlap_ends"] = [float(x) for x in out["ends"].detach().cpu().tolist()]
    return dec


def _candidate_score(result: dict, peak_probs: np.ndarray) -> float:
    if result is None:
        return -1e9
    return (
        float(result["mean_max_prob"])
        + 0.20 * float(result["mean_margin"])
        + (0.05 * float(np.mean(peak_probs)) if len(peak_probs) else 0.0)
        + 0.30 * float(result.get("length_prior_score", 0.0))
    )


def _decode_candidate_segment(
    rf_model,
    overlap_model,
    stage3_model,
    stage3_target_len: int,
    stage3_classes: np.ndarray,
    stage3_means: np.ndarray,
    stage3_stds: np.ndarray,
    device: torch.device,
    crop_imu: np.ndarray,
    crop_ts: np.ndarray,
    sample_rate_hz: float,
    k_values: list[int],
    length_priors: dict[int, dict[str, float]],
    ref: str | None,
) -> dict:
    ep = {
        "imu": crop_imu,
        "timestamps_ns": crop_ts,
        "sample_rate_hz": sample_rate_hz,
    }
    peaks, sm, _ = _propose_peaks(ep)
    if len(peaks) == 0:
        return {
            "num_proposed_peaks": 0,
            "fixed": None,
            "overlap": None,
        }

    X = np.stack([_peak_feature_vector(sm, peaks, i, sample_rate_hz) for i in range(len(peaks))]).astype(np.float32)
    probs = rf_model.predict_proba(X)[:, 1]

    best_fixed = None
    best_overlap = None
    best_fixed_score = -1e9
    best_overlap_score = -1e9
    crop_duration_s = float(len(crop_ts) / max(sample_rate_hz, 1e-6))

    for k in k_values:
        selected = _select_k_peaks(peaks, probs, int(k), sample_rate_hz)
        if len(selected) == 0:
            continue
        peak_mask = np.isin(peaks, selected)
        peak_probs = probs[peak_mask]
        prior = length_priors.get(int(k))
        if prior is not None:
            z = (crop_duration_s - float(prior["mean_s"])) / max(float(prior["std_s"]), 1e-6)
            length_prior_score = float(np.exp(-0.5 * z * z))
        else:
            length_prior_score = 0.0

        fixed = _run_stage3_fixed(
            stage3_model,
            stage3_target_len,
            stage3_classes,
            stage3_means,
            stage3_stds,
            device,
            crop_imu,
            sample_rate_hz,
            selected,
            ref,
        )
        overlap = _run_stage3_overlap(
            overlap_model,
            stage3_model,
            stage3_classes,
            stage3_means,
            stage3_stds,
            device,
            crop_imu,
            sample_rate_hz,
            selected,
            ref,
        )
        if fixed is not None:
            fixed["chosen_k"] = int(k)
            fixed["selected_frames"] = [int(x) for x in selected.tolist()]
            fixed["mean_peak_prob"] = float(np.mean(peak_probs)) if len(peak_probs) else 0.0
            fixed["length_prior_score"] = float(length_prior_score)
            fixed["crop_duration_s"] = float(crop_duration_s)
            s = _candidate_score(fixed, peak_probs)
            if s > best_fixed_score:
                best_fixed_score = s
                best_fixed = fixed
        if overlap is not None:
            overlap["chosen_k"] = int(k)
            overlap["selected_frames"] = [int(x) for x in selected.tolist()]
            overlap["mean_peak_prob"] = float(np.mean(peak_probs)) if len(peak_probs) else 0.0
            overlap["length_prior_score"] = float(length_prior_score)
            overlap["crop_duration_s"] = float(crop_duration_s)
            s = _candidate_score(overlap, peak_probs)
            if s > best_overlap_score:
                best_overlap_score = s
                best_overlap = overlap

    return {
        "num_proposed_peaks": int(len(peaks)),
        "fixed": best_fixed,
        "overlap": best_overlap,
    }


def _aggregate_rows(rows: list[dict]) -> dict:
    if not rows:
        return {
            "num_rows": 0,
            "num_chars": 0,
            "char_top1": 0.0,
            "char_top3": 0.0,
            "char_top5": 0.0,
            "cer": 1.0,
            "exact_match": 0.0,
            "sequence_top100_hit": 0.0,
        }
    num_chars = int(sum(len(r["reference"]) for r in rows))
    return {
        "num_rows": len(rows),
        "num_chars": num_chars,
        "char_top1": float(sum(r["char_top1"] * len(r["reference"]) for r in rows) / max(num_chars, 1)),
        "char_top3": float(sum(r["char_top3"] * len(r["reference"]) for r in rows) / max(num_chars, 1)),
        "char_top5": float(sum(r["char_top5"] * len(r["reference"]) for r in rows) / max(num_chars, 1)),
        "cer": float(sum(levenshtein(r["reference"], r["prediction"]) for r in rows) / max(num_chars, 1)),
        "exact_match": float(np.mean([r["reference"] == r["prediction"] for r in rows])),
        "sequence_top100_hit": float(np.mean([r.get("sequence_top100_hit", 0.0) for r in rows])),
    }


def _pick_session_top_candidates(detail: dict, top_n: int) -> list[dict]:
    preds = detail.get("pred_segments_top5", [])[: max(int(top_n), 0)]
    return sorted(preds, key=lambda x: int(x["start_frame"]))


def _best_assignment_rows(gt_refs: list[str], cand_rows: list[dict]) -> list[dict]:
    n = min(len(gt_refs), len(cand_rows))
    if n == 0:
        return []
    best_perm = None
    best_cost = 10**9
    for perm in itertools.permutations(range(len(cand_rows)), n):
        cost = 0
        for i, idx in enumerate(perm):
            cost += levenshtein(gt_refs[i], cand_rows[idx]["prediction"])
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    rows = []
    assert best_perm is not None
    for i, idx in enumerate(best_perm):
        row = dict(cand_rows[idx])
        row["reference"] = gt_refs[i]
        topk_per_pos = row.get("topk_per_pos", [])
        top1 = top3 = top5 = 0
        for pos, ch in enumerate(gt_refs[i]):
            preds = topk_per_pos[pos] if pos < len(topk_per_pos) else []
            if ch in preds[:1]:
                top1 += 1
            if ch in preds[:3]:
                top3 += 1
            if ch in preds[:5]:
                top5 += 1
        row["char_top1"] = float(top1 / max(len(gt_refs[i]), 1))
        row["char_top3"] = float(top3 / max(len(gt_refs[i]), 1))
        row["char_top5"] = float(top5 / max(len(gt_refs[i]), 1))
        row["cer"] = float(cer(gt_refs[i], row["prediction"]))
        row["exact_match"] = float(gt_refs[i] == row["prediction"])
        top100 = [x["candidate"] for x in row.get("top_sequence_candidates", [])[:100]]
        row["sequence_top100_hit"] = float(gt_refs[i] in top100)
        rows.append(row)
    return rows


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_mixed_root", required=True)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--stage1_details_json", required=True)
    ap.add_argument("--password_dirs", nargs="*", default=[])
    ap.add_argument("--stage3_checkpoint", required=True)
    ap.add_argument("--stage3_scaler", required=True)
    ap.add_argument("--overlap_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--k_values", nargs="*", type=int, default=[8, 9, 10])
    ap.add_argument("--beam_width", type=int, default=100)
    ap.add_argument("--branch_topk", type=int, default=5)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[Info] device={device}")
    print("[Info] building keyness training episodes...")
    train_eps = []
    for d in args.password_dirs:
        train_eps.extend(_load_password_attempt_episodes(d))
    train_eps.extend(_load_mixed_episodes(args.train_mixed_root))
    length_priors = {}
    for k in sorted({len(ep["chars"]) for ep in train_eps}):
        arr = np.asarray(
            [len(ep["timestamps_ns"]) / max(float(ep["sample_rate_hz"]), 1e-6) for ep in train_eps if len(ep["chars"]) == k],
            dtype=np.float64,
        )
        if len(arr) == 0:
            continue
        length_priors[int(k)] = {
            "mean_s": float(np.mean(arr)),
            "std_s": float(max(np.std(arr), 0.8)),
        }
    X, y, meta = _build_dataset(train_eps)
    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
    )
    rf_model.fit(X, y)
    with open(out_dir / "keyness_dataset_summary.json", "w") as f:
        json.dump(
            {
                "num_train_episodes": len(train_eps),
                "num_peak_candidates": int(len(X)),
                "num_positive": int(np.sum(y == 1)),
                "num_negative": int(np.sum(y == 0)),
                "num_sessions": len(sorted({m["session_id"] for m in meta})),
                "length_priors": length_priors,
            },
            f,
            indent=2,
        )
    with open(out_dir / "keyness_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    print("[Info] loading stage3 and overlap checkpoints...")
    stage3_model, stage3_classes, stage3_means, stage3_stds = load_final_inception(
        args.stage3_checkpoint,
        args.stage3_scaler,
        device,
    )
    stage3_target_len = int(torch.load(args.stage3_checkpoint, map_location="cpu", weights_only=False)["n_timesteps"])
    stage3_model.eval()
    overlap_model = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap_model.eval()

    eval_eps = build_password_episodes(args.eval_root)
    episode_map = {ep.episode_id: ep for ep in eval_eps}
    session_map = _session_prefix_map(args.eval_root)

    with open(args.stage1_details_json) as f:
        stage1_details = json.load(f)

    gt_fixed_rows, gt_overlap_rows = [], []
    gt_keyframes_fixed_rows, gt_keyframes_overlap_rows = [], []
    oracle_fixed_rows, oracle_overlap_rows = [], []
    session_fixed_rows, session_overlap_rows = [], []
    all_session_details = []

    for detail in stage1_details:
        session_id = detail["session_id"]
        session_prefix = session_map.get(session_id)
        if session_prefix is None:
            continue
        loader = SessionLoader(session_prefix)
        full_ts, full_imu = loader.get_imu()
        if len(full_ts) == 0:
            continue

        gt_rows_sorted = sorted(detail["gt_rows"], key=lambda x: x["gt_start_frame"])
        per_session = {
            "session_id": session_id,
            "num_gt": len(gt_rows_sorted),
            "num_pred_top5": len(detail.get("pred_segments_top5", [])),
            "gt_segment_results": [],
            "oracle_bestpred_results": [],
            "session_top_results": [],
        }

        # GT segment upper bound for Stage2/3
        for row in gt_rows_sorted:
            ep = episode_map.get(row["episode_id"])
            if ep is None:
                continue
            gt_fixed = _run_stage3_fixed(
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                ep.imu,
                ep.sample_rate_hz,
                ep.key_frames,
                ep.password,
            )
            gt_overlap = _run_stage3_overlap(
                overlap_model,
                stage3_model,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                ep.imu,
                ep.sample_rate_hz,
                ep.key_frames,
                ep.password,
            )
            if gt_fixed is not None:
                r = dict(gt_fixed)
                r["session_id"] = session_id
                r["episode_id"] = ep.episode_id
                gt_keyframes_fixed_rows.append(r)
            if gt_overlap is not None:
                r = dict(gt_overlap)
                r["session_id"] = session_id
                r["episode_id"] = ep.episode_id
                gt_keyframes_overlap_rows.append(r)
            dec = _decode_candidate_segment(
                rf_model,
                overlap_model,
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                ep.imu,
                ep.timestamps_ns,
                ep.sample_rate_hz,
                args.k_values,
                length_priors,
                ep.password,
            )
            if dec["fixed"] is not None:
                r = dict(dec["fixed"])
                r["session_id"] = session_id
                r["episode_id"] = ep.episode_id
                gt_fixed_rows.append(r)
            if dec["overlap"] is not None:
                r = dict(dec["overlap"])
                r["session_id"] = session_id
                r["episode_id"] = ep.episode_id
                gt_overlap_rows.append(r)
            per_session["gt_segment_results"].append({
                "episode_id": ep.episode_id,
                "reference": ep.password,
                "gt_keyframes_fixed": gt_fixed,
                "gt_keyframes_overlap": gt_overlap,
                "fixed": dec["fixed"],
                "overlap": dec["overlap"],
            })

        # Stage1 best_pred linked
        for row in gt_rows_sorted:
            pred = row.get("best_pred")
            if pred is None:
                continue
            start = int(pred["start_frame"])
            end = int(pred["end_frame"])
            crop_imu = full_imu[start : end + 1]
            crop_ts = full_ts[start : end + 1]
            sr = float(1e9 / max(np.median(np.diff(crop_ts)), 1.0)) if len(crop_ts) >= 3 else 200.0
            dec = _decode_candidate_segment(
                rf_model,
                overlap_model,
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                crop_imu,
                crop_ts,
                sr,
                args.k_values,
                length_priors,
                row["password"],
            )
            if dec["fixed"] is not None:
                r = dict(dec["fixed"])
                r["session_id"] = session_id
                r["episode_id"] = row["episode_id"]
                oracle_fixed_rows.append(r)
            if dec["overlap"] is not None:
                r = dict(dec["overlap"])
                r["session_id"] = session_id
                r["episode_id"] = row["episode_id"]
                oracle_overlap_rows.append(r)
            per_session["oracle_bestpred_results"].append({
                "episode_id": row["episode_id"],
                "reference": row["password"],
                "best_iou": row["best_iou"],
                "best_key_recall": row["best_key_recall"],
                "segment": pred,
                "fixed": dec["fixed"],
                "overlap": dec["overlap"],
            })

        # Realistic session top-N candidates
        top_candidates = _pick_session_top_candidates(detail, len(gt_rows_sorted))
        fixed_candidates = []
        overlap_candidates = []
        for cand in top_candidates:
            start = int(cand["start_frame"])
            end = int(cand["end_frame"])
            crop_imu = full_imu[start : end + 1]
            crop_ts = full_ts[start : end + 1]
            sr = float(1e9 / max(np.median(np.diff(crop_ts)), 1.0)) if len(crop_ts) >= 3 else 200.0
            dec = _decode_candidate_segment(
                rf_model,
                overlap_model,
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                crop_imu,
                crop_ts,
                sr,
                args.k_values,
                length_priors,
                ref=None,
            )
            if dec["fixed"] is not None:
                r = dict(dec["fixed"])
                r["segment"] = cand
                fixed_candidates.append(r)
            if dec["overlap"] is not None:
                r = dict(dec["overlap"])
                r["segment"] = cand
                overlap_candidates.append(r)
        gt_refs = [r["password"] for r in gt_rows_sorted]
        sess_fixed = _best_assignment_rows(gt_refs, fixed_candidates)
        sess_overlap = _best_assignment_rows(gt_refs, overlap_candidates)
        for r in sess_fixed:
            r["session_id"] = session_id
        for r in sess_overlap:
            r["session_id"] = session_id
        session_fixed_rows.extend(sess_fixed)
        session_overlap_rows.extend(sess_overlap)
        per_session["session_top_results"] = {
            "fixed": sess_fixed,
            "overlap": sess_overlap,
        }
        all_session_details.append(per_session)

    report = {
        "device": str(device),
        "k_values": args.k_values,
        "gt_keyframes_fixed": _aggregate_rows(gt_keyframes_fixed_rows),
        "gt_keyframes_overlap": _aggregate_rows(gt_keyframes_overlap_rows),
        "gt_segment_fixed": _aggregate_rows(gt_fixed_rows),
        "gt_segment_overlap": _aggregate_rows(gt_overlap_rows),
        "stage1_bestpred_fixed": _aggregate_rows(oracle_fixed_rows),
        "stage1_bestpred_overlap": _aggregate_rows(oracle_overlap_rows),
        "session_top_fixed": _aggregate_rows(session_fixed_rows),
        "session_top_overlap": _aggregate_rows(session_overlap_rows),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(out_dir / "session_details.json", "w") as f:
        json.dump(all_session_details, f, indent=2)
    with open(out_dir / "rows_gt_fixed.json", "w") as f:
        json.dump(gt_fixed_rows, f, indent=2)
    with open(out_dir / "rows_gt_overlap.json", "w") as f:
        json.dump(gt_overlap_rows, f, indent=2)
    with open(out_dir / "rows_gt_keyframes_fixed.json", "w") as f:
        json.dump(gt_keyframes_fixed_rows, f, indent=2)
    with open(out_dir / "rows_gt_keyframes_overlap.json", "w") as f:
        json.dump(gt_keyframes_overlap_rows, f, indent=2)
    with open(out_dir / "rows_stage1_bestpred_fixed.json", "w") as f:
        json.dump(oracle_fixed_rows, f, indent=2)
    with open(out_dir / "rows_stage1_bestpred_overlap.json", "w") as f:
        json.dump(oracle_overlap_rows, f, indent=2)
    with open(out_dir / "rows_session_top_fixed.json", "w") as f:
        json.dump(session_fixed_rows, f, indent=2)
    with open(out_dir / "rows_session_top_overlap.json", "w") as f:
        json.dump(session_overlap_rows, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
