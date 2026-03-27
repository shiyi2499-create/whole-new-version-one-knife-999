#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader
from onset_detection.stage2_segmental.data import build_password_episodes
from onset_detection.stage2_segmental.length_model import (
    build_length_subregion_from_energy,
    compute_region_length_features,
    load_length_model,
)
from onset_detection.stage2_segmental.metrics import char_topk_from_logits, levenshtein
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.scripts.eval_overlap_single_coarse_energy_cls import (
    propose_energy_classifier_anchors,
    score_candidate_region_from_frames,
)
from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import (
    _peak_feature_vector,
    _propose_peaks,
    _select_k_peaks,
)
from phase3_password_inception.run_password_closure_inception import (
    load_final_inception,
    topk_strings_from_prob_vectors,
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


def load_keyness_model(path: str):
    return joblib.load(path)


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
    sequence_hit_cutoff: int = 100,
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
    store_top_n = max(10, int(sequence_hit_cutoff))
    out = {
        "prediction": pred_text,
        "mean_max_prob": float(np.mean(max_prob)) if len(max_prob) else 0.0,
        "mean_margin": float(np.mean(margin)) if len(margin) else 0.0,
        "avg_log_prob": float(np.mean([c["log_prob"] for c in seq_candidates[:1]])) if seq_candidates else -1e9,
        "top_sequence_candidates": seq_candidates[:store_top_n],
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
        seq_key = f"sequence_top{int(sequence_hit_cutoff)}_hit"
        topn = [x["candidate"] for x in seq_candidates[: max(int(sequence_hit_cutoff), 1)]]
        out[seq_key] = float(ref in topn)
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
    beam_width: int = 100,
    branch_topk: int = 5,
    sequence_hit_cutoff: int = 100,
    pre_ms: float = 100.0,
    post_ms: float = 200.0,
    norm_mode: str = "global",
) -> dict | None:
    windows = []
    for frame in local_frames.tolist():
        win = _extract_window_from_signal(
            crop_imu,
            int(frame),
            sample_rate_hz,
            target_len,
            pre_ms=pre_ms,
            post_ms=post_ms,
        )
        if win is None:
            return None
        windows.append(win)
    xb = np.stack(windows).astype(np.float32)
    if norm_mode == "per_window":
        per_mean = xb.mean(axis=1, keepdims=True)
        per_std = xb.std(axis=1, keepdims=True)
        xb = (xb - per_mean) / (per_std + 1e-6)
    else:
        xb = (xb - means[None, None, :]) / (stds[None, None, :] + 1e-6)
    with torch.no_grad():
        logits = classifier(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
    out = _decode_logits(
        logits,
        classes,
        ref=ref,
        beam_width=beam_width,
        branch_topk=branch_topk,
        sequence_hit_cutoff=sequence_hit_cutoff,
    )
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
    beam_width: int = 100,
    branch_topk: int = 5,
    sequence_hit_cutoff: int = 100,
    norm_mode: str = "global",
) -> dict | None:
    if len(local_frames) == 0:
        return None
    with torch.no_grad():
        imu = torch.tensor(crop_imu, dtype=torch.float32, device=device)
        key_frames = torch.tensor(local_frames, dtype=torch.long, device=device)
        out = overlap_model.forward_episode(imu, key_frames, sample_rate_hz)
        windows = out["windows"].detach().cpu().numpy()
    if norm_mode == "per_window":
        per_mean = windows.mean(axis=1, keepdims=True)
        per_std = windows.std(axis=1, keepdims=True)
        xb = (windows - per_mean) / (per_std + 1e-6)
    else:
        xb = (windows - means[None, None, :]) / (stds[None, None, :] + 1e-6)
    with torch.no_grad():
        logits = classifier(torch.tensor(xb, dtype=torch.float32, device=device)).cpu().numpy()
    dec = _decode_logits(
        logits,
        classes,
        ref=ref,
        beam_width=beam_width,
        branch_topk=branch_topk,
        sequence_hit_cutoff=sequence_hit_cutoff,
    )
    dec["mode"] = "overlap"
    dec["windows_n"] = int(len(windows))
    dec["overlap_starts"] = [float(x) for x in out["starts"].detach().cpu().tolist()]
    dec["overlap_ends"] = [float(x) for x in out["ends"].detach().cpu().tolist()]
    return dec


def _load_stage3_runtime_params(checkpoint_path: str) -> tuple[float, float, str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pre_ms = float(ckpt.get("pre_ms", 100.0))
    post_ms = float(ckpt.get("post_ms", 200.0))
    norm_mode = str(ckpt.get("norm_mode", "global"))
    return pre_ms, post_ms, norm_mode


def _infer_expected_keys_from_region(
    crop_imu: np.ndarray,
    crop_ts: np.ndarray,
    length_model,
    sample_rate_hz: float,
) -> tuple[int | None, dict, tuple[int, int] | None]:
    if length_model is None:
        return None, {"used_length_model": False}, None
    model, labels, meta = length_model
    subregion, sub_debug = build_length_subregion_from_energy(crop_imu, sample_rate_hz)
    if subregion is not None:
        lo, hi = subregion
        feat_imu = crop_imu[lo:hi]
        feat_ts = crop_ts[lo:hi]
    else:
        feat_imu = crop_imu
        feat_ts = crop_ts

    feature_mode = meta.get("feature_mode")
    if feature_mode is None:
        n_features = int(getattr(model, "n_features_in_", 0) or 0)
        # Historical checkpoints often have empty meta; infer the intended mode.
        feature_mode = "no_time" if n_features == 24 else "legacy_time"
    feature_mode = str(feature_mode)

    feat = compute_region_length_features(feat_imu, feat_ts, feature_mode=feature_mode).reshape(1, -1)
    pred = int(model.predict(feat)[0])
    debug = {
        "used_length_model": True,
        "predicted_length": pred,
        "candidate_labels": [int(x) for x in labels],
        "feature_num_frames": int(len(feat_imu)),
        "feature_duration_s": float((feat_ts[-1] - feat_ts[0]) / 1e9) if len(feat_ts) > 1 else 0.0,
        "feature_mode": feature_mode,
        "subregion_debug": sub_debug,
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0]
        debug["length_probs"] = {str(int(lbl)): float(p) for lbl, p in zip(model.classes_, proba)}
    return pred, debug, subregion


def _propose_keyness_anchors(
    raw_imu: np.ndarray,
    sample_rate_hz: float,
    keyness_model,
    threshold: float,
    min_keys: int,
    max_keys: int,
    gap_prior_s: float,
    force_k: int | None = None,
) -> tuple[np.ndarray, dict]:
    ep = {
        "imu": raw_imu,
        "sample_rate_hz": float(sample_rate_hz),
    }
    peaks, sm, _ = _propose_peaks(ep)
    if len(peaks) == 0:
        return np.asarray([], dtype=np.int64), {
            "method": "peak_keyness",
            "num_raw_peaks": 0,
            "selection_mode": "no_peaks",
            "threshold": float(threshold),
        }

    feats = np.stack([
        _peak_feature_vector(sm, peaks, i, sample_rate_hz)
        for i in range(len(peaks))
    ]).astype(np.float32)
    probs = keyness_model.predict_proba(feats)[:, 1]

    if force_k is not None and force_k > 0:
        anchors = _select_k_peaks(peaks, probs, int(force_k), sample_rate_hz, gap_prior_s=gap_prior_s)
        selection_mode = "force_k_dp"
    else:
        mask = probs >= float(threshold)
        chosen = peaks[mask]
        selection_mode = "threshold"
        if len(chosen) < min_keys:
            top_idx = np.argsort(-probs)[: min(min_keys, len(peaks))]
            chosen = np.sort(peaks[top_idx])
            selection_mode = "fallback_topk_min"
        elif len(chosen) > max_keys:
            chosen_idx = np.where(mask)[0]
            top_idx = chosen_idx[np.argsort(-probs[chosen_idx])[:max_keys]]
            chosen = np.sort(peaks[top_idx])
            selection_mode = "clamped_max"
        anchors = chosen

    debug = {
        "method": "peak_keyness",
        "threshold": float(threshold),
        "selection_mode": selection_mode,
        "num_raw_peaks": int(len(peaks)),
        "num_chosen": int(len(anchors)),
        "peak_frames": [int(x) for x in peaks.tolist()],
        "peak_probs": [float(x) for x in probs.tolist()],
    }
    return np.asarray(anchors, dtype=np.int64), debug


def _decode_candidate_segment_strong(
    overlap_model,
    runtime_classifier,
    stage3_model,
    stage3_target_len: int,
    stage3_classes: np.ndarray,
    stage3_means: np.ndarray,
    stage3_stds: np.ndarray,
    length_model,
    device: torch.device,
    crop_imu: np.ndarray,
    crop_ts: np.ndarray,
    sample_rate_hz: float,
    ref: str | None,
    min_keys: int,
    max_keys: int,
    gap_prior_s: float,
    oracle_align_to_ref_k: bool,
    force_ref_key_count: bool,
    multi_k_hypotheses: list[int] | None = None,
    length_prior_weight: float = 0.15,
    keyness_model=None,
    keyness_threshold: float = 0.5,
    beam_width: int = 100,
    branch_topk: int = 5,
    sequence_hit_cutoff: int = 100,
    stage3_pre_ms: float = 100.0,
    stage3_post_ms: float = 200.0,
    stage3_norm_mode: str = "global",
) -> dict:
    if keyness_model is not None:
        force_k = int(len(ref)) if (force_ref_key_count and ref is not None) else None
        local_anchor_frames, anchor_debug = _propose_keyness_anchors(
            crop_imu,
            sample_rate_hz,
            keyness_model,
            threshold=keyness_threshold,
            min_keys=min_keys,
            max_keys=max_keys,
            gap_prior_s=gap_prior_s,
            force_k=force_k,
        )
        local_anchor_frames = np.asarray(local_anchor_frames, dtype=np.int64)
        candidate_score = score_candidate_region_from_frames(
            runtime_classifier,
            crop_imu,
            local_anchor_frames if len(local_anchor_frames) else np.asarray([], dtype=np.int64),
            sample_rate_hz,
            device,
        )

        fixed = None
        overlap = None
        if len(local_anchor_frames):
            fixed = _run_stage3_fixed(
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                crop_imu,
                sample_rate_hz,
                local_anchor_frames,
                ref,
                beam_width=beam_width,
                branch_topk=branch_topk,
                sequence_hit_cutoff=sequence_hit_cutoff,
                pre_ms=stage3_pre_ms,
                post_ms=stage3_post_ms,
                norm_mode=stage3_norm_mode,
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
                local_anchor_frames,
                ref,
                beam_width=beam_width,
                branch_topk=branch_topk,
                sequence_hit_cutoff=sequence_hit_cutoff,
                norm_mode=stage3_norm_mode,
            )

        anchor_debug["force_ref_key_count"] = bool(force_ref_key_count)
        anchor_debug["expected_keys_used"] = None
        anchor_debug["oracle_align_to_ref_k"] = False
        anchor_debug["length_debug"] = {"used_length_model": False, "reason": "keyness_model_path"}
        for result in (fixed, overlap):
            if result is not None:
                result["chosen_k"] = int(len(local_anchor_frames))
                result["selected_frames"] = [int(x) for x in local_anchor_frames.tolist()]
                result["candidate_anchor_score"] = None if candidate_score is None else float(candidate_score["score"])
                result["candidate_mean_max_prob"] = None if candidate_score is None else float(candidate_score["mean_max_prob"])
                result["candidate_mean_margin"] = None if candidate_score is None else float(candidate_score["mean_margin"])
                result["pred_text_runtime"] = None if candidate_score is None else str(candidate_score["pred_text"])
                result["length_debug"] = {"used_length_model": False, "reason": "keyness_model_path"}
                result["anchor_debug"] = anchor_debug

        return {
            "num_proposed_peaks": int(anchor_debug.get("num_raw_peaks", 0)),
            "fixed": fixed,
            "overlap": overlap,
        }

    inferred_keys, length_debug, length_subregion = _infer_expected_keys_from_region(
        crop_imu,
        crop_ts,
        length_model,
        sample_rate_hz,
    )
    expected_keys = int(inferred_keys) if inferred_keys is not None else 0
    if force_ref_key_count and ref is not None:
        expected_keys = int(len(ref))

    work_crop_start = 0
    work_imu = crop_imu
    work_ts = crop_ts
    if length_subregion is not None:
        sub_lo, sub_hi = length_subregion
        if sub_hi - sub_lo >= 10:
            work_crop_start = int(sub_lo)
            work_imu = crop_imu[sub_lo:sub_hi]
            work_ts = crop_ts[sub_lo:sub_hi]

    if multi_k_hypotheses is not None and not force_ref_key_count:
        length_probs = {}
        if length_debug.get("length_probs"):
            length_probs = {int(k): float(v) for k, v in length_debug["length_probs"].items()}

        best_fixed = None
        best_overlap = None
        best_fixed_score = -float("inf")
        best_overlap_score = -float("inf")
        best_fixed_anchors = None
        best_overlap_anchors = None
        best_fixed_k = None
        best_overlap_k = None
        best_fixed_anchor_debug = None
        best_overlap_anchor_debug = None
        all_hypotheses = []

        for k_hyp in multi_k_hypotheses:
            anchors_k, anchor_debug_k = propose_energy_classifier_anchors(
                work_imu,
                sample_rate_hz,
                runtime_classifier,
                device,
                expected_keys=int(k_hyp),
                min_keys=int(k_hyp),
                max_keys=int(k_hyp),
                gap_prior_s=gap_prior_s,
            )
            anchors_k = np.asarray(anchors_k, dtype=np.int64)
            if len(anchors_k) == 0:
                all_hypotheses.append({"k": int(k_hyp), "status": "no_anchors"})
                continue

            anchors_global = anchors_k + int(work_crop_start)
            fixed_k = _run_stage3_fixed(
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                crop_imu,
                sample_rate_hz,
                anchors_global,
                ref,
                beam_width=beam_width,
                branch_topk=branch_topk,
                sequence_hit_cutoff=sequence_hit_cutoff,
                pre_ms=stage3_pre_ms,
                post_ms=stage3_post_ms,
                norm_mode=stage3_norm_mode,
            )
            overlap_k = _run_stage3_overlap(
                overlap_model,
                stage3_model,
                stage3_classes,
                stage3_means,
                stage3_stds,
                device,
                crop_imu,
                sample_rate_hz,
                anchors_global,
                ref,
                beam_width=beam_width,
                branch_topk=branch_topk,
                sequence_hit_cutoff=sequence_hit_cutoff,
                norm_mode=stage3_norm_mode,
            )

            lp = math.log(max(length_probs.get(int(k_hyp), 1.0 / max(len(multi_k_hypotheses), 1)), 1e-6))
            hyp_info = {
                "k": int(k_hyp),
                "status": "ok",
                "num_anchors": int(len(anchors_global)),
                "length_log_prior": float(lp),
            }

            if fixed_k is not None:
                stage3_score_fixed = fixed_k.get("avg_log_prob", -1e9)
                combined_fixed = stage3_score_fixed + length_prior_weight * lp
                hyp_info["fixed_avg_log_prob"] = float(stage3_score_fixed)
                hyp_info["fixed_combined_score"] = float(combined_fixed)
                if combined_fixed > best_fixed_score:
                    best_fixed_score = combined_fixed
                    best_fixed = fixed_k
                    best_fixed_anchors = anchors_global
                    best_fixed_k = int(k_hyp)
                    best_fixed_anchor_debug = anchor_debug_k

            if overlap_k is not None:
                stage3_score_overlap = overlap_k.get("avg_log_prob", -1e9)
                combined_overlap = stage3_score_overlap + length_prior_weight * lp
                hyp_info["overlap_avg_log_prob"] = float(stage3_score_overlap)
                hyp_info["overlap_combined_score"] = float(combined_overlap)
                if combined_overlap > best_overlap_score:
                    best_overlap_score = combined_overlap
                    best_overlap = overlap_k
                    best_overlap_anchors = anchors_global
                    best_overlap_k = int(k_hyp)
                    best_overlap_anchor_debug = anchor_debug_k

            all_hypotheses.append(hyp_info)

        multi_k_debug = {
            "multi_k_hypotheses": [int(x) for x in multi_k_hypotheses],
            "length_prior_weight": float(length_prior_weight),
            "all_hypotheses": all_hypotheses,
            "length_debug": length_debug,
            "work_crop_start": int(work_crop_start),
        }

        if best_fixed is not None and best_fixed_anchors is not None:
            candidate_score = score_candidate_region_from_frames(
                runtime_classifier,
                work_imu,
                np.asarray(best_fixed_anchors - int(work_crop_start), dtype=np.int64),
                sample_rate_hz,
                device,
            )
            best_fixed["chosen_k"] = int(best_fixed_k)
            best_fixed["selected_frames"] = [int(x) for x in best_fixed_anchors.tolist()]
            best_fixed["multi_k_debug"] = multi_k_debug
            best_fixed["length_debug"] = length_debug
            best_fixed["anchor_debug"] = best_fixed_anchor_debug
            best_fixed["candidate_anchor_score"] = None if candidate_score is None else float(candidate_score["score"])
            best_fixed["candidate_mean_max_prob"] = None if candidate_score is None else float(candidate_score["mean_max_prob"])
            best_fixed["candidate_mean_margin"] = None if candidate_score is None else float(candidate_score["mean_margin"])
            best_fixed["pred_text_runtime"] = None if candidate_score is None else str(candidate_score["pred_text"])

        if best_overlap is not None and best_overlap_anchors is not None:
            candidate_score_ov = score_candidate_region_from_frames(
                runtime_classifier,
                work_imu,
                np.asarray(best_overlap_anchors - int(work_crop_start), dtype=np.int64),
                sample_rate_hz,
                device,
            )
            best_overlap["chosen_k"] = int(best_overlap_k)
            best_overlap["selected_frames"] = [int(x) for x in best_overlap_anchors.tolist()]
            best_overlap["multi_k_debug"] = multi_k_debug
            best_overlap["length_debug"] = length_debug
            best_overlap["anchor_debug"] = best_overlap_anchor_debug
            best_overlap["candidate_anchor_score"] = None if candidate_score_ov is None else float(candidate_score_ov["score"])
            best_overlap["candidate_mean_max_prob"] = None if candidate_score_ov is None else float(candidate_score_ov["mean_max_prob"])
            best_overlap["candidate_mean_margin"] = None if candidate_score_ov is None else float(candidate_score_ov["mean_margin"])
            best_overlap["pred_text_runtime"] = None if candidate_score_ov is None else str(candidate_score_ov["pred_text"])

        return {
            "num_proposed_peaks": int(all_hypotheses[0].get("num_anchors", 0)) if all_hypotheses else 0,
            "fixed": best_fixed,
            "overlap": best_overlap,
        }

    local_anchor_frames, anchor_debug = propose_energy_classifier_anchors(
        work_imu,
        sample_rate_hz,
        runtime_classifier,
        device,
        expected_keys=expected_keys,
        min_keys=min_keys,
        max_keys=max_keys,
        gap_prior_s=gap_prior_s,
    )
    local_anchor_frames = np.asarray(local_anchor_frames, dtype=np.int64)
    if len(local_anchor_frames):
        local_anchor_frames = local_anchor_frames + int(work_crop_start)

    # Historical "strongest single" evaluation interpolated anchors back to the
    # reference key count when auto anchors did not match the true length.
    # Keep this behind an explicit flag so the strict automatic path remains
    # unchanged by default.
    if oracle_align_to_ref_k and ref is not None and len(local_anchor_frames) > 0:
        target_k = int(len(ref))
        if target_k > 0 and len(local_anchor_frames) != target_k:
            xs = np.linspace(0, len(local_anchor_frames) - 1, target_k)
            local_anchor_frames = np.interp(
                xs,
                np.arange(len(local_anchor_frames)),
                local_anchor_frames.astype(np.float64),
            ).round().astype(np.int64)

    anchor_debug["expected_keys_used"] = int(expected_keys)
    anchor_debug["length_debug"] = length_debug
    anchor_debug["work_crop_start"] = int(work_crop_start)
    anchor_debug["oracle_align_to_ref_k"] = bool(oracle_align_to_ref_k)
    anchor_debug["force_ref_key_count"] = bool(force_ref_key_count)

    candidate_score = score_candidate_region_from_frames(
        runtime_classifier,
        work_imu,
        np.asarray(local_anchor_frames - int(work_crop_start), dtype=np.int64) if len(local_anchor_frames) else np.asarray([], dtype=np.int64),
        sample_rate_hz,
        device,
    )

    fixed = None
    overlap = None
    if len(local_anchor_frames):
        fixed = _run_stage3_fixed(
            stage3_model,
            stage3_target_len,
            stage3_classes,
            stage3_means,
            stage3_stds,
            device,
            crop_imu,
            sample_rate_hz,
            local_anchor_frames,
            ref,
            beam_width=beam_width,
            branch_topk=branch_topk,
            sequence_hit_cutoff=sequence_hit_cutoff,
            pre_ms=stage3_pre_ms,
            post_ms=stage3_post_ms,
            norm_mode=stage3_norm_mode,
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
            local_anchor_frames,
            ref,
            beam_width=beam_width,
            branch_topk=branch_topk,
            sequence_hit_cutoff=sequence_hit_cutoff,
            norm_mode=stage3_norm_mode,
        )

    for result in (fixed, overlap):
        if result is not None:
            result["chosen_k"] = int(len(local_anchor_frames))
            result["selected_frames"] = [int(x) for x in local_anchor_frames.tolist()]
            result["candidate_anchor_score"] = None if candidate_score is None else float(candidate_score["score"])
            result["candidate_mean_max_prob"] = None if candidate_score is None else float(candidate_score["mean_max_prob"])
            result["candidate_mean_margin"] = None if candidate_score is None else float(candidate_score["mean_margin"])
            result["pred_text_runtime"] = None if candidate_score is None else str(candidate_score["pred_text"])
            result["length_debug"] = length_debug
            result["anchor_debug"] = anchor_debug

    return {
        "num_proposed_peaks": int(anchor_debug.get("num_raw_peaks", 0)),
        "fixed": fixed,
        "overlap": overlap,
    }


def _aggregate_rows(rows: list[dict], sequence_hit_cutoff: int = 100) -> dict:
    seq_key = f"sequence_top{int(sequence_hit_cutoff)}_hit"
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
            seq_key: 0.0,
        }
    num_chars = int(sum(len(r["reference"]) for r in rows))
    out = {
        "num_rows": len(rows),
        "num_chars": num_chars,
        "char_top1": float(sum(r["char_top1"] * len(r["reference"]) for r in rows) / max(num_chars, 1)),
        "char_top3": float(sum(r["char_top3"] * len(r["reference"]) for r in rows) / max(num_chars, 1)),
        "char_top5": float(sum(r["char_top5"] * len(r["reference"]) for r in rows) / max(num_chars, 1)),
        "cer": float(sum(levenshtein(r["reference"], r["prediction"]) for r in rows) / max(num_chars, 1)),
        "exact_match": float(np.mean([r["reference"] == r["prediction"] for r in rows])),
        "sequence_top100_hit": float(np.mean([r.get("sequence_top100_hit", 0.0) for r in rows])),
    }
    out[seq_key] = float(np.mean([r.get(seq_key, 0.0) for r in rows]))
    return out


def _pick_session_top_candidates(detail: dict, top_n: int) -> list[dict]:
    preds = detail.get("pred_segments_top5", [])[: max(int(top_n), 0)]
    return sorted(preds, key=lambda x: int(x["start_frame"]))


def _best_assignment_rows(gt_refs: list[str], cand_rows: list[dict], sequence_hit_cutoff: int = 100) -> list[dict]:
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
        seq_key = f"sequence_top{int(sequence_hit_cutoff)}_hit"
        topn = [x["candidate"] for x in row.get("top_sequence_candidates", [])[: max(int(sequence_hit_cutoff), 1)]]
        row[seq_key] = float(gt_refs[i] in topn)
        rows.append(row)
    return rows


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--stage1_details_json", required=True)
    ap.add_argument("--stage3_checkpoint", required=True)
    ap.add_argument("--stage3_scaler", required=True)
    ap.add_argument("--overlap_checkpoint", required=True)
    ap.add_argument("--length_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--min_keys", type=int, default=6)
    ap.add_argument("--max_keys", type=int, default=12)
    ap.add_argument("--gap_prior_s", type=float, default=1.3)
    ap.add_argument("--beam_width", type=int, default=100)
    ap.add_argument("--branch_topk", type=int, default=5)
    ap.add_argument("--sequence_hit_cutoff", type=int, default=100)
    ap.add_argument("--oracle_align_to_ref_k", action="store_true")
    ap.add_argument("--force_ref_key_count", action="store_true")
    ap.add_argument("--multi-k", nargs="*", type=int, default=None)
    ap.add_argument("--length-prior-weight", type=float, default=0.15)
    ap.add_argument("--keyness_model", default=None)
    ap.add_argument("--keyness_threshold", type=float, default=0.5)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    stage3_model, stage3_classes, stage3_means, stage3_stds = load_final_inception(
        args.stage3_checkpoint,
        args.stage3_scaler,
        device,
    )
    stage3_target_len = int(torch.load(args.stage3_checkpoint, map_location="cpu", weights_only=False)["n_timesteps"])
    stage3_pre_ms, stage3_post_ms, stage3_norm_mode = _load_stage3_runtime_params(args.stage3_checkpoint)
    stage3_model.eval()
    overlap_model = load_overlap_checkpoint(args.overlap_checkpoint, device)
    overlap_model.eval()
    overlap_model.freeze_classifier(True)
    runtime_classifier = load_external_inception(args.stage3_checkpoint, args.stage3_scaler, device)
    runtime_classifier.eval()
    length_model = load_length_model(args.length_model)
    keyness_model = load_keyness_model(args.keyness_model) if args.keyness_model else None

    eval_eps = build_password_episodes(args.eval_root)
    episode_map = {ep.episode_id: ep for ep in eval_eps}
    session_map = _session_prefix_map(args.eval_root)

    with open(args.stage1_details_json, "r", encoding="utf-8") as f:
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
                beam_width=args.beam_width,
                branch_topk=args.branch_topk,
                sequence_hit_cutoff=args.sequence_hit_cutoff,
                pre_ms=stage3_pre_ms,
                post_ms=stage3_post_ms,
                norm_mode=stage3_norm_mode,
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
                beam_width=args.beam_width,
                branch_topk=args.branch_topk,
                sequence_hit_cutoff=args.sequence_hit_cutoff,
                norm_mode=stage3_norm_mode,
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

            dec = _decode_candidate_segment_strong(
                overlap_model,
                runtime_classifier,
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                length_model,
                device,
                ep.imu,
                ep.timestamps_ns,
                ep.sample_rate_hz,
                ep.password,
                args.min_keys,
                args.max_keys,
                args.gap_prior_s,
                args.oracle_align_to_ref_k,
                args.force_ref_key_count,
                args.multi_k,
                args.length_prior_weight,
                keyness_model,
                args.keyness_threshold,
                args.beam_width,
                args.branch_topk,
                args.sequence_hit_cutoff,
                stage3_pre_ms,
                stage3_post_ms,
                stage3_norm_mode,
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

        for row in gt_rows_sorted:
            pred = row.get("best_pred")
            if pred is None:
                continue
            start = int(pred["start_frame"])
            end = int(pred["end_frame"])
            crop_imu = full_imu[start : end + 1]
            crop_ts = full_ts[start : end + 1]
            sr = float(1e9 / max(np.median(np.diff(crop_ts)), 1.0)) if len(crop_ts) >= 3 else 200.0
            dec = _decode_candidate_segment_strong(
                overlap_model,
                runtime_classifier,
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                length_model,
                device,
                crop_imu,
                crop_ts,
                sr,
                row["password"],
                args.min_keys,
                args.max_keys,
                args.gap_prior_s,
                args.oracle_align_to_ref_k,
                args.force_ref_key_count,
                args.multi_k,
                args.length_prior_weight,
                keyness_model,
                args.keyness_threshold,
                args.beam_width,
                args.branch_topk,
                args.sequence_hit_cutoff,
                stage3_pre_ms,
                stage3_post_ms,
                stage3_norm_mode,
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

        top_candidates = _pick_session_top_candidates(detail, len(gt_rows_sorted))
        fixed_candidates = []
        overlap_candidates = []
        for cand in top_candidates:
            start = int(cand["start_frame"])
            end = int(cand["end_frame"])
            crop_imu = full_imu[start : end + 1]
            crop_ts = full_ts[start : end + 1]
            sr = float(1e9 / max(np.median(np.diff(crop_ts)), 1.0)) if len(crop_ts) >= 3 else 200.0
            dec = _decode_candidate_segment_strong(
                overlap_model,
                runtime_classifier,
                stage3_model,
                stage3_target_len,
                stage3_classes,
                stage3_means,
                stage3_stds,
                length_model,
                device,
                crop_imu,
                crop_ts,
                sr,
                ref=None,
                min_keys=args.min_keys,
                max_keys=args.max_keys,
                gap_prior_s=args.gap_prior_s,
                oracle_align_to_ref_k=args.oracle_align_to_ref_k,
                force_ref_key_count=args.force_ref_key_count,
                multi_k_hypotheses=args.multi_k,
                length_prior_weight=args.length_prior_weight,
                keyness_model=keyness_model,
                keyness_threshold=args.keyness_threshold,
                beam_width=args.beam_width,
                branch_topk=args.branch_topk,
                sequence_hit_cutoff=args.sequence_hit_cutoff,
                stage3_pre_ms=stage3_pre_ms,
                stage3_post_ms=stage3_post_ms,
                stage3_norm_mode=stage3_norm_mode,
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
        sess_fixed = _best_assignment_rows(
            gt_refs,
            fixed_candidates,
            sequence_hit_cutoff=args.sequence_hit_cutoff,
        )
        sess_overlap = _best_assignment_rows(
            gt_refs,
            overlap_candidates,
            sequence_hit_cutoff=args.sequence_hit_cutoff,
        )
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
        "stage3_window_ms": {"pre_ms": float(stage3_pre_ms), "post_ms": float(stage3_post_ms)},
        "stage3_norm_mode": stage3_norm_mode,
        "length_model": args.length_model,
        "min_keys": args.min_keys,
        "max_keys": args.max_keys,
        "gap_prior_s": args.gap_prior_s,
        "beam_width": args.beam_width,
        "branch_topk": args.branch_topk,
        "sequence_hit_cutoff": args.sequence_hit_cutoff,
        "oracle_align_to_ref_k": bool(args.oracle_align_to_ref_k),
        "force_ref_key_count": bool(args.force_ref_key_count),
        "multi_k_hypotheses": args.multi_k,
        "length_prior_weight": args.length_prior_weight if args.multi_k else None,
        "keyness_model": args.keyness_model,
        "keyness_threshold": args.keyness_threshold if args.keyness_model else None,
        "gt_keyframes_fixed": _aggregate_rows(gt_keyframes_fixed_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
        "gt_keyframes_overlap": _aggregate_rows(gt_keyframes_overlap_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
        "gt_segment_fixed": _aggregate_rows(gt_fixed_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
        "gt_segment_overlap": _aggregate_rows(gt_overlap_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
        "stage1_bestpred_fixed": _aggregate_rows(oracle_fixed_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
        "stage1_bestpred_overlap": _aggregate_rows(oracle_overlap_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
        "session_top_fixed": _aggregate_rows(session_fixed_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
        "session_top_overlap": _aggregate_rows(session_overlap_rows, sequence_hit_cutoff=args.sequence_hit_cutoff),
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out_dir / "session_details.json", "w", encoding="utf-8") as f:
        json.dump(all_session_details, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_gt_fixed.json", "w", encoding="utf-8") as f:
        json.dump(gt_fixed_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_gt_overlap.json", "w", encoding="utf-8") as f:
        json.dump(gt_overlap_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_gt_keyframes_fixed.json", "w", encoding="utf-8") as f:
        json.dump(gt_keyframes_fixed_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_gt_keyframes_overlap.json", "w", encoding="utf-8") as f:
        json.dump(gt_keyframes_overlap_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_stage1_bestpred_fixed.json", "w", encoding="utf-8") as f:
        json.dump(oracle_fixed_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_stage1_bestpred_overlap.json", "w", encoding="utf-8") as f:
        json.dump(oracle_overlap_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_session_top_fixed.json", "w", encoding="utf-8") as f:
        json.dump(session_fixed_rows, f, ensure_ascii=False, indent=2)
    with open(out_dir / "rows_session_top_overlap.json", "w", encoding="utf-8") as f:
        json.dump(session_overlap_rows, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
