from __future__ import annotations

from typing import Any

import numpy as np
import torch

from inference.pipeline_inference import (
    _candidate_list,
    _resolve_ckpt_path,
    load_all_models as _base_load_all_models,
    run_ctc as _base_run_ctc,
    run_stage1 as _base_run_stage1,
)
from inference.preprocess import resample_to_target_hz
from onset_detection.stage2_segmental.scripts.eval_overlap_single_coarse_energy_cls import score_candidate_region_from_frames
from onset_detection.stage2_segmental.scripts.eval_stage123_end_to_end_strongstage2 import (
    _propose_keyness_anchors,
    _run_stage3_fixed,
)


def _scale_frames(frames: np.ndarray, src_hz: float, dst_hz: float, max_len: int) -> np.ndarray:
    if len(frames) == 0:
        return frames.astype(np.int64)
    scale = float(dst_hz) / float(src_hz)
    out = np.round(np.asarray(frames, dtype=np.float64) * scale).astype(np.int64)
    out = np.clip(out, 0, max(max_len - 1, 0))
    out = np.unique(out)
    return out.astype(np.int64)


def load_all_models(checkpoint_dir: str) -> dict[str, Any]:
    models = _base_load_all_models(checkpoint_dir)
    manifest = dict(models.get("manifest", {}))
    runtime_cfg = dict(manifest.get("runtime", {}))

    stage3_ckpt = _resolve_ckpt_path(checkpoint_dir, manifest["stage3"])
    stage3_meta = torch.load(stage3_ckpt, map_location="cpu", weights_only=False)
    stage3_target_hz = float(stage3_meta.get("target_rate_hz", 190.0))

    models["stage3_target_hz"] = stage3_target_hz
    models["demo_capture_target_hz"] = float(runtime_cfg.get("capture_target_hz", stage3_target_hz))
    models["demo_capture_report_interval_us"] = int(runtime_cfg.get("capture_report_interval_us", 1250))
    models["demo_capture_device_rate_control"] = bool(runtime_cfg.get("capture_device_rate_control", True))
    models["demo_force_fixed_only"] = bool(runtime_cfg.get("force_fixed_only", True))
    return models


def run_stage1(imu_array: np.ndarray, models: dict[str, Any], source_hz: float | None = None) -> list[dict[str, Any]]:
    """
    Input is the raw/high-rate segment for the 800Hz demo path.
    Stage1 model still runs on its own training rate (190Hz), then segment
    boundaries are mapped back to the source coordinate system.
    """
    imu = np.asarray(imu_array, dtype=np.float32)
    stage1_hz = float(models["stage1_config"].get("sample_rate_hz", 190.0))
    src_hz = float(source_hz or stage1_hz)
    imu_stage1 = resample_to_target_hz(imu, src_hz, stage1_hz)
    segs_stage1 = _base_run_stage1(imu_stage1, models)
    scale = src_hz / stage1_hz if stage1_hz > 1e-6 else 1.0

    out: list[dict[str, Any]] = []
    for seg in segs_stage1:
        start_src = int(round(int(seg["start"]) * scale))
        end_src = int(round(int(seg["end"]) * scale))
        start_src = max(0, min(start_src, len(imu)))
        end_src = max(start_src, min(end_src, len(imu)))
        out.append(
            {
                "start": start_src,
                "end": end_src,
                "confidence": float(seg.get("confidence", 0.0)),
                "start_stage1": int(seg["start"]),
                "end_stage1": int(seg["end"]),
                "stage1_rate_hz": stage1_hz,
                "source_rate_hz": src_hz,
            }
        )
    return out


def run_pipeline_stage23(
    imu_segment: np.ndarray,
    models: dict[str, Any],
    beam_width: int = 500,
    source_hz: float | None = None,
) -> dict[str, Any]:
    """
    Dual-rate demo path:
    - keyness anchors on 190Hz view
    - Stage3 fixed decoding on 800Hz view
    - overlap route intentionally disabled for this demo because old 200Hz
      overlap model is mismatched on 800Hz data
    """
    imu = np.asarray(imu_segment, dtype=np.float32)
    stage12_hz = float(models["stage1_config"].get("sample_rate_hz", 190.0))
    stage3_hz = float(models.get("stage3_target_hz", stage12_hz))
    src_hz = float(source_hz or stage3_hz)

    defaults = dict(models.get("pipeline_defaults", {}))
    threshold = float(defaults.get("keyness_threshold", 0.7))
    min_keys = int(defaults.get("min_keys", 6))
    max_keys = int(defaults.get("max_keys", 12))
    gap_prior_s = float(defaults.get("gap_prior_s", 1.3))
    branch_topk = int(defaults.get("branch_topk", 5))
    sequence_hit_cutoff = int(defaults.get("sequence_hit_cutoff", max(100, beam_width)))

    imu_stage12 = resample_to_target_hz(imu, src_hz, stage12_hz)
    imu_stage3 = resample_to_target_hz(imu, src_hz, stage3_hz)

    local_anchor_frames_190, anchor_debug = _propose_keyness_anchors(
        imu_stage12,
        stage12_hz,
        models["keyness_rf"],
        threshold=threshold,
        min_keys=min_keys,
        max_keys=max_keys,
        gap_prior_s=gap_prior_s,
        force_k=None,
    )
    local_anchor_frames_190 = np.asarray(local_anchor_frames_190, dtype=np.int64)
    local_anchor_frames_800 = _scale_frames(local_anchor_frames_190, stage12_hz, stage3_hz, len(imu_stage3))

    anchor_debug = dict(anchor_debug)
    anchor_debug["stage12_rate_hz"] = float(stage12_hz)
    anchor_debug["stage3_rate_hz"] = float(stage3_hz)
    anchor_debug["source_rate_hz"] = float(src_hz)
    anchor_debug["selected_frames_190hz"] = [int(x) for x in local_anchor_frames_190.tolist()]
    anchor_debug["selected_frames_800hz"] = [int(x) for x in local_anchor_frames_800.tolist()]

    if len(local_anchor_frames_800) == 0:
        return {
            "num_keys": 0,
            "char_top1": "",
            "top_candidates": [],
            "selected_frames": [],
            "selected_frames_190hz": [],
            "selected_frames_800hz": [],
            "mode_used": None,
            "anchor_debug": anchor_debug,
            "selection_debug": {"reason": "no_stage3_result"},
        }

    candidate_score = score_candidate_region_from_frames(
        models["runtime_stage3_classifier"],
        imu_stage3,
        local_anchor_frames_800,
        stage3_hz,
        models["device"],
    )

    fixed = _run_stage3_fixed(
        models["stage3_model"],
        models["stage3_target_len"],
        models["stage3_classes"],
        models["stage3_means"],
        models["stage3_stds"],
        models["device"],
        imu_stage3,
        stage3_hz,
        local_anchor_frames_800,
        None,
        beam_width=beam_width,
        branch_topk=branch_topk,
        sequence_hit_cutoff=sequence_hit_cutoff,
        pre_ms=float(models.get("stage3_pre_ms", 100.0)),
        post_ms=float(models.get("stage3_post_ms", 200.0)),
        norm_mode=str(models.get("stage3_norm_mode", "global")),
        use_diff_channels=bool(models.get("stage3_use_diff_channels", False)),
    )

    selection_debug = {
        "reason": "fixed_only_800hz_demo",
        "stage12_rate_hz": float(stage12_hz),
        "stage3_rate_hz": float(stage3_hz),
        "source_rate_hz": float(src_hz),
        "fixed_avg_log_prob": None if fixed is None else float(fixed.get("avg_log_prob", -1e9)),
    }

    if fixed is None:
        return {
            "num_keys": int(len(local_anchor_frames_800)),
            "char_top1": "",
            "top_candidates": [],
            "selected_frames": [int(x) for x in local_anchor_frames_800.tolist()],
            "selected_frames_190hz": [int(x) for x in local_anchor_frames_190.tolist()],
            "selected_frames_800hz": [int(x) for x in local_anchor_frames_800.tolist()],
            "mode_used": None,
            "anchor_debug": anchor_debug,
            "selection_debug": selection_debug,
        }

    return {
        "num_keys": int(len(local_anchor_frames_800)),
        "char_top1": str(fixed.get("prediction", "")),
        "top_candidates": _candidate_list(fixed),
        "selected_frames": [int(x) for x in local_anchor_frames_800.tolist()],
        "selected_frames_190hz": [int(x) for x in local_anchor_frames_190.tolist()],
        "selected_frames_800hz": [int(x) for x in local_anchor_frames_800.tolist()],
        "mode_used": "fixed",
        "anchor_debug": anchor_debug,
        "candidate_score": candidate_score,
        "fixed_prediction": str(fixed.get("prediction", "")),
        "overlap_prediction": None,
        "selection_debug": selection_debug,
    }


def run_ctc(imu_segment: np.ndarray, models: dict[str, Any], source_hz: float | None = None) -> dict[str, Any]:
    src_hz = float(source_hz or 190.0)
    imu = np.asarray(imu_segment, dtype=np.float32)
    imu190 = resample_to_target_hz(imu, src_hz, 190.0)
    return _base_run_ctc(imu190, models)
