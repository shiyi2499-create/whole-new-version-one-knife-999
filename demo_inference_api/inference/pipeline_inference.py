from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.signal import resample

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
for p in (REPO_ROOT, REPO_ROOT / 'onset_detection', REPO_ROOT / 'onset_detection' / 'stage2_ctc', REPO_ROOT / 'phase3_password_inception'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from onset_detection.stage2_ctc.models.frame_ctc import FrameCTCModel
from onset_detection.stage2_ctc.utils.decode import greedy_decode, prefix_beam_search
from onset_detection.stage2_ctc.utils.signal_processing import preprocess as ctc_preprocess
from onset_detection.stage2_segmental.model import load_external_inception
from onset_detection.stage2_segmental.model_v2 import load_overlap_checkpoint
from onset_detection.stage2_segmental.scripts.eval_overlap_single_coarse_energy_cls import score_candidate_region_from_frames
from onset_detection.stage2_segmental.scripts.eval_stage123_end_to_end_strongstage2 import (
    _propose_keyness_anchors,
    _run_stage3_fixed,
    _run_stage3_overlap,
)
from onset_detection.stage2_segmental.scripts.train_eval_stage1_dense_labeling import (
    UNet1D,
    _build_dense_features,
    extract_segments,
)
from phase3_password_inception.run_password_closure_inception import load_final_inception


DEFAULT_STAGE1_CONFIG = {
    'feature_mode': 'raw6_energy_activity_pulse',
    'sample_rate_hz': 190.0,
    'base_filters': 24,
    'depth': 4,
    'kernel_size': 7,
    'dropout': 0.1,
    'use_attention': False,
}


def _resolve_device(device_name: str = 'auto') -> torch.device:
    req = (device_name or 'auto').lower()
    if req == 'auto':
        if torch.cuda.is_available():
            req = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            req = 'mps'
        else:
            req = 'cpu'
    return torch.device(req)


def _load_manifest(checkpoint_dir: str) -> dict[str, Any]:
    checkpoint_root = Path(checkpoint_dir)
    manifest_path = checkpoint_root / 'CHECKPOINT_MANIFEST.json'
    if not manifest_path.exists():
        manifest_path = THIS_DIR / 'checkpoints' / 'CHECKPOINT_MANIFEST.json'
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    data['_checkpoint_dir'] = str(checkpoint_root)
    return data


def _resolve_ckpt_path(checkpoint_dir: str, entry: dict[str, Any]) -> str:
    root = Path(checkpoint_dir)
    filename = entry.get('filename')
    if filename:
        candidate = root / filename
        if candidate.exists():
            return str(candidate)
    source_path = entry.get('source_path')
    if source_path:
        return str((REPO_ROOT / source_path).resolve())
    raise FileNotFoundError(f'Cannot resolve checkpoint path for entry: {entry}')


def _coarse_merge_segments(segments: list[dict[str, Any]], max_gap_s: float, sr: float) -> list[dict[str, Any]]:
    if len(segments) <= 1 or max_gap_s <= 0:
        return list(segments)
    max_gap_frames = int(round(float(max_gap_s) * float(sr)))
    ordered = sorted((dict(seg) for seg in segments), key=lambda x: (int(x['start']), int(x['end'])))
    merged = [ordered[0]]
    for seg in ordered[1:]:
        if int(seg['start']) - int(merged[-1]['end']) <= max_gap_frames:
            merged[-1]['end'] = max(int(merged[-1]['end']), int(seg['end']))
            merged[-1]['confidence'] = max(float(merged[-1].get('confidence', 0.0)), float(seg.get('confidence', 0.0)))
        else:
            merged.append(seg)
    return merged


def _load_stage1_model(path: str, device: torch.device, cfg: dict[str, Any]) -> UNet1D:
    model = UNet1D(
        in_channels=9,
        base_filters=int(cfg['base_filters']),
        depth=int(cfg['depth']),
        kernel_size=int(cfg['kernel_size']),
        dropout=float(cfg['dropout']),
        use_attention=bool(cfg['use_attention']),
    ).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_ctc_model(path: str, device: torch.device) -> FrameCTCModel:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(blob['model_cfg'])
    model = FrameCTCModel(
        in_ch=int(cfg['input_channels']),
        hidden=int(cfg['hidden_channels']),
        num_layers=int(cfg['num_layers']),
        kernel=int(cfg['kernel_size']),
        dropout=float(cfg['dropout']),
        num_classes=int(cfg['num_classes']),
    ).to(device)
    model.load_state_dict(blob['model'])
    model.eval()
    return model


def _extract_window_from_signal(
    signal: np.ndarray,
    center_frame: int,
    sample_rate_hz: float,
    target_len: int,
    pre_ms: float = 100.0,
    post_ms: float = 200.0,
) -> np.ndarray | None:
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


def _candidate_list(result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not result:
        return []
    return [
        {'password': str(x['candidate']), 'score': float(x['log_prob'])}
        for x in result.get('top_sequence_candidates', [])
    ]


def load_all_models(checkpoint_dir: str) -> dict:
    """加载所有模型，返回 dict。"""
    manifest = _load_manifest(checkpoint_dir)
    device = _resolve_device(manifest.get('runtime', {}).get('device', 'auto'))

    stage1_cfg = dict(DEFAULT_STAGE1_CONFIG)
    stage1_cfg.update(manifest.get('stage1', {}).get('model_config', {}))
    stage1_posthoc = manifest.get('stage1_posthoc', {}).get('params', {})

    stage1_model = _load_stage1_model(
        _resolve_ckpt_path(checkpoint_dir, manifest['stage1']),
        device,
        stage1_cfg,
    )
    keyness_rf = joblib.load(_resolve_ckpt_path(checkpoint_dir, manifest['keyness_rf']))
    overlap_model = load_overlap_checkpoint(_resolve_ckpt_path(checkpoint_dir, manifest['overlap']), device)
    overlap_model.eval()
    overlap_model.freeze_classifier(True)

    stage3_ckpt = _resolve_ckpt_path(checkpoint_dir, manifest['stage3'])
    stage3_scaler = _resolve_ckpt_path(checkpoint_dir, manifest['stage3_scaler'])
    stage3_model, stage3_classes, stage3_means, stage3_stds = load_final_inception(stage3_ckpt, stage3_scaler, device)
    runtime_stage3_classifier = load_external_inception(stage3_ckpt, stage3_scaler, device)
    runtime_stage3_classifier.eval()
    stage3_target_len = int(torch.load(stage3_ckpt, map_location='cpu', weights_only=False)['n_timesteps'])

    ctc_model = _load_ctc_model(_resolve_ckpt_path(checkpoint_dir, manifest['ctc']), device)

    return {
        'device': device,
        'manifest': manifest,
        'stage1_model': stage1_model,
        'stage1_config': stage1_cfg,
        'stage1_posthoc': stage1_posthoc,
        'keyness_rf': keyness_rf,
        'overlap_model': overlap_model,
        'stage3_model': stage3_model,
        'stage3_classes': stage3_classes,
        'stage3_means': stage3_means,
        'stage3_stds': stage3_stds,
        'stage3_target_len': stage3_target_len,
        'runtime_stage3_classifier': runtime_stage3_classifier,
        'ctc_model': ctc_model,
        'pipeline_defaults': manifest.get('pipeline_defaults', {}),
        'ctc_defaults': manifest.get('ctc_defaults', {}),
    }


def run_stage1(imu_array: np.ndarray, models: dict) -> list:
    """
    输入：(T, 6) 的 IMU numpy array（已重采样到 190Hz）
    输出：检测到的段列表 [{'start': int, 'end': int, 'confidence': float}, ...]
    """
    imu = np.asarray(imu_array, dtype=np.float32)
    cfg = models['stage1_config']
    sr = float(cfg.get('sample_rate_hz', 190.0))
    features = _build_dense_features(imu, sr, feature_mode=str(cfg['feature_mode']))
    x = torch.from_numpy(features).float().unsqueeze(0).to(models['device'])
    with torch.no_grad():
        probs = torch.sigmoid(models['stage1_model'](x)).squeeze().cpu().numpy()

    params = models['stage1_posthoc']
    segs = extract_segments(
        probs,
        threshold=float(params.get('threshold', 0.6)),
        min_length=max(1, int(round(float(params.get('min_segment_s', 0.5)) * sr))),
        merge_gap=max(0, int(round(float(params.get('merge_gap_s', 0.25)) * sr))),
        prob_smooth_window=int(params.get('prob_smooth_window', 1)),
        valley_merge_threshold=float(params.get('valley_merge_threshold', 0.3)),
        valley_merge_max_gap=max(0, int(round(float(params.get('valley_merge_gap_s', 1.5)) * sr))),
    )
    out = [
        {'start': int(lo), 'end': int(hi), 'confidence': float(conf)}
        for lo, hi, conf in segs
    ]
    coarse_merge_gap_s = float(models.get('pipeline_defaults', {}).get('stage1_coarse_merge_gap_s', 0.0) or 0.0)
    if coarse_merge_gap_s > 0:
        out = _coarse_merge_segments(out, max_gap_s=coarse_merge_gap_s, sr=sr)
    return out


def run_pipeline_stage23(imu_segment: np.ndarray, models: dict, beam_width: int = 500) -> dict:
    """
    输入：一个密码段的 IMU array (T_seg, 6)
    输出：{
        'num_keys': int,
        'top_candidates': [{'password': str, 'score': float}, ...],
        'char_top1': str,
    }
    """
    imu = np.asarray(imu_segment, dtype=np.float32)
    sr = float(models['stage1_config'].get('sample_rate_hz', 190.0))
    defaults = dict(models.get('pipeline_defaults', {}))
    threshold = float(defaults.get('keyness_threshold', 0.7))
    min_keys = int(defaults.get('min_keys', 6))
    max_keys = int(defaults.get('max_keys', 12))
    gap_prior_s = float(defaults.get('gap_prior_s', 1.3))
    branch_topk = int(defaults.get('branch_topk', 5))
    sequence_hit_cutoff = int(defaults.get('sequence_hit_cutoff', max(100, beam_width)))

    local_anchor_frames, anchor_debug = _propose_keyness_anchors(
        imu,
        sr,
        models['keyness_rf'],
        threshold=threshold,
        min_keys=min_keys,
        max_keys=max_keys,
        gap_prior_s=gap_prior_s,
        force_k=None,
    )
    local_anchor_frames = np.asarray(local_anchor_frames, dtype=np.int64)

    fixed = None
    overlap = None
    candidate_score = None
    if len(local_anchor_frames):
        candidate_score = score_candidate_region_from_frames(
            models['runtime_stage3_classifier'],
            imu,
            local_anchor_frames,
            sr,
            models['device'],
        )
        fixed = _run_stage3_fixed(
            models['stage3_model'],
            models['stage3_target_len'],
            models['stage3_classes'],
            models['stage3_means'],
            models['stage3_stds'],
            models['device'],
            imu,
            sr,
            local_anchor_frames,
            None,
            beam_width=beam_width,
            branch_topk=branch_topk,
            sequence_hit_cutoff=sequence_hit_cutoff,
        )
        overlap = _run_stage3_overlap(
            models['overlap_model'],
            models['stage3_model'],
            models['stage3_classes'],
            models['stage3_means'],
            models['stage3_stds'],
            models['device'],
            imu,
            sr,
            local_anchor_frames,
            None,
            beam_width=beam_width,
            branch_topk=branch_topk,
            sequence_hit_cutoff=sequence_hit_cutoff,
        )

    chosen = overlap or fixed
    if chosen is None:
        return {
            'num_keys': 0,
            'char_top1': '',
            'top_candidates': [],
            'selected_frames': [],
            'mode_used': None,
            'anchor_debug': anchor_debug,
        }

    out = {
        'num_keys': int(len(local_anchor_frames)),
        'char_top1': str(chosen.get('prediction', '')),
        'top_candidates': _candidate_list(chosen),
        'selected_frames': [int(x) for x in local_anchor_frames.tolist()],
        'mode_used': str(chosen.get('mode', 'overlap' if overlap else 'fixed')),
        'anchor_debug': anchor_debug,
        'candidate_score': candidate_score,
        'fixed_prediction': None if fixed is None else str(fixed.get('prediction', '')),
        'overlap_prediction': None if overlap is None else str(overlap.get('prediction', '')),
    }
    return out


def run_ctc(imu_segment: np.ndarray, models: dict) -> dict:
    """
    输入：一个密码段的 IMU array (T_seg, 6)
    输出：{
        'prediction': str,
        'beam_candidates': [{'password': str, 'score': float}, ...],
    }
    """
    imu = np.asarray(imu_segment, dtype=np.float32)
    proc, _ = ctc_preprocess(imu, sample_rate=190, add_mag=True, norm=True, stats=None)
    x = torch.tensor(proc.T[None, ...], dtype=torch.float32, device=models['device'])
    with torch.no_grad():
        log_probs = models['ctc_model'].log_probs(x).squeeze(0).cpu().numpy()
    greedy = greedy_decode(log_probs)
    beam_width = int(models.get('ctc_defaults', {}).get('beam_width', 20))
    beam = prefix_beam_search(log_probs, beam_width=beam_width)
    return {
        'prediction': str(greedy),
        'beam_candidates': [
            {'password': str(x['candidate']), 'score': float(x['score'])}
            for x in beam
        ],
    }
