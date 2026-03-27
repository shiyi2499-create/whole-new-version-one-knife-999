from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'demo_inference_api') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'demo_inference_api'))
if str(REPO_ROOT / 'onset_detection') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'onset_detection'))

from inference.pipeline_inference import load_all_models
from inference.preprocess import csv_to_array, estimate_sample_rate, extract_timestamps, resample_to_190hz
from onset_detection.stage2_segmental.scripts.eval_stage123_end_to_end_strongstage2 import (
    _run_stage3_fixed,
    _run_stage3_overlap,
    load_external_inception,
)
from phase3_password_inception.run_password_closure_inception import load_final_inception


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
    with path.open('r', newline='') as f:
        return [
            {
                'timestamp_ns': int(r['timestamp_ns']),
                'key': r['key'],
                'event_type': r['event_type'],
            }
            for r in csv.DictReader(f)
        ]


def _align_presses_to_reference(events: list[dict[str, Any]], reference: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    presses = [e for e in events if e['event_type'] == 'press' and e['key'] not in {'enter', 'return', 'backspace'}]
    obs = ''.join(e['key'] if len(e['key']) == 1 else ' ' for e in presses)
    matched = []
    j = 0
    skipped = []
    for idx, e in enumerate(presses):
        if j < len(reference) and e['key'] == reference[j]:
            matched.append((idx, e))
            j += 1
            if j == len(reference):
                skipped.extend(list(range(idx + 1, len(presses))))
                break
        else:
            skipped.append(idx)
    ok = j == len(reference)
    debug = {
        'observed_press_text': obs,
        'reference_text': reference,
        'observed_press_count': len(presses),
        'matched_count': len(matched),
        'skipped_press_indices': skipped,
        'matched_press_indices': [i for i, _ in matched],
        'alignment_ok': ok,
    }
    return [e for _, e in matched], debug


def _ns_to_resampled_frame(timestamps_ns: np.ndarray, ts_ns: int, target_hz: float = 190.0) -> int:
    orig_hz = estimate_sample_rate(np.empty((len(timestamps_ns), 6), dtype=np.float32), timestamps_ns)
    scale = float(target_hz) / float(orig_hz) if orig_hz and orig_hz > 1e-6 else 1.0
    orig_idx = int(np.searchsorted(timestamps_ns, ts_ns, side='left'))
    return int(round(orig_idx * scale))


def _result_block(res: dict[str, Any] | None, truth: str) -> dict[str, Any] | None:
    if res is None:
        return None
    pred = str(res.get('prediction', ''))
    cands = res.get('top_sequence_candidates', []) or []
    truth_rank = None
    for i, c in enumerate(cands, 1):
        if str(c.get('candidate', '')) == truth:
            truth_rank = i
            break
    return {
        'prediction': pred,
        'cer': cer(truth, pred),
        'char_top1': float(res.get('char_top1', 0.0)) if 'char_top1' in res else None,
        'windows_n': int(res.get('windows_n', 0)),
        'top_candidates': [
            {'password': str(c['candidate']), 'score': float(c['log_prob'])}
            for c in cands[:10]
        ],
        'truth_rank_topk': truth_rank,
        'truth_in_top500': truth_rank is not None,
    }


def _load_stage3_window_params(checkpoint_path: Path) -> tuple[float, float, int]:
    ckpt = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    pre_ms = float(ckpt.get('pre_ms', 100.0))
    post_ms = float(ckpt.get('post_ms', 200.0))
    target_len = int(ckpt['n_timesteps'])
    return pre_ms, post_ms, target_len


def main() -> int:
    ap = argparse.ArgumentParser(description='GT segment + GT key Stage3 eval on still-password-still probe set')
    ap.add_argument('--dataset-root', default='data/raw/clean_password_probe')
    ap.add_argument('--checkpoint-dir', default='')
    ap.add_argument('--stage3-checkpoint', default='')
    ap.add_argument('--stage3-scaler', default='')
    ap.add_argument('--output-dir', default='results/still_password_probe_gt_stage3_eval')
    ap.add_argument('--beam-width', type=int, default=500)
    ap.add_argument('--pre-margin-sec', type=float, default=0.20)
    ap.add_argument('--post-margin-sec', type=float, default=0.30)
    args = ap.parse_args()

    dataset_root = (REPO_ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or str(REPO_ROOT)

    protocol_paths = sorted(dataset_root.glob('*_protocol.json'))
    if not protocol_paths:
        raise SystemExit(f'No protocol files found in {dataset_root}')

    models = load_all_models(checkpoint_dir)
    if args.stage3_checkpoint:
        stage3_checkpoint = (REPO_ROOT / args.stage3_checkpoint).resolve() if not Path(args.stage3_checkpoint).is_absolute() else Path(args.stage3_checkpoint)
        if not args.stage3_scaler:
            raise SystemExit('--stage3-scaler is required when --stage3-checkpoint is provided')
        stage3_scaler = (REPO_ROOT / args.stage3_scaler).resolve() if not Path(args.stage3_scaler).is_absolute() else Path(args.stage3_scaler)
        stage3_model, stage3_classes, stage3_means, stage3_stds = load_final_inception(str(stage3_checkpoint), str(stage3_scaler), models['device'])
        runtime_stage3_classifier = load_external_inception(str(stage3_checkpoint), str(stage3_scaler), models['device'])
        runtime_stage3_classifier.eval()
        pre_ms, post_ms, target_len = _load_stage3_window_params(stage3_checkpoint)
        models['stage3_model'] = stage3_model
        models['stage3_classes'] = stage3_classes
        models['stage3_means'] = stage3_means
        models['stage3_stds'] = stage3_stds
        models['runtime_stage3_classifier'] = runtime_stage3_classifier
        models['stage3_target_len'] = target_len
        models['stage3_pre_ms'] = pre_ms
        models['stage3_post_ms'] = post_ms
    sr = float(models['stage1_config'].get('sample_rate_hz', 190.0))
    rows = []

    for proto_path in protocol_paths:
        protocol = json.loads(proto_path.read_text(encoding='utf-8'))
        truth = str(protocol['reference_text'])
        sensor_csv = Path(protocol['paths']['sensor_csv'])
        events_csv = Path(protocol['paths']['events_csv'])
        events = _load_events(events_csv)
        matched_presses, align_debug = _align_presses_to_reference(events, truth)

        csv_string = sensor_csv.read_text(encoding='utf-8')
        timestamps = extract_timestamps(csv_string)
        imu = csv_to_array(csv_string)
        orig_hz = estimate_sample_rate(imu, timestamps)
        imu190 = resample_to_190hz(imu, orig_hz)

        gt_frames_global = np.asarray([_ns_to_resampled_frame(timestamps, e['timestamp_ns'], target_hz=sr) for e in matched_presses], dtype=np.int64)
        crop_start = max(0, int(min(gt_frames_global) - round(args.pre_margin_sec * sr)))
        crop_end = min(len(imu190), int(max(gt_frames_global) + round(args.post_margin_sec * sr)))
        crop = imu190[crop_start:crop_end]
        local_frames = gt_frames_global - crop_start

        fixed = _run_stage3_fixed(
            models['stage3_model'],
            models['stage3_target_len'],
            models['stage3_classes'],
            models['stage3_means'],
            models['stage3_stds'],
            models['device'],
            crop,
            sr,
            local_frames,
            ref=truth,
            beam_width=args.beam_width,
            branch_topk=5,
            sequence_hit_cutoff=max(100, args.beam_width),
            pre_ms=float(models.get('stage3_pre_ms', 100.0)),
            post_ms=float(models.get('stage3_post_ms', 200.0)),
        )
        overlap = _run_stage3_overlap(
            models['overlap_model'],
            models['stage3_model'],
            models['stage3_classes'],
            models['stage3_means'],
            models['stage3_stds'],
            models['device'],
            crop,
            sr,
            local_frames,
            ref=truth,
            beam_width=args.beam_width,
            branch_topk=5,
            sequence_hit_cutoff=max(100, args.beam_width),
        )

        row = {
            'session': sensor_csv.stem.replace('_sensor', ''),
            'truth': truth,
            'truth_len': len(truth),
            'alignment_debug': align_debug,
            'matched_press_timestamps_ns': [int(e['timestamp_ns']) for e in matched_presses],
            'matched_press_keys': ''.join(e['key'] for e in matched_presses),
            'gt_frames_global': gt_frames_global.tolist(),
            'gt_segment': {
                'start': int(crop_start),
                'end': int(crop_end),
                'duration_sec': float((crop_end - crop_start) / sr),
            },
            'stage3_fixed': _result_block(fixed, truth),
            'stage3_overlap': _result_block(overlap, truth),
        }
        rows.append(row)

    def mean_metric(key_path: tuple[str, str]) -> float:
        vals = []
        for r in rows:
            blk = r.get(key_path[0]) or {}
            v = blk.get(key_path[1])
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float('nan')

    summary = {
        'n_samples': len(rows),
        'stage3_window_ms': {
            'pre_ms': float(models.get('stage3_pre_ms', 100.0)),
            'post_ms': float(models.get('stage3_post_ms', 200.0)),
        },
        'fixed_mean_cer': mean_metric(('stage3_fixed', 'cer')),
        'overlap_mean_cer': mean_metric(('stage3_overlap', 'cer')),
        'fixed_exact': int(sum((r['stage3_fixed'] or {}).get('prediction', '') == r['truth'] for r in rows)),
        'overlap_exact': int(sum((r['stage3_overlap'] or {}).get('prediction', '') == r['truth'] for r in rows)),
        'fixed_truth_in_top500': int(sum(bool((r['stage3_fixed'] or {}).get('truth_in_top500')) for r in rows)),
        'overlap_truth_in_top500': int(sum(bool((r['stage3_overlap'] or {}).get('truth_in_top500')) for r in rows)),
    }

    (output_dir / 'rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    with (output_dir / 'rows.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['session', 'truth', 'fixed_pred', 'fixed_cer', 'fixed_truth_rank', 'overlap_pred', 'overlap_cer', 'overlap_truth_rank', 'matched_press_keys', 'gt_segment_sec'])
        for r in rows:
            sf = r['stage3_fixed'] or {}
            so = r['stage3_overlap'] or {}
            w.writerow([
                r['session'], r['truth'], sf.get('prediction', ''), sf.get('cer', ''), sf.get('truth_rank_topk', ''),
                so.get('prediction', ''), so.get('cer', ''), so.get('truth_rank_topk', ''), r['matched_press_keys'], r['gt_segment']['duration_sec']
            ])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
