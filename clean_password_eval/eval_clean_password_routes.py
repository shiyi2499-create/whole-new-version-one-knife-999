from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'demo_inference_api') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'demo_inference_api'))

from inference.pipeline_inference import load_all_models, run_ctc, run_pipeline_stage23, run_stage1
from inference.preprocess import csv_to_array, estimate_sample_rate, extract_timestamps, resample_to_190hz


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
    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'timestamp_ns': int(row['timestamp_ns']),
                'key': row['key'],
                'event_type': row['event_type'],
            })
    return rows


def _event_segment_indices(timestamps_ns: np.ndarray, events: list[dict[str, Any]], target_hz: float = 190.0) -> tuple[int | None, int | None, float | None]:
    if len(timestamps_ns) == 0:
        return None, None, None
    press_ts = [e['timestamp_ns'] for e in events if e['event_type'] == 'press' and e['key'] != 'enter']
    enter_ts = [e['timestamp_ns'] for e in events if e['event_type'] == 'press' and e['key'] == 'enter']
    if not press_ts or not enter_ts:
        return None, None, None
    start_ns = min(press_ts)
    end_ns = min(t for t in enter_ts if t >= start_ns) if any(t >= start_ns for t in enter_ts) else max(enter_ts)
    lo = int(np.searchsorted(timestamps_ns, start_ns, side='left'))
    hi = int(np.searchsorted(timestamps_ns, end_ns, side='right'))
    orig_hz = estimate_sample_rate(np.empty((len(timestamps_ns), 6), dtype=np.float32), timestamps_ns)
    scale = float(target_hz) / float(orig_hz) if orig_hz and orig_hz > 1e-6 else 1.0
    return int(round(lo * scale)), int(round(hi * scale)), float(scale)


def _choose_stage1_segment(segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not segments:
        return None
    return max(segments, key=lambda s: (float(s.get('confidence', 0.0)), int(s['end']) - int(s['start'])))


def _iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return 0.0 if union <= 0 else inter / union


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate pipeline and CTC on the eval-only clean password dataset.')
    parser.add_argument('--dataset-root', default='data/raw/clean_password_eval')
    parser.add_argument('--checkpoint-dir', default='')
    parser.add_argument('--output-dir', default='results/clean_password_eval_routes')
    parser.add_argument('--beam-width', type=int, default=500)
    args = parser.parse_args()

    dataset_root = (REPO_ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or str(REPO_ROOT)

    protocol_paths = sorted(dataset_root.glob('*_protocol.json'))
    if not protocol_paths:
        raise SystemExit(f'No protocol files found in {dataset_root}')

    models = load_all_models(checkpoint_dir)
    rows = []

    for proto_path in protocol_paths:
        protocol = json.loads(proto_path.read_text(encoding='utf-8'))
        prefix = protocol['session_prefix']
        ref = protocol['reference_text']
        sensor_csv = Path(protocol['paths']['sensor_csv'])
        events_csv = Path(protocol['paths']['events_csv'])
        csv_string = sensor_csv.read_text(encoding='utf-8')
        timestamps = extract_timestamps(csv_string)
        imu = csv_to_array(csv_string)
        orig_hz = estimate_sample_rate(imu, timestamps)
        imu190 = resample_to_190hz(imu, orig_hz)
        segments = run_stage1(imu190, models)
        best_seg = _choose_stage1_segment(segments)
        events = _load_events(events_csv)
        ev_lo, ev_hi, _ = _event_segment_indices(timestamps, events, target_hz=190.0)
        stage1_iou = None
        if best_seg is not None and ev_lo is not None and ev_hi is not None:
            stage1_iou = _iou(int(best_seg['start']), int(best_seg['end']), int(ev_lo), int(ev_hi))

        if best_seg is None:
            pipe_pred = ''
            ctc_pred = ''
            pipe = {'num_keys': 0, 'top_candidates': [], 'char_top1': ''}
            ctc = {'prediction': '', 'beam_candidates': []}
            seg_len = 0
        else:
            lo = max(0, int(best_seg['start']))
            hi = min(len(imu190), int(best_seg['end']))
            seg = imu190[lo:hi]
            seg_len = len(seg)
            pipe = run_pipeline_stage23(seg, models, beam_width=args.beam_width)
            ctc = run_ctc(seg, models)
            pipe_pred = str(pipe.get('char_top1', ''))
            ctc_pred = str(ctc.get('prediction', ''))

        row = {
            'session_prefix': prefix,
            'reference': ref,
            'reference_length': len(ref),
            'typed_matches_reference': bool(protocol.get('typed_matches_reference', False)),
            'orig_hz_est': round(float(orig_hz), 3),
            'num_stage1_segments': len(segments),
            'stage1_best_start': None if best_seg is None else int(best_seg['start']),
            'stage1_best_end': None if best_seg is None else int(best_seg['end']),
            'stage1_best_confidence': None if best_seg is None else float(best_seg['confidence']),
            'stage1_segment_len': seg_len,
            'event_lo_190': ev_lo,
            'event_hi_190': ev_hi,
            'stage1_iou_vs_events': stage1_iou,
            'pipeline_prediction': pipe_pred,
            'pipeline_num_keys': int(pipe.get('num_keys', 0)),
            'pipeline_length': len(pipe_pred),
            'pipeline_exact_match': pipe_pred == ref,
            'pipeline_cer': cer(ref, pipe_pred),
            'pipeline_top_candidate': '' if not pipe.get('top_candidates') else pipe['top_candidates'][0]['password'],
            'ctc_prediction': ctc_pred,
            'ctc_length': len(ctc_pred),
            'ctc_exact_match': ctc_pred == ref,
            'ctc_cer': cer(ref, ctc_pred),
            'ctc_top_beam_candidate': '' if not ctc.get('beam_candidates') else ctc['beam_candidates'][0]['password'],
        }
        rows.append(row)

    def _mean(key: str) -> float:
        vals = [float(r[key]) for r in rows]
        return sum(vals) / len(vals) if vals else float('nan')

    summary = {
        'dataset_root': str(dataset_root),
        'checkpoint_dir': str(checkpoint_dir),
        'n_trials': len(rows),
        'pipeline_exact_match_rate': sum(1 for r in rows if r['pipeline_exact_match']) / len(rows),
        'pipeline_mean_cer': _mean('pipeline_cer'),
        'pipeline_length_accuracy': sum(1 for r in rows if r['pipeline_length'] == r['reference_length']) / len(rows),
        'ctc_exact_match_rate': sum(1 for r in rows if r['ctc_exact_match']) / len(rows),
        'ctc_mean_cer': _mean('ctc_cer'),
        'ctc_length_accuracy': sum(1 for r in rows if r['ctc_length'] == r['reference_length']) / len(rows),
        'mean_stage1_iou_vs_events': None if not any(r['stage1_iou_vs_events'] is not None for r in rows) else sum(r['stage1_iou_vs_events'] for r in rows if r['stage1_iou_vs_events'] is not None) / sum(1 for r in rows if r['stage1_iou_vs_events'] is not None),
    }

    (output_dir / 'report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (output_dir / 'rows.json').write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    with (output_dir / 'rows.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote: {output_dir / "report.json"}')
    print(f'Wrote: {output_dir / "rows.csv"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
