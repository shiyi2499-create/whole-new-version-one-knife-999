from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from inference.preprocess import csv_to_array, extract_timestamps, estimate_sample_rate, resample_to_190hz
from inference.pipeline_inference import load_all_models, run_stage1, run_pipeline_stage23, run_ctc


def main() -> int:
    parser = argparse.ArgumentParser(description='Smoke-test the demo inference API on one existing sensor CSV.')
    parser.add_argument('--checkpoint-dir', default=str(REPO_ROOT))
    parser.add_argument('--sensor-csv', required=True)
    parser.add_argument('--beam-width', type=int, default=20)
    args = parser.parse_args()

    csv_path = Path(args.sensor_csv)
    text = csv_path.read_text(encoding='utf-8')
    timestamps = extract_timestamps(text)
    imu = csv_to_array(text)
    sr = estimate_sample_rate(imu, timestamps)
    imu190 = resample_to_190hz(imu, sr)

    models = load_all_models(args.checkpoint_dir)
    segs = run_stage1(imu190, models)
    best = None if not segs else max(segs, key=lambda s: (float(s.get('confidence', 0.0)), int(s['end']) - int(s['start'])))
    out = {
        'sensor_csv': str(csv_path),
        'rows': int(len(imu)),
        'orig_hz_est': float(sr),
        'resampled_rows': int(len(imu190)),
        'num_stage1_segments': int(len(segs)),
        'best_segment': best,
    }
    if best is not None:
        seg = imu190[int(best['start']):int(best['end'])]
        pipe = run_pipeline_stage23(seg, models, beam_width=args.beam_width)
        ctc = run_ctc(seg, models)
        out.update({
            'pipeline_char_top1': pipe.get('char_top1'),
            'pipeline_num_keys': pipe.get('num_keys'),
            'pipeline_top_candidate': None if not pipe.get('top_candidates') else pipe['top_candidates'][0],
            'ctc_prediction': ctc.get('prediction'),
            'ctc_top_beam_candidate': None if not ctc.get('beam_candidates') else ctc['beam_candidates'][0],
        })
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
