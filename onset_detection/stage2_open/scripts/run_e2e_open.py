#!/usr/bin/env python3
"""
Run Open Stage 2 pipeline on mixed2 test data.

python scripts/run_e2e_open.py \
    --mixed2_dir data/onset_mixed2 \
    --checkpoint runs/stage2_open/best.pt \
    --output_dir results/open
"""
import sys, os, json, argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn import OpenTCN
from utils.signal_processing import preprocess
from utils.decoder import decode_frame_labels
from utils.metrics import full_eval, frame_accuracy, format_report
from data.loaders import SessionLoader, discover_sessions
from configs.config import ModelConfig, DecoderConfig, SignalConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt.get('model_cfg')
    if mcfg is None:
        mcfg = ModelConfig()
    elif isinstance(mcfg, dict):
        mcfg = ModelConfig(**mcfg)
    dcfg = ckpt.get('decoder_cfg')
    if dcfg is None:
        dcfg = DecoderConfig()
    elif isinstance(dcfg, dict):
        dcfg = DecoderConfig(**dcfg)

    model = OpenTCN(
        in_ch=mcfg.input_channels, hidden=mcfg.hidden_channels,
        num_layers=mcfg.num_layers, kernel=mcfg.kernel_size,
        dropout=mcfg.dropout, num_classes=mcfg.num_classes,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, mcfg, dcfg


def load_mixed2_gt(session_dir, sr):
    """Load mixed2/mixed_training session and extract GT groups + onsets."""
    loader = SessionLoader(session_dir)
    ts, imu = loader.get_imu()
    attempts = loader.get_attempts()
    presses = loader.get_press_events()

    if attempts:
        # Legacy path: attempts.csv available
        starts = [a['start_ns'] for a in attempts if 'start_ns' in a]
        ends = [a['end_ns'] for a in attempts if 'end_ns' in a]
        if not starts or not ends:
            return None

        pad_ns = int(0.5e9)
        region_s = min(starts) - pad_ns
        region_e = max(ends) + pad_ns

        mask = (ts >= region_s) & (ts <= region_e)
        idx = np.where(mask)[0]
        if len(idx) < 10:
            return None

        region_imu = imu[idx]
        region_ts = ts[idx]

        gt_groups = []
        for att in sorted(attempts, key=lambda a: a.get('start_ns', 0)):
            a_s, a_e = att['start_ns'], att.get('end_ns', att['start_ns'] + int(5e9))
            gs = int(np.searchsorted(region_ts, a_s))
            ge = int(np.searchsorted(region_ts, a_e))
            gs = min(gs, len(region_ts) - 1)
            ge = min(ge, len(region_ts))

            group_onsets = []
            group_chars = []
            for p in presses:
                if a_s <= p['ts'] <= a_e:
                    oi = int(np.searchsorted(region_ts, p['ts']))
                    oi = min(oi, len(region_ts) - 1)
                    group_onsets.append(oi)
                    group_chars.append(p['key'])

            gt_groups.append({
                'start': gs, 'end': ge,
                'onsets': group_onsets,
                'chars': group_chars,
                'num_keys': len(group_onsets),
            })

        return {'imu': region_imu, 'gt_groups': gt_groups}

    # New path: derive groups from password block + Enter separators
    block = loader.get_password_block()
    groups = loader.split_password_groups_from_enters()
    if block is None or block.get('start_ns') is None or block.get('end_ns') is None:
        return None
    if not groups:
        return None

    pad_ns = int(0.5e9)
    region_s = block['start_ns'] - pad_ns
    region_e = block['end_ns'] + pad_ns

    mask = (ts >= region_s) & (ts <= region_e)
    idx = np.where(mask)[0]
    if len(idx) < 10:
        return None

    region_imu = imu[idx]
    region_ts = ts[idx]

    gt_groups = []
    for group in groups:
        a_s, a_e = group['start_ns'], group['end_ns']
        gs = int(np.searchsorted(region_ts, a_s))
        ge = int(np.searchsorted(region_ts, a_e))
        gs = min(gs, len(region_ts) - 1)
        ge = min(ge, len(region_ts))

        group_onsets = [
            min(int(np.searchsorted(region_ts, p['ts'])), len(region_ts) - 1)
            for p in group['keys']
        ]
        group_chars = [p['key'] for p in group['keys']]

        gt_groups.append({
            'start': gs, 'end': ge,
            'onsets': group_onsets,
            'chars': group_chars,
            'num_keys': len(group_onsets),
        })

    return {'imu': region_imu, 'gt_groups': gt_groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mixed2_dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', default='results/open')
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate

    dev = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else args.device if args.device != 'auto' else 'cpu'
    )

    model, mcfg, dcfg = load_model(args.checkpoint, dev)
    scfg = SignalConfig(sample_rate=sr)

    sessions = discover_sessions(args.mixed2_dir)
    if not sessions:
        if (Path(args.mixed2_dir) / 'sensor.csv').exists():
            sessions = [args.mixed2_dir]

    print(f"Found {len(sessions)} sessions\n")

    for i, sd in enumerate(sessions):
        print(f"--- Session {i + 1}: {sd} ---")
        data = load_mixed2_gt(sd, sr)
        if data is None:
            print("  skip (no data)")
            continue

        imu = data['imu']
        gt_groups = data['gt_groups']
        print(f"  Region: {len(imu)} samples ({len(imu)/sr:.1f}s), "
              f"GT groups: {len(gt_groups)}, "
              f"GT keys/group: {[g['num_keys'] for g in gt_groups]}")

        # Preprocess & predict
        proc, _ = preprocess(imu, sr, scfg.use_magnitude, scfg.normalize)
        x = torch.from_numpy(proc.T).float().unsqueeze(0).to(dev)

        with torch.no_grad():
            preds_t, probs_t = model.predict(x)
        preds = preds_t[0].cpu().numpy()

        # Decode
        dec = decode_frame_labels(
            preds, sr,
            min_keystroke_run=dcfg.min_keystroke_run,
            min_separator_run_ms=dcfg.min_separator_run_ms,
            min_gap_between_onsets_ms=dcfg.min_gap_between_onsets_ms,
        )

        print(f"  Pred groups: {dec['num_passwords']}, "
              f"keys/group: {[g['num_keys'] for g in dec['groups']]}, "
              f"total onsets: {dec['total_onsets']}")

        # Evaluate
        ev = full_eval(dec['groups'], gt_groups, sr)
        fa = frame_accuracy(preds, np.zeros_like(preds))  # placeholder if no GT frame labels
        report = format_report(ev, None)
        print(report)

        # Save
        name = Path(sd).name
        with open(out / f"{name}_results.json", 'w') as f:
            json.dump({
                'pred_groups': dec['groups'],
                'gt_groups': [{k: v for k, v in g.items() if k != 'chars'}
                              for g in gt_groups],
                'eval': {k: v for k, v in ev.items() if k != 'group_match'},
                'num_pred': dec['num_passwords'],
                'num_gt': len(gt_groups),
            }, f, indent=2, default=str)

    print("\nDone!")


if __name__ == '__main__':
    main()
