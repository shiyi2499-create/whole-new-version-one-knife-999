#!/usr/bin/env python3
"""
Run Episode-based Stage 2 pipeline on mixed2/mixed_training test data.

Usage:
    python scripts/run_e2e_episode.py \
        --mixed2_dir data/raw/onset_mixed2 \
        --checkpoint runs/stage2_episode/best.pt \
        --output_dir results/episode

    # Sweep episode_gap_ms to find best threshold:
    python scripts/run_e2e_episode.py \
        --mixed2_dir data/raw/onset_mixed2 \
        --checkpoint runs/stage2_episode/best.pt \
        --sweep_gap 300,400,500,600,700,800,1000
"""
import sys, os, json, argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tcn import EpisodeTCN
from utils.signal_processing import preprocess
from utils.decoder import decode_episodes, episodes_to_groups
from utils.metrics import full_eval, frame_accuracy_2class, format_report
from utils.onset_refine import (
    load_stage2b_refiner,
    detect_onsets_with_refiner,
    refine_onsets_with_guidance,
)
from data.loaders import SessionLoader, discover_sessions
from configs.config import ModelConfig, EpisodeConfig, SignalConfig


def load_model(ckpt_path, device):
    from models.tcn import load_episode_tcn
    model, mcfg = load_episode_tcn(ckpt_path, device, use_onset_head=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ecfg = ckpt.get('episode_cfg')
    if ecfg is None:
        ecfg = EpisodeConfig()
    elif isinstance(ecfg, dict):
        ecfg = EpisodeConfig(**ecfg)
    return model, mcfg, ecfg


def load_gt(session_dir, sr):
    """Load session and extract GT episodes + onsets from password block."""
    loader = SessionLoader(session_dir)
    ts, imu = loader.get_imu()

    # Try attempts.csv first (legacy)
    attempts = loader.get_attempts()
    presses = loader.get_press_events()

    if attempts:
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

        gt_episodes = []
        for att in sorted(attempts, key=lambda a: a.get('start_ns', 0)):
            a_s = att['start_ns']
            a_e = att.get('end_ns', a_s + int(5e9))
            gs = min(int(np.searchsorted(region_ts, a_s)), len(region_ts) - 1)
            ge = min(int(np.searchsorted(region_ts, a_e)), len(region_ts))

            ep_onsets = []
            ep_chars = []
            for p in presses:
                if a_s <= p['ts'] <= a_e:
                    oi = min(int(np.searchsorted(region_ts, p['ts'])),
                             len(region_ts) - 1)
                    ep_onsets.append(oi)
                    ep_chars.append(p['key'])

            gt_episodes.append({
                'start': gs, 'end': ge,
                'onsets': ep_onsets, 'chars': ep_chars,
                'num_keys': len(ep_onsets),
            })

        return {'imu': region_imu, 'ts': region_ts, 'gt_episodes': gt_episodes}

    # New path: Enter-separated groups
    block = loader.get_password_block()
    groups = loader.split_password_groups_from_enters()
    if block is None or block.get('start_ns') is None or not groups:
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

    gt_episodes = []
    for group in groups:
        a_s, a_e = group['start_ns'], group['end_ns']
        gs = min(int(np.searchsorted(region_ts, a_s)), len(region_ts) - 1)
        ge = min(int(np.searchsorted(region_ts, a_e)), len(region_ts))

        ep_onsets = [
            min(int(np.searchsorted(region_ts, p['ts'])), len(region_ts) - 1)
            for p in group['keys']
        ]
        ep_chars = [p['key'] for p in group['keys']]

        gt_episodes.append({
            'start': gs, 'end': ge,
            'onsets': ep_onsets, 'chars': ep_chars,
            'num_keys': len(ep_onsets),
        })

    return {'imu': region_imu, 'ts': region_ts, 'gt_episodes': gt_episodes}


def run_one_session(model, data, sr, scfg, ecfg, device, onset_aux=None,
                    typing_threshold=None):
    """Run model on one session and return decoded episodes + eval."""
    imu = data['imu']
    gt_episodes = data['gt_episodes']

    proc, _ = preprocess(imu, sr, scfg.use_magnitude, scfg.normalize)
    x = torch.from_numpy(proc.T).float().unsqueeze(0).to(device)

    with torch.no_grad():
        typing_logits, onset_logits = model(x)
        typing_probs = torch.softmax(typing_logits, dim=1)[0, 1].cpu().numpy()
        if typing_threshold is None:
            preds = typing_logits.argmax(dim=1)[0].cpu().numpy()
        else:
            preds = (typing_probs >= float(typing_threshold)).astype(np.int64)

        # Extract onset_probs from dual-head model if available
        onset_probs = None
        if onset_logits is not None:
            onset_probs = torch.sigmoid(onset_logits)[0, 0].cpu().numpy()

    dec = decode_episodes(
        preds,
        raw_imu=imu,
        typing_probs=typing_probs,
        onset_probs=onset_probs,
        sample_rate=sr,
        median_kernel=ecfg.median_kernel,
        min_typing_run_ms=ecfg.min_typing_run_ms,
        episode_gap_ms=ecfg.episode_gap_ms,
        min_onset_gap_ms=ecfg.min_onset_gap_ms,
        min_episode_keys=ecfg.min_episode_keys,
        min_episode_duration_ms=ecfg.min_episode_duration_ms,
    )

    # onset_aux (stage2b refiner) is now an optional secondary refinement.
    # With the dual-head model producing onset_probs, onset_aux has less value
    # but is kept for backward compatibility / ablation experiments.
    if onset_aux is not None and dec['episodes']:
        onset_model, onset_cfg = onset_aux
        for ep in dec['episodes']:
            anchor_onsets = np.array(
                [o - ep['start'] for o in ep.get('onsets', [])],
                dtype=np.int64,
            )
            refined = refine_onsets_with_guidance(
                onset_model,
                onset_cfg,
                imu[ep['start']:ep['end']],
                sample_rate=sr,
                anchor_onsets=anchor_onsets,
            )
            if len(refined) > 0:
                ep['onsets'] = (refined + ep['start']).tolist()
                ep['num_keys'] = len(refined)
        dec['total_onsets'] = int(sum(ep['num_keys'] for ep in dec['episodes']))

    ev = full_eval(dec['episodes'], gt_episodes, sr)

    return dec, ev, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mixed2_dir', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', default='results/episode')
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--stage2b-ckpt', default=None,
                    help='Optional stage2_rebuild Stage2B checkpoint for per-episode onset refinement')
    ap.add_argument('--episode_gap_ms', type=float, default=None,
                    help='Override episode_gap_ms from checkpoint')
    ap.add_argument('--typing-threshold', type=float, default=None,
                    help='Use explicit typing probability threshold instead of argmax')
    ap.add_argument('--sweep_gap', default=None,
                    help='Comma-separated list of gap values to sweep')
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate

    dev = torch.device(
        'cuda' if args.device == 'auto' and torch.cuda.is_available()
        else args.device if args.device != 'auto' else 'cpu'
    )

    model, mcfg, ecfg = load_model(args.checkpoint, dev)
    scfg = SignalConfig(sample_rate=sr)
    onset_aux = None
    if args.stage2b_ckpt:
        onset_aux = load_stage2b_refiner(args.stage2b_ckpt, dev)

    if args.episode_gap_ms is not None:
        ecfg.episode_gap_ms = args.episode_gap_ms

    # Load all sessions
    sessions = discover_sessions(args.mixed2_dir)
    if not sessions:
        if (Path(args.mixed2_dir) / 'sensor.csv').exists():
            sessions = [args.mixed2_dir]

    print(f"Found {len(sessions)} sessions\n")

    all_data = []
    for sd in sessions:
        data = load_gt(sd, sr)
        if data is not None:
            all_data.append((sd, data))

    print(f"Loaded {len(all_data)} valid sessions\n")

    if args.sweep_gap:
        # Gap threshold sweep mode
        gap_values = [float(v) for v in args.sweep_gap.split(',')]
        print(f"Sweeping episode_gap_ms: {gap_values}\n")

        sweep_results = []
        for gap in gap_values:
            ecfg.episode_gap_ms = gap
            all_det = []
            all_f1 = []
            all_count_err = []

            for sd, data in all_data:
                dec, ev, _ = run_one_session(
                    model, data, sr, scfg, ecfg, dev,
                    onset_aux=onset_aux,
                    typing_threshold=args.typing_threshold,
                )
                all_det.append(ev['episode_detection_rate'])
                all_f1.append(ev['avg_onset_f1'])
                n_pred = ev['episode_match']['pred_count']
                n_gt = ev['episode_match']['gt_count']
                all_count_err.append(abs(n_pred - n_gt))

            avg_det = float(np.mean(all_det))
            avg_f1 = float(np.mean(all_f1))
            avg_count_err = float(np.mean(all_count_err))

            result = {
                'gap_ms': gap,
                'avg_detection_rate': avg_det,
                'avg_onset_f1': avg_f1,
                'avg_count_error': avg_count_err,
            }
            sweep_results.append(result)
            print(f"  gap={gap:6.0f}ms  det={avg_det:.3f}  "
                  f"onset_f1={avg_f1:.3f}  count_err={avg_count_err:.2f}")

        # Find best
        best = max(sweep_results, key=lambda r: r['avg_detection_rate'])
        print(f"\nBest gap: {best['gap_ms']}ms "
              f"(det={best['avg_detection_rate']:.3f})")

        with open(out / 'sweep_results.json', 'w') as f:
            json.dump(sweep_results, f, indent=2)

        return

    # Normal evaluation mode
    for i, (sd, data) in enumerate(all_data):
        print(f"--- Session {i + 1}: {sd} ---")
        gt_episodes = data['gt_episodes']
        print(f"  Region: {len(data['imu'])} samples ({len(data['imu'])/sr:.1f}s), "
              f"GT episodes: {len(gt_episodes)}, "
              f"GT keys/ep: {[ep['num_keys'] for ep in gt_episodes]}")

        dec, ev, preds = run_one_session(
            model, data, sr, scfg, ecfg, dev,
            onset_aux=onset_aux,
            typing_threshold=args.typing_threshold,
        )

        print(f"  Pred episodes: {dec['num_episodes']}, "
              f"keys/ep: {[ep['num_keys'] for ep in dec['episodes']]}, "
              f"total onsets: {dec['total_onsets']}")

        report = format_report(ev)
        print(report)

        # Save
        name = Path(sd).name
        with open(out / f"{name}_results.json", 'w') as f:
            json.dump({
                'pred_episodes': [{k: v for k, v in ep.items()
                                   if k != 'duration_ms' or True}
                                  for ep in dec['episodes']],
                'gt_episodes': [{k: v for k, v in ep.items() if k != 'chars'}
                                for ep in gt_episodes],
                'eval': {k: v for k, v in ev.items()
                         if k != 'episode_match'},
                'num_pred': dec['num_episodes'],
                'num_gt': len(gt_episodes),
                'episode_gap_ms': ecfg.episode_gap_ms,
                'typing_threshold': args.typing_threshold,
            }, f, indent=2, default=str)

    print("\nDone!")


if __name__ == '__main__':
    main()
