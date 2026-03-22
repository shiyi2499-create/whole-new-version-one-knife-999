#!/usr/bin/env python3
"""
Build CTC episode dataset from existing mixed_training / password sessions.

Two data sources:
  1. Real episodes from mixed_training sessions (per-key timestamp + label)
  2. Synthetic episodes from password attempt segments (augmented)

Usage:
    python scripts/build_dataset.py \
        --mixed_training_dir data/raw/mixed_training \
        --password_dir data/raw/password \
        --neg_dir data/raw/onset_negative \
        --output_dir data/stage2_ctc \
        --num_synth 600

This produces data/stage2_ctc/{train,val,test}/*.npz with:
    imu, hard_targets, soft_weights, ctc_target, password
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np

# Package imports
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from data.loaders import SessionLoader, NegativeLoader, discover_sessions
from data.datasets import build_frame_targets
from data.synthesis import CTCSynthesizer, load_all_segments
from utils.vocab import char_index, is_ignored_key
from configs.config import DataConfig


def build_real_episodes(session_path: str, sample_rate: int = 100,
                        margin_ms: float = 300.0, sigma_ms: float = 20.0):
    """
    Extract per-episode CTC training samples from one mixed_training session.

    Uses the per-key timestamps and character labels already recorded.
    """
    loader = SessionLoader(session_path)
    ts, imu = loader.get_imu()
    if len(ts) == 0:
        return []

    groups = loader.split_password_groups_from_enters()
    if not groups:
        return []

    margin_frames = int(margin_ms / 1000.0 * sample_rate)
    sigma_frames = max(1.0, sigma_ms / 1000.0 * sample_rate)
    samples = []

    for group in groups:
        keys = group['keys']
        if not keys:
            continue

        # Get frame indices for each key
        key_events = []
        for event in keys:
            if is_ignored_key(event['key']):
                continue
            frame_idx = int(np.searchsorted(ts, event['ts']))
            frame_idx = min(max(frame_idx, 0), len(ts) - 1)
            key_events.append({
                'ts_frame': frame_idx,
                'char': event['key'],
            })

        if not key_events:
            continue

        # Cut episode region with margin
        ep_start = max(0, key_events[0]['ts_frame'] - margin_frames)
        ep_end = min(len(ts), key_events[-1]['ts_frame'] + margin_frames)

        episode_imu = imu[ep_start:ep_end].astype(np.float32)
        T = len(episode_imu)
        if T < 10:
            continue

        # Adjust frame indices to episode-local coordinates
        local_events = [
            {'ts_frame': e['ts_frame'] - ep_start, 'char': e['char']}
            for e in key_events
        ]

        hard_targets, soft_weights, ctc_target = build_frame_targets(
            T, local_events, sigma_frames
        )

        password = ''.join(e['char'].lower() for e in local_events)

        samples.append({
            'imu': episode_imu,
            'hard_targets': hard_targets,
            'soft_weights': soft_weights,
            'ctc_target': np.array(ctc_target, dtype=np.int64),
            'password': password,
            'source': str(session_path),
        })

    return samples


def main():
    ap = argparse.ArgumentParser(description="Build CTC episode dataset")
    ap.add_argument('--mixed_training_dir', required=True,
                    help='Directory containing mixed_training sessions')
    ap.add_argument('--password_dir', default='',
                    help='Directory containing password sessions (for synth source)')
    ap.add_argument('--neg_dir', default='',
                    help='Directory containing negative samples')
    ap.add_argument('--output_dir', required=True,
                    help='Output dataset directory')
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--margin_ms', type=float, default=300.0)
    ap.add_argument('--sigma_ms', type=float, default=20.0)
    ap.add_argument('--num_synth', type=int, default=600,
                    help='Number of synthetic sessions to generate')
    ap.add_argument('--val_ratio', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    rng = np.random.RandomState(args.seed)

    # ── 1. Build real episodes from mixed_training ──
    print("="*60)
    print("  STEP 1: Extract real episodes from mixed_training")
    print("="*60)

    sessions = discover_sessions(args.mixed_training_dir)
    print(f"Found {len(sessions)} sessions in {args.mixed_training_dir}")

    real_episodes = []
    for sp in sessions:
        eps = build_real_episodes(sp, args.sample_rate, args.margin_ms, args.sigma_ms)
        if eps:
            print(f"  {Path(sp).name}: {len(eps)} episodes, "
                  f"passwords: {[e['password'] for e in eps]}")
            real_episodes.extend(eps)

    print(f"\nTotal real episodes: {len(real_episodes)}")

    # ── 2. Generate synthetic episodes ──
    synth_episodes = []
    if args.password_dir and args.num_synth > 0:
        print(f"\n{'='*60}")
        print(f"  STEP 2: Generate {args.num_synth} synthetic sessions")
        print(f"{'='*60}")

        segments = load_all_segments(args.password_dir)
        if segments:
            neg_loader = NegativeLoader(args.neg_dir) if args.neg_dir else NegativeLoader('')
            dcfg = DataConfig(num_synth_sessions=args.num_synth)
            synth = CTCSynthesizer(segments, neg_loader, dcfg,
                                   sample_rate=args.sample_rate, seed=args.seed)

            for i in range(args.num_synth):
                if (i + 1) % 100 == 0:
                    print(f"  session {i + 1}/{args.num_synth}")
                eps = synth.generate_one_session()
                synth_episodes.extend(eps)

            print(f"  Generated {len(synth_episodes)} synthetic episodes")
        else:
            print("  No password segments found, skipping synthesis")

    # ── 3. Combine and split ──
    print(f"\n{'='*60}")
    print(f"  STEP 3: Combine and split")
    print(f"{'='*60}")

    all_episodes = real_episodes + synth_episodes
    n_total = len(all_episodes)
    print(f"Total episodes: {n_total} (real={len(real_episodes)}, synth={len(synth_episodes)})")

    # Shuffle
    indices = list(range(n_total))
    rng.shuffle(indices)

    # Real episodes go to both train and val; synthetic only to train
    # This ensures validation reflects real data performance
    real_indices = list(range(len(real_episodes)))
    rng.shuffle(real_indices)
    n_real_val = max(1, int(len(real_episodes) * args.val_ratio))
    real_val_idx = set(real_indices[:n_real_val])

    train_eps = []
    val_eps = []

    for i, ep in enumerate(real_episodes):
        if i in real_val_idx:
            val_eps.append(ep)
        else:
            train_eps.append(ep)

    # All synthetic go to train
    train_eps.extend(synth_episodes)
    rng.shuffle(train_eps)

    print(f"Split: train={len(train_eps)}, val={len(val_eps)}")

    # ── 4. Save ──
    def save_split(episodes, split_name):
        d = out / split_name
        d.mkdir(parents=True, exist_ok=True)
        for j, ep in enumerate(episodes):
            np.savez_compressed(
                d / f"episode_{j:05d}.npz",
                imu=ep['imu'],
                hard_targets=ep['hard_targets'],
                soft_weights=ep['soft_weights'],
                ctc_target=ep['ctc_target'],
                password=ep['password'],
            )

    save_split(train_eps, 'train')
    save_split(val_eps, 'val')

    meta = {
        'source_mixed_training': args.mixed_training_dir,
        'source_password': args.password_dir,
        'num_real': len(real_episodes),
        'num_synth': len(synth_episodes),
        'splits': {'train': len(train_eps), 'val': len(val_eps)},
        'sample_rate': args.sample_rate,
        'margin_ms': args.margin_ms,
        'sigma_ms': args.sigma_ms,
        'format': 'ctc_episode',
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved CTC dataset to {out}")
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
