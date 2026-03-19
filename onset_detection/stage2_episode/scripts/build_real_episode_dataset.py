#!/usr/bin/env python3
"""
Build a real episode-based dataset from mixed_training / mixed2 sessions.

Each session contributes one password block sample with:
  frame_labels:   [T] int64   - 0=silence, 1=typing (whole episode interior)
  onset_targets:  [T] float32 - Gaussian impulse at each key onset (NEW)
                  Supervision signal for the dual-head onset_head.
                  sigma = onset_gaussian_sigma_ms (default 20ms @100Hz = 2 frames)

Episode (group) supervision is derived from Enter-separated password groups.

Usage:
    python scripts/build_real_episode_dataset.py \
        --input_dir data/raw/mixed_training \
        --output_dir data/stage2_episode_real \
        --episode_margin_ms 40
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import SessionLoader, discover_sessions


def _make_onset_targets(T: int, onsets: list, sigma_frames: float) -> np.ndarray:
    """Build a Gaussian-smoothed impulse array for a list of onset frame indices."""
    targets = np.zeros(T, dtype=np.float32)
    if not onsets or sigma_frames < 1e-6:
        return targets
    t = np.arange(T, dtype=np.float32)
    for oi in onsets:
        targets += np.exp(-0.5 * ((t - oi) / sigma_frames) ** 2)
    return np.clip(targets, 0.0, 1.0).astype(np.float32)


def build_one(session_path: str, sample_rate: int, episode_margin_ms: float,
              pad_s: float, onset_gaussian_sigma_ms: float = 20.0):
    loader = SessionLoader(session_path)
    ts, imu = loader.get_imu()
    block = loader.get_password_block()
    groups = loader.split_password_groups_from_enters()

    if len(ts) == 0 or block is None:
        return None
    if block.get('start_ns') is None or block.get('end_ns') is None:
        return None
    if not groups:
        return None

    pad_ns = int(pad_s * 1e9)
    region_s = block['start_ns'] - pad_ns
    region_e = block['end_ns'] + pad_ns

    mask = (ts >= region_s) & (ts <= region_e)
    idx = np.where(mask)[0]
    if len(idx) < 10:
        return None

    region_ts = ts[idx]
    region_imu = imu[idx].astype(np.float32)
    T = len(region_ts)

    # 2-class frame labels: 0=silence, 1=typing
    labels = np.zeros(T, dtype=np.int64)
    episode_margin = int(episode_margin_ms / 1000.0 * sample_rate)
    sigma_frames = max(1.0, onset_gaussian_sigma_ms / 1000.0 * sample_rate)

    episode_infos = []
    all_onsets_global = []

    for gi, group in enumerate(groups):
        local_onsets = []
        chars = []
        for event in group['keys']:
            oi = int(np.searchsorted(region_ts, event['ts']))
            oi = min(max(oi, 0), T - 1)
            local_onsets.append(oi)
            chars.append(event['key'])

        if local_onsets:
            ep_start = max(0, local_onsets[0] - episode_margin)
            ep_end = min(T, local_onsets[-1] + episode_margin + 1)
            labels[ep_start:ep_end] = 1
            all_onsets_global.extend(local_onsets)

            episode_infos.append({
                'start': ep_start,
                'end': ep_end,
                'onsets': local_onsets,
                'chars': chars,
                'num_keys': len(local_onsets),
            })

    # Build Gaussian onset targets over the full region
    onset_targets = _make_onset_targets(T, all_onsets_global, sigma_frames)

    return {
        'imu': region_imu,
        'frame_labels': labels,
        'onset_targets': onset_targets,
        'episodes': episode_infos,
        'num_passwords': len(episode_infos),
        'password_lengths': [ep['num_keys'] for ep in episode_infos],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True,
                    help='mixed_training or mixed2 root')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--episode_margin_ms', type=float, default=40.0)
    ap.add_argument('--pad_s', type=float, default=0.5)
    ap.add_argument('--val_ratio', type=float, default=0.34)
    ap.add_argument('--onset_gaussian_sigma_ms', type=float, default=12.0,
                    help='Sigma (ms) for Gaussian onset targets. '
                         'At 100Hz: 12ms~1.2 frames, 20ms=2 frames.')
    args = ap.parse_args()

    sessions = discover_sessions(args.input_dir)
    if not sessions:
        print(f"ERROR: no sessions found in {args.input_dir}")
        raise SystemExit(1)

    built = []
    for session_path in sessions:
        sample = build_one(session_path, args.sample_rate,
                           args.episode_margin_ms, args.pad_s,
                           args.onset_gaussian_sigma_ms)
        if sample is None:
            print(f"  skip: {session_path}")
            continue
        built.append((session_path, sample))

    if not built:
        print("ERROR: no valid samples built")
        raise SystemExit(1)

    out = Path(args.output_dir)
    train_dir = out / 'train'
    val_dir = out / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(built)
    n_val = max(1, int(round(n_total * args.val_ratio))) if n_total > 1 else 0
    n_train = n_total - n_val

    for i, (session_path, sample) in enumerate(built):
        if i < n_train:
            split_dir = train_dir
            split = 'train'
            split_idx = i
        else:
            split_dir = val_dir
            split = 'val'
            split_idx = i - n_train

        np.savez_compressed(
            split_dir / f"session_{split_idx:04d}.npz",
            imu=sample['imu'],
            frame_labels=sample['frame_labels'],
            onset_targets=sample['onset_targets'],
            episodes_json=json.dumps(sample['episodes']),
            num_passwords=sample['num_passwords'],
            password_lengths=np.array(sample['password_lengths']),
            source_session=str(session_path),
        )
        print(f"  {split}: {Path(session_path).name} -> "
              f"{split_dir.name}/session_{split_idx:04d}.npz  "
              f"({sample['num_passwords']} episodes, "
              f"keys={sample['password_lengths']})")

    meta = {
        'source_dir': args.input_dir,
        'num_sessions': n_total,
        'splits': {'train': n_train, 'val': n_val},
        'sample_rate': args.sample_rate,
        'label_scheme': '2-class: 0=silence, 1=typing + onset_targets',
        'episode_margin_ms': args.episode_margin_ms,
        'onset_gaussian_sigma_ms': args.onset_gaussian_sigma_ms,
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved episode dataset to {out}")
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
