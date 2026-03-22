#!/usr/bin/env python3
"""
Build a real open-stage2 dataset from mixed_training / mixed2 style sessions.

Each session contributes one coarse password block sample with frame labels:
  0 = gap
  1 = keystroke
  2 = separator

Group supervision is derived from Enter-separated password groups inside the
password typing block.
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import SessionLoader, discover_sessions


def build_one(session_path: str, sample_rate: int, key_radius_ms: float, pad_s: float):
    loader = SessionLoader(session_path)
    ts, imu = loader.get_imu()
    block = loader.get_password_block()
    groups = loader.split_password_groups_from_enters()

    if len(ts) == 0 or block is None or block.get('start_ns') is None or block.get('end_ns') is None:
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
    labels = np.zeros(len(region_ts), dtype=np.int64)
    key_radius = int(key_radius_ms / 1000.0 * sample_rate)

    group_infos = []
    for gi, group in enumerate(groups):
        local_onsets = []
        chars = []
        for event in group['keys']:
            oi = int(np.searchsorted(region_ts, event['ts']))
            oi = min(max(oi, 0), len(region_ts) - 1)
            local_onsets.append(oi)
            chars.append(event['key'])

            lo = max(0, oi - key_radius)
            hi = min(len(labels), oi + key_radius + 1)
            labels[lo:hi] = 1

        if local_onsets:
            group_infos.append({
                'start': max(0, local_onsets[0] - key_radius),
                'end': min(len(labels), local_onsets[-1] + key_radius + 1),
                'onsets': local_onsets,
                'chars': chars,
                'num_keys': len(local_onsets),
            })

    # Mark separators between consecutive groups
    for prev, nxt in zip(group_infos[:-1], group_infos[1:]):
        sep_start = prev['end']
        sep_end = nxt['start']
        if sep_end > sep_start:
            labels[sep_start:sep_end] = 2

    return {
        'imu': region_imu,
        'frame_labels': labels,
        'groups': group_infos,
        'num_passwords': len(group_infos),
        'password_lengths': [g['num_keys'] for g in group_infos],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True, help='mixed_training or mixed2 root')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--key_radius_ms', type=float, default=30.0)
    ap.add_argument('--pad_s', type=float, default=0.5)
    ap.add_argument('--val_ratio', type=float, default=0.34)
    args = ap.parse_args()

    sessions = discover_sessions(args.input_dir)
    if not sessions:
        print(f"ERROR: no sessions found in {args.input_dir}")
        raise SystemExit(1)

    built = []
    for session_path in sessions:
        sample = build_one(session_path, args.sample_rate, args.key_radius_ms, args.pad_s)
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
    test_dir = out / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

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
            groups_json=json.dumps(sample['groups']),
            num_passwords=sample['num_passwords'],
            password_lengths=np.array(sample['password_lengths']),
            source_session=str(session_path),
        )
        print(f"  {split}: {Path(session_path).name} -> {split_dir.name}/session_{split_idx:04d}.npz")

    meta = {
        'source_dir': args.input_dir,
        'num_sessions': n_total,
        'splits': {
            'train': n_train,
            'val': n_val,
            'test': 0,
        },
        'sample_rate': args.sample_rate,
        'notes': 'Real open dataset built from Enter-separated password groups.',
    }
    with open(out / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved real open dataset to {out}")
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
