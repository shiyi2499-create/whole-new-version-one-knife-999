#!/usr/bin/env python3
"""
Convert real mixed_training / mixed2-style sessions into the same .npz format
used by stage2_rebuild synthetic datasets.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import discover_sessions, load_mixed2_session


def _pad_onsets(onset_lists, pad_value=-1):
    if not onset_lists:
        return np.full((1, 1), pad_value, dtype=np.int64)
    max_len = max(len(lst) for lst in onset_lists)
    arr = np.full((len(onset_lists), max(max_len, 1)), pad_value, dtype=np.int64)
    for i, lst in enumerate(onset_lists):
        for j, v in enumerate(lst):
            arr[i, j] = int(v)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Build real mixed dataset in stage2_rebuild NPZ format")
    parser.add_argument("--input_dir", required=True, help="Directory containing mixed_training sessions")
    parser.add_argument("--output_dir", required=True, help="Output dataset dir with train/val/test subdirs")
    parser.add_argument("--sample_rate", type=int, default=190)
    parser.add_argument("--keyword", default="mixed_training")
    parser.add_argument("--expected_groups", type=int, default=5)
    parser.add_argument("--expected_len", type=int, default=8)
    parser.add_argument("--drop_nonexact", action="store_true",
                        help="Drop sessions unless all groups have expected_len onsets")
    args = parser.parse_args()

    sessions = discover_sessions(args.input_dir, keyword=args.keyword)
    out_root = Path(args.output_dir)
    for split in ("train", "val", "test"):
        (out_root / split).mkdir(parents=True, exist_ok=True)

    kept = []
    dropped = []
    for sess in sessions:
        obj = load_mixed2_session(sess, target_rate_hz=args.sample_rate)
        if obj is None:
            dropped.append({"session": sess, "reason": "loader_failed"})
            continue
        group_lengths = [len(x) for x in obj["gt_onset_positions"]]
        if len(group_lengths) != args.expected_groups:
            dropped.append({"session": sess, "reason": f"group_count={len(group_lengths)}"})
            continue
        if args.drop_nonexact and any(g != args.expected_len for g in group_lengths):
            dropped.append({"session": sess, "reason": f"group_lengths={group_lengths}"})
            continue

        imu = obj["region_imu"].astype(np.float32)
        boundaries = np.asarray(obj["gt_group_boundaries"], dtype=np.int64)
        group_labels = np.zeros(len(imu), dtype=np.float32)
        for start, end in boundaries:
            group_labels[int(start):int(end)] = 1.0

        kept.append({
            "session": sess,
            "imu": imu,
            "group_labels": group_labels,
            "group_boundaries": boundaries,
            "onset_positions": _pad_onsets(obj["gt_onset_positions"]),
            "onset_chars": json.dumps(obj["gt_chars"]),
            "num_groups": len(obj["gt_group_boundaries"]),
            "keys_per_group": args.expected_len,
            "group_lengths": group_lengths,
        })

    if not kept:
        raise SystemExit("No valid sessions kept.")

    # Very small-data split: keep last item for val, rest for train, no test by default.
    train_items = kept[:-1] if len(kept) > 1 else kept
    val_items = kept[-1:] if len(kept) > 1 else kept
    test_items = []

    for split, items in (("train", train_items), ("val", val_items), ("test", test_items)):
        for idx, item in enumerate(items):
            out_path = out_root / split / f"session_{idx:04d}.npz"
            np.savez_compressed(
                out_path,
                imu=item["imu"],
                group_labels=item["group_labels"],
                group_boundaries=item["group_boundaries"],
                onset_positions=item["onset_positions"],
                onset_chars=item["onset_chars"],
                num_groups=item["num_groups"],
                keys_per_group=item["keys_per_group"],
            )

    meta = {
        "input_dir": args.input_dir,
        "sample_rate": args.sample_rate,
        "keyword": args.keyword,
        "drop_nonexact": args.drop_nonexact,
        "kept_sessions": [
            {
                "session": item["session"],
                "group_lengths": item["group_lengths"],
            }
            for item in kept
        ],
        "dropped_sessions": dropped,
        "splits": {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        },
    }
    (out_root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
