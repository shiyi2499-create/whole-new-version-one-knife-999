#!/usr/bin/env python3
"""
Run Stage 2A + 2B on the current mixed2 held-out sessions.

This script evaluates Stage 2 only:
- coarse region is taken from the GT mixed2 password block
- GT groups/onsets are reconstructed from the existing activity log + events
- Stage 3 is intentionally optional and disabled by default
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import PipelineConfig
from data.loaders import discover_sessions, load_mixed2_session
from models.pipeline import Stage2Pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Stage 2 rebuild E2E on mixed2")
    parser.add_argument("--mixed2_dir", type=str, required=True)
    parser.add_argument("--stage2a_ckpt", type=str, required=True)
    parser.add_argument("--stage2b_ckpt", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=190)
    parser.add_argument("--output_dir", type=str, default="results/stage2_rebuild/e2e_mixed2")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pad_ms", type=int, default=500)
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage2 Rebuild E2E On mixed2")
    print("=" * 60)

    config = PipelineConfig()
    config.signal.sample_rate = args.sample_rate
    pipeline = Stage2Pipeline.from_checkpoints(
        stage2a_ckpt=args.stage2a_ckpt,
        stage2b_ckpt=args.stage2b_ckpt,
        config=config,
        device=args.device,
    )

    sessions = discover_sessions(args.mixed2_dir, keyword="mixed2")
    if not sessions:
        print("No mixed2 sessions found.")
        return

    print(f"\nFound {len(sessions)} mixed2 session(s)")
    all_results = []

    for idx, sess_ref in enumerate(sessions, start=1):
        print(f"\n--- Session {idx}/{len(sessions)}: {sess_ref} ---")
        session_data = load_mixed2_session(
            sess_ref,
            target_rate_hz=args.sample_rate,
            pad_ms=args.pad_ms,
        )
        if session_data is None:
            print("  Skipped: failed to derive mixed2 GT.")
            continue

        print(
            f"  Region length: {len(session_data['region_imu'])} samples "
            f"({len(session_data['region_imu']) / args.sample_rate:.2f}s)"
        )
        print(f"  GT groups: {session_data['num_groups']}")
        for g_idx, (gt_ons, gt_chars) in enumerate(
            zip(session_data["gt_onset_positions"], session_data["gt_chars"])
        ):
            print(f"    Group {g_idx}: {len(gt_ons)} onsets, chars={''.join(gt_chars[:8])}")

        result = pipeline.evaluate_on_session(
            coarse_region_imu=session_data["region_imu"],
            gt_group_boundaries=session_data["gt_group_boundaries"],
            gt_onset_positions=session_data["gt_onset_positions"],
            gt_chars=session_data["gt_chars"],
            classifier_fn=None,
            sample_rate=args.sample_rate,
        )
        print(result["report"])
        all_results.append(result)

        session_name = Path(sess_ref).name
        result_path = output_path / f"{session_name}_results.json"
        with open(result_path, "w") as f:
            json.dump(
                {
                    "stage2a_metrics": result["stage2a_metrics"],
                    "stage2b_metrics": result["stage2b_metrics"],
                    "e2e_metrics": result["e2e_metrics"],
                    "pred_groups": result["pipeline_results"]["group_boundaries"],
                    "pred_onsets": [
                        arr.tolist() if hasattr(arr, "tolist") else list(arr)
                        for arr in result["pipeline_results"]["onset_positions"]
                    ],
                },
                f,
                indent=2,
                default=str,
            )
        print(f"  Saved: {result_path}")

    if not all_results:
        print("\nNo session produced valid output.")
        return

    avg_iou = float(np.mean([r["stage2a_metrics"]["mean_iou"] for r in all_results]))
    avg_f1 = float(np.mean([
        np.mean([m["f1"] for m in r["stage2b_metrics"]])
        for r in all_results if r["stage2b_metrics"]
    ]))
    avg_recall = float(np.mean([
        np.mean([m["recall"] for m in r["stage2b_metrics"]])
        for r in all_results if r["stage2b_metrics"]
    ]))

    print("\n" + "=" * 60)
    print("Aggregate")
    print("=" * 60)
    print(f"  Avg Group IoU: {avg_iou:.4f}")
    print(f"  Avg Onset F1: {avg_f1:.4f}")
    print(f"  Avg Onset Recall: {avg_recall:.4f}")

    agg_path = output_path / "aggregate_results.json"
    with open(agg_path, "w") as f:
        json.dump(
            {
                "num_sessions": len(all_results),
                "avg_group_iou": avg_iou,
                "avg_onset_f1": avg_f1,
                "avg_onset_recall": avg_recall,
            },
            f,
            indent=2,
        )
    print(f"  Saved: {agg_path}")


if __name__ == "__main__":
    main()
