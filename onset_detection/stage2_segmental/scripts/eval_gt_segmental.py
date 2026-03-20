#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
for p in (PROJECT_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_segmental.data import build_password_episodes
from onset_detection.stage2_segmental.model import load_segmental_checkpoint
from onset_detection.stage2_segmental.scripts.train_gt_segmental import (
    evaluate_fixed_window,
    evaluate_segmental,
    resolve_device,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate GT episode segmental model")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_json", default="")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    return ap.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    episodes = build_password_episodes(args.input_dir)
    model = load_segmental_checkpoint(args.checkpoint, device)
    baseline_metrics, baseline_results = evaluate_fixed_window(model.classifier, episodes)
    segmental_metrics, segmental_results, segmental_debug = evaluate_segmental(model, episodes, device)
    payload = {
        "baseline_fixed_window": baseline_metrics,
        "segmental": segmental_metrics,
        "delta_top1": segmental_metrics["char_top1"] - baseline_metrics["char_top1"],
        "delta_top5": segmental_metrics["char_top5"] - baseline_metrics["char_top5"],
        "delta_cer": segmental_metrics["cer"] - baseline_metrics["cer"],
        "num_episodes": len(episodes),
    }
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **payload,
                    "baseline_results": baseline_results,
                    "segmental_results": segmental_results,
                    "segmental_debug": segmental_debug,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
