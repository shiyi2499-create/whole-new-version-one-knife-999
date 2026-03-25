#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def _load_dense_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("stage1_dense_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_script", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--eval_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--feature_mode", type=str, default="raw6_energy_activity_pulse")
    ap.add_argument("--label_pre_pad_ms", type=int, default=220)
    ap.add_argument("--label_post_pad_ms", type=int, default=380)
    ap.add_argument("--min_password_len", type=int, default=6)
    ap.add_argument("--base_filters", type=int, default=12)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--kernel_size", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_attention", action="store_true")
    ap.add_argument("--sweep_thresholds", nargs="+", type=float, default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    ap.add_argument("--sweep_min_segment_s", nargs="+", type=float, default=[0.5, 0.8])
    ap.add_argument("--sweep_merge_gap_s", nargs="+", type=float, default=[0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
    ap.add_argument("--sweep_prob_smooth_windows", nargs="+", type=int, default=[1, 161])
    ap.add_argument("--sweep_valley_merge_thresholds", nargs="+", type=float, default=[0.0, 0.15, 0.25, 0.3])
    ap.add_argument("--sweep_valley_merge_gap_s", nargs="+", type=float, default=[0.0, 1.5, 2.0, 2.5, 3.0])
    return ap.parse_args()


def main():
    args = parse_args()
    module = _load_dense_module(args.main_script)
    device = module.resolve_device(args.device)
    eval_records = module._build_session_records(
        roots=[args.eval_dir],
        pre_pad_ms=args.label_pre_pad_ms,
        post_pad_ms=args.label_post_pad_ms,
        feature_mode=args.feature_mode,
        min_password_len=args.min_password_len,
    )
    if not eval_records:
        raise RuntimeError(f"No eval records found in {args.eval_dir}")

    in_channels = int(eval_records[0].features.shape[0])
    model = module.UNet1D(
        in_channels=in_channels,
        base_filters=args.base_filters,
        depth=args.depth,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        use_attention=args.use_attention,
    ).to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)

    best_bundle, best_details, _ = module.evaluate_posthoc_grid(
        model=model,
        records=eval_records,
        device=device,
        thresholds=args.sweep_thresholds,
        min_segment_seconds=args.sweep_min_segment_s,
        merge_gap_seconds=args.sweep_merge_gap_s,
        prob_smooth_windows=args.sweep_prob_smooth_windows,
        valley_merge_thresholds=args.sweep_valley_merge_thresholds,
        valley_merge_gap_seconds=args.sweep_valley_merge_gap_s,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "eval_dir": str(args.eval_dir),
        "selection_rule": "2.0*oracle_complete_hit + oracle_mean_iou + 0.30*single_mean_iou",
        **best_bundle,
    }
    (args.output_dir / "best_posthoc_completefirst.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "best_details_completefirst.json").write_text(
        json.dumps(best_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "threshold": best_bundle["threshold"],
                "min_segment_s": best_bundle["min_segment_s"],
                "merge_gap_s": best_bundle["merge_gap_s"],
                "valley_merge_threshold": best_bundle["valley_merge_threshold"],
                "valley_merge_gap_s": best_bundle["valley_merge_gap_s"],
                "single_mean_iou": best_bundle["report"]["single_session_top1"]["mean_iou"],
                "oracle_mean_iou": best_bundle["report"]["all_gt_oracle"]["mean_best_iou"],
                "oracle_complete_hit": best_bundle["report"]["all_gt_oracle"]["complete_hit_rate"],
                "mean_pred_segments": best_bundle["report"]["mean_pred_segments"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
