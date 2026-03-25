#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT, THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_segmental.scripts.train_eval_peak_keyness import (  # noqa: E402
    _build_dataset,
    _evaluate_peak_model,
    _load_mixed_episodes,
    _load_password_attempt_episodes,
    _peak_feature_vector,
    _propose_peaks,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Train and save peak keyness RF model")
    ap.add_argument("--password-dir", action="append", default=[])
    ap.add_argument("--mixed-dir", action="append", default=[])
    ap.add_argument("--exclude-session-id", action="append", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    return ap.parse_args()


def main():
    args = parse_args()
    excluded = {str(x) for x in args.exclude_session_id}
    episodes = []
    for d in args.password_dir:
        episodes.extend(_load_password_attempt_episodes(d))
    for d in args.mixed_dir:
        episodes.extend(_load_mixed_episodes(d))
    if excluded:
        episodes = [ep for ep in episodes if ep["session_id"] not in excluded]
    if not episodes:
        raise RuntimeError("No episodes loaded for peak keyness training.")

    X, y, meta = _build_dataset(episodes)
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=42,
    )
    model.fit(X, y)

    self_eval = _evaluate_peak_model(model, episodes)
    threshold_eval = {}
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        exact_k = 0
        total = 0
        for ep in episodes:
            peaks, sm, _ = _propose_peaks(ep)
            if len(peaks) == 0:
                continue
            feats = np.stack([
                _peak_feature_vector(sm, peaks, i, ep["sample_rate_hz"])
                for i in range(len(peaks))
            ]).astype(np.float32)
            probs = model.predict_proba(feats)[:, 1]
            pred_k = int(np.sum(probs >= threshold))
            true_k = int(len(ep["key_frames"]))
            total += 1
            if pred_k == true_k:
                exact_k += 1
        threshold_eval[f"{threshold:.2f}"] = {
            "episodes": int(total),
            "exact_k": int(exact_k),
            "k_accuracy": float(exact_k / max(total, 1)),
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

    report = {
        "output_model": str(out_path.resolve()),
        "num_episodes": len(episodes),
        "num_peak_candidates": int(len(X)),
        "num_positive": int(np.sum(y == 1)),
        "num_negative": int(np.sum(y == 0)),
        "num_sessions": len({ep["session_id"] for ep in episodes}),
        "excluded_sessions": sorted(excluded),
        "threshold_eval": threshold_eval,
        "episode_metrics": {
            "exact_all_keys": float(np.mean([r["peak_top1"] for r in self_eval])) if self_eval else 0.0,
            "mean_peak_recall": float(np.mean([r["peak_recall"] for r in self_eval])) if self_eval else 0.0,
            "mean_peak_precision": float(np.mean([r["peak_precision"] for r in self_eval])) if self_eval else 0.0,
        },
    }
    with open(out_path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
