#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)


def main():
    data_dir = os.path.join(REPO_ROOT, "data_samples")
    out_dir = os.path.join(REPO_ROOT, "runs", "stage2_segmental_smoke")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "stage2_segmental", "scripts", "train_gt_segmental.py"),
        "--input_dir", data_dir,
        "--output_dir", out_dir,
        "--holdout_sessions", "mixed_training_p02_noisy",
        "--classifier_epochs", "40",
        "--segmental_epochs", "30",
        "--device", "cpu",
    ]
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
