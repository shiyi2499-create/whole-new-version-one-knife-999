from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "demo_inference_api") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "demo_inference_api"))

from inference.pipeline_inference import _load_manifest
from clean_password_eval.eval_still_password_probe_800hz_auto_segmentpad import main as _segmentpad_main


def main() -> int:
    ap = argparse.ArgumentParser(description="Experimental native 800Hz full-auto eval with Stage1 boundary fix D.")
    ap.add_argument("--dataset-root", default="data/raw/800hz/clean_password_probe")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--beam-width", type=int, default=500)
    ap.add_argument("--segment-pad-sec", type=float, default=None)
    args = ap.parse_args()

    manifest = _load_manifest(str((REPO_ROOT / args.checkpoint_dir).resolve() if not Path(args.checkpoint_dir).is_absolute() else Path(args.checkpoint_dir)))
    pad = args.segment_pad_sec
    if pad is None:
        pad = float(manifest.get("stage1_segment_pad_sec", 0.0))

    sys.argv = [
        "eval_still_password_probe_800hz_auto_segmentpad.py",
        "--dataset-root",
        args.dataset_root,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--output-dir",
        args.output_dir,
        "--beam-width",
        str(args.beam_width),
        "--segment-pad-sec",
        str(pad),
    ]
    return _segmentpad_main()


if __name__ == "__main__":
    raise SystemExit(main())
