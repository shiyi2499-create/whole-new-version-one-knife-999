#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_password_closure_inception import (
    WindowConfig,
    build_no_space_sequences,
    discover_freetype_sessions,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rebuild Stage3 merged npz with custom window width.")
    ap.add_argument(
        "--free-type-dirs",
        nargs="+",
        required=True,
        help="Directories containing password/free_type raw sessions.",
    )
    ap.add_argument("--pre-ms", type=float, default=100.0)
    ap.add_argument("--post-ms", type=float, default=200.0)
    ap.add_argument("--output", required=True, help="Output .npz path")
    ap.add_argument("--yes-only", action="store_true", default=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    window_cfg = WindowConfig(
        pre_trigger_ms=int(round(args.pre_ms)),
        post_trigger_ms=int(round(args.post_ms)),
        min_window_samples=2,
    )
    sessions = discover_freetype_sessions(args.free_type_dirs)
    if not sessions:
        raise SystemExit("No password/free_type sessions found.")

    x_all = []
    y_all = []
    for sess in sessions:
        seqs = build_no_space_sequences(
            sess,
            yes_only=args.yes_only,
            eval_max_sequences=0,
            window_cfg=window_cfg,
        )
        for seq in seqs:
            for item in seq["items"]:
                x_all.append(item["window"])
                y_all.append(item["key"])

    if not x_all:
        raise SystemExit("No Stage3 windows extracted.")

    X = np.stack(x_all).astype(np.float32)
    y = np.asarray(y_all, dtype="U1")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, X=X, y=y)

    print(
        {
            "output": str(out_path.resolve()),
            "shape": list(X.shape),
            "n_unique_chars": int(len(set(y_all))),
            "pre_ms": float(args.pre_ms),
            "post_ms": float(args.post_ms),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
