from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from collect_clean_password_eval import run_capture


DEFAULT_FAIR6_PROBES = [
    "b15bp8ws",
    "ijtplv3am8",
    "0xc8pugot",
    "1kfxksa8",
    "kodtpoxk",
]


def _parse_passwords(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        if not value:
            continue
        for chunk in value.split(","):
            s = chunk.strip()
            if s:
                out.append(s)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-record still-password-still probe sessions with ground-truth passwords."
    )
    parser.add_argument("--participant", default="p01")
    parser.add_argument("--dataset-root", default="data/raw/clean_password_probe")
    parser.add_argument("--idle-sec", type=float, default=3.0)
    parser.add_argument(
        "--passwords",
        nargs="*",
        default=None,
        help="Passwords to record. You can pass space-separated values or a single comma-separated string.",
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--force-macimu", action="store_true")
    args = parser.parse_args()

    passwords = _parse_passwords(args.passwords or []) or list(DEFAULT_FAIR6_PROBES)
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = (Path(__file__).resolve().parent.parent / dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    print("\nStill-password-still probe batch")
    print(f"Participant:  {args.participant}")
    print(f"Dataset root: {dataset_root}")
    print(f"Idle sec:     {args.idle_sec:.1f}")
    print("Passwords:")
    for idx, pw in enumerate(passwords, 1):
        print(f"  {idx}. {pw}")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "participant": args.participant,
        "dataset_root": str(dataset_root),
        "idle_sec": float(args.idle_sec),
        "passwords": passwords,
        "sessions": [],
        "note": args.note,
        "purpose": "Protocol-mismatch probe batch for still-password-still evaluation.",
        "eval_only": True,
        "include_in_training": False,
    }

    for idx, pw in enumerate(passwords, 1):
        print(f"\n=== Probe {idx}/{len(passwords)}: {pw} ===")
        paths = run_capture(
            reference_text=pw,
            participant_id=args.participant,
            dataset_root=str(dataset_root),
            idle_sec=args.idle_sec,
            note=args.note,
            trial_index=None,
            force_macimu=args.force_macimu,
        )
        manifest["sessions"].append(
            {
                "reference_text": pw,
                "session_prefix": paths.prefix,
                "sensor_csv": paths.sensor_csv,
                "events_csv": paths.events_csv,
                "attempts_csv": paths.attempts_csv,
                "protocol_json": paths.protocol_json,
                "meta_txt": paths.meta_txt,
            }
        )

    manifest_path = dataset_root / "batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBatch complete. Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
