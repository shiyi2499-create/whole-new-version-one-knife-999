#!/usr/bin/env python3
"""
Scan session sampling rates and flag sessions that are not near target Hz.

Examples:
  python3 scan_sampling_rates.py
  python3 scan_sampling_rates.py --mode single_key --target-hz 199 --tol 5
  python3 scan_sampling_rates.py --mode all --json-out results/rate_scan.json
"""

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import asdict, dataclass

import numpy as np


SESSION_RE = re.compile(
    r"(?P<participant>p\d+)_(?P<mode>single_key|free_type)"
    r"(?:_(?P<tag>g\d+|part\d+))?_(?P<ts>\d{8}_\d{6})$"
)


@dataclass
class RateRow:
    source_dir: str
    session: str
    participant: str
    mode: str
    tag: str
    target_metric: str
    target_hz: float
    tol_hz: float
    effective_hz: float
    median_hz: float
    p10_hz: float
    p90_hz: float
    min_hz: float
    max_hz: float
    n_samples: int
    within_target: bool


def compute_rate(sensor_csv: str) -> dict:
    ts = []
    with open(sensor_csv, "r") as f:
        for row in csv.DictReader(f):
            ts.append(int(row["timestamp_ns"]))
    ts = np.array(ts, dtype=np.int64)
    if len(ts) < 2:
        return {
            "effective_hz": 0.0,
            "median_hz": 0.0,
            "p10_hz": 0.0,
            "p90_hz": 0.0,
            "min_hz": 0.0,
            "max_hz": 0.0,
            "n_samples": int(len(ts)),
        }

    dt = np.diff(ts)
    pos = dt[dt > 0]
    hz = 1e9 / pos if len(pos) else np.array([], dtype=np.float64)

    effective_hz = (len(ts) - 1) / max((ts[-1] - ts[0]) / 1e9, 1e-9)
    return {
        "effective_hz": float(effective_hz),
        "median_hz": float(np.median(hz)) if len(hz) else 0.0,
        "p10_hz": float(np.percentile(hz, 10)) if len(hz) else 0.0,
        "p90_hz": float(np.percentile(hz, 90)) if len(hz) else 0.0,
        "min_hz": float(np.min(hz)) if len(hz) else 0.0,
        "max_hz": float(np.max(hz)) if len(hz) else 0.0,
        "n_samples": int(len(ts)),
    }


def collect_rows(
    data_root: str,
    mode: str,
    target_hz: float,
    tol: float,
    metric: str,
    sources: list[str],
) -> list[RateRow]:
    rows: list[RateRow] = []
    for source_dir in sources:
        for sensor in sorted(glob.glob(os.path.join(data_root, source_dir, "*_sensor.csv"))):
            session = os.path.basename(sensor).replace("_sensor.csv", "")
            m = SESSION_RE.match(session)
            if not m:
                continue
            s_mode = m.group("mode")
            if mode != "all" and s_mode != mode:
                continue
            rate = compute_rate(sensor)
            value = rate["median_hz"] if metric == "median" else rate["effective_hz"]
            within = abs(value - target_hz) <= tol
            rows.append(
                RateRow(
                    source_dir=source_dir,
                    session=session,
                    participant=m.group("participant"),
                    mode=s_mode,
                    tag=m.group("tag") or "",
                    target_metric=metric,
                    target_hz=target_hz,
                    tol_hz=tol,
                    effective_hz=rate["effective_hz"],
                    median_hz=rate["median_hz"],
                    p10_hz=rate["p10_hz"],
                    p90_hz=rate["p90_hz"],
                    min_hz=rate["min_hz"],
                    max_hz=rate["max_hz"],
                    n_samples=rate["n_samples"],
                    within_target=within,
                )
            )
    return rows


def print_report(rows: list[RateRow], metric: str):
    print(f"scanned sessions: {len(rows)}")
    if not rows:
        return

    print("\nall sessions:")
    for r in rows:
        mark = "OK" if r.within_target else "OUT"
        print(
            f"  {r.source_dir:<12s} {r.session:<42s}  "
            f"eff={r.effective_hz:7.2f}  med={r.median_hz:7.2f}  "
            f"p10={r.p10_hz:7.2f}  p90={r.p90_hz:7.2f}  {mark}"
        )

    out_rows = [r for r in rows if not r.within_target]
    print(f"\nnon-target sessions by {metric}_hz ({len(out_rows)}):")
    for r in out_rows:
        value = r.median_hz if metric == "median" else r.effective_hz
        print(
            f"  {r.source_dir:<12s} {r.session}  {metric}={value:.2f}Hz  "
            f"(target={r.target_hz:.1f}±{r.tol_hz:.1f})"
        )

    if any(r.mode == "single_key" for r in out_rows):
        print("\nnon-target single_key groups:")
        for r in sorted([x for x in out_rows if x.mode == "single_key"], key=lambda x: (x.source_dir, x.tag, x.session)):
            print(f"  {r.source_dir:<12s} {r.tag or '-':>5s}  {r.session}")


def main():
    parser = argparse.ArgumentParser(description="Scan session sampling rates and flag non-target sessions")
    parser.add_argument("--data-root", default="data/raw", help="raw data root (default: data/raw)")
    parser.add_argument(
        "--mode", choices=["single_key", "free_type", "all"], default="single_key",
        help="session type to scan (default: single_key)"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Raw data source folders under data-root. "
             "Default: existing canonical dirs among single_key/free_type/boost; "
             "fallback to all first-level dirs.",
    )
    parser.add_argument("--target-hz", type=float, default=199.0, help="target sampling frequency (default: 199)")
    parser.add_argument("--tol", type=float, default=5.0, help="allowed deviation from target (default: 5)")
    parser.add_argument(
        "--metric", choices=["median", "effective"], default="median",
        help="metric used for target check (default: median)"
    )
    parser.add_argument("--json-out", default=None, help="optional path to save JSON report")
    args = parser.parse_args()

    if args.sources:
        sources = args.sources
    else:
        canonical = ["single_key", "free_type", "boost"]
        existing = [s for s in canonical if os.path.isdir(os.path.join(args.data_root, s))]
        if existing:
            sources = existing
        else:
            sources = sorted([
                d for d in os.listdir(args.data_root)
                if os.path.isdir(os.path.join(args.data_root, d)) and not d.startswith(".")
            ])

    rows = collect_rows(args.data_root, args.mode, args.target_hz, args.tol, args.metric, sources)
    print_report(rows, args.metric)

    if args.json_out:
        payload = {
            "config": {
                "data_root": args.data_root,
                "mode": args.mode,
                "target_hz": args.target_hz,
                "tol": args.tol,
                "metric": args.metric,
                "sources": sources,
            },
            "rows": [asdict(r) for r in rows],
            "non_target_sessions": [asdict(r) for r in rows if not r.within_target],
        }
        out_dir = os.path.dirname(args.json_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\njson report saved: {args.json_out}")


if __name__ == "__main__":
    main()
