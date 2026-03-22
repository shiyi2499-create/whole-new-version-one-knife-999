#!/usr/bin/env python3
"""
Create an interpretable debug bundle for one mixed2 session.

Outputs:
- summary.json
- full_timeline.svg
- group_0.svg ... group_4.svg

No matplotlib dependency is required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import PipelineConfig
from data.loaders import discover_sessions, load_mixed2_session
from models.pipeline import Stage2Pipeline


def _svg_polyline(xs, ys, color="#2b6cb0", width=2.0):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />'


def _svg_line(x1, y1, x2, y2, color="#111827", width=1.0, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr} />'
    )


def _svg_rect(x, y, w, h, fill, opacity=0.2, stroke=None):
    stroke_attr = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="{fill}" fill-opacity="{opacity}"{stroke_attr} />'
    )


def _svg_text(x, y, text, size=12, color="#111827", anchor="start"):
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}" font-family="monospace">{safe}</text>'
    )


def _normalize(y, ymin, ymax):
    if ymax <= ymin:
        return np.zeros_like(y, dtype=np.float64)
    return (y - ymin) / (ymax - ymin)


def _write_svg(path: Path, width: int, height: int, body: str):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />'
        f"{body}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def render_full_timeline(
    out_path: Path,
    group_probs: np.ndarray,
    gt_groups,
    pred_groups,
    gt_onsets,
    pred_onsets,
):
    width, height = 1600, 520
    left, right, top, bottom = 70, 30, 40, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    T = max(len(group_probs), 1)
    xs = left + np.linspace(0, plot_w, T)
    ys = top + (1.0 - np.asarray(group_probs)) * plot_h

    body = []
    body.append(_svg_text(left, 24, "Mixed2 Group Probability Debug", size=18))
    body.append(_svg_line(left, top, left, top + plot_h, color="#374151"))
    body.append(_svg_line(left, top + plot_h, left + plot_w, top + plot_h, color="#374151"))
    body.append(_svg_line(left, top + plot_h * 0.5, left + plot_w, top + plot_h * 0.5, color="#d1d5db", dash="4 4"))
    body.append(_svg_polyline(xs, ys, color="#2563eb", width=2.0))

    for idx, (s, e) in enumerate(gt_groups):
        x = left + plot_w * s / T
        w = plot_w * (e - s) / T
        body.append(_svg_rect(x, top, w, plot_h, fill="#10b981", opacity=0.10, stroke="#10b981"))
        body.append(_svg_text(x + 4, top + 16, f"GT{idx}", size=11, color="#047857"))

    for idx, (s, e) in enumerate(pred_groups):
        x = left + plot_w * s / T
        w = plot_w * (e - s) / T
        body.append(_svg_rect(x, top + 24, w, plot_h - 48, fill="#f59e0b", opacity=0.08, stroke="#f59e0b"))
        body.append(_svg_text(x + 4, top + plot_h - 10, f"P{idx}", size=11, color="#92400e"))

    for group in gt_onsets:
        for t in group:
            x = left + plot_w * t / T
            body.append(_svg_line(x, top + plot_h - 42, x, top + plot_h, color="#059669", width=1.2))
    for group in pred_onsets:
        for t in group:
            x = left + plot_w * t / T
            body.append(_svg_line(x, top, x, top + 42, color="#dc2626", width=1.2))

    for frac, label in [(0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")]:
        y = top + (1.0 - frac) * plot_h
        body.append(_svg_text(left - 10, y + 4, label, size=11, anchor="end", color="#6b7280"))

    legend_y = height - 28
    body.append(_svg_line(left, legend_y, left + 18, legend_y, color="#2563eb", width=2))
    body.append(_svg_text(left + 24, legend_y + 4, "group prob", size=12))
    body.append(_svg_line(left + 140, legend_y, left + 158, legend_y, color="#059669", width=2))
    body.append(_svg_text(left + 164, legend_y + 4, "GT onsets", size=12))
    body.append(_svg_line(left + 270, legend_y, left + 288, legend_y, color="#dc2626", width=2))
    body.append(_svg_text(left + 294, legend_y + 4, "Pred onsets", size=12))
    body.append(_svg_text(width - 10, legend_y + 4, f"T={T} samples", size=12, anchor="end", color="#6b7280"))
    _write_svg(out_path, width, height, "".join(body))


def render_group_view(
    out_path: Path,
    group_idx: int,
    onset_probs: np.ndarray,
    pred_group,
    gt_group,
    pred_onsets,
    gt_onsets,
):
    width, height = 1400, 360
    left, right, top, bottom = 70, 30, 40, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    T = max(len(onset_probs), 1)
    xs = left + np.linspace(0, plot_w, T)
    ys = top + (1.0 - np.asarray(onset_probs)) * plot_h

    body = []
    body.append(_svg_text(left, 24, f"Group {group_idx} onset debug", size=18))
    body.append(_svg_line(left, top, left, top + plot_h, color="#374151"))
    body.append(_svg_line(left, top + plot_h, left + plot_w, top + plot_h, color="#374151"))
    body.append(_svg_line(left, top + plot_h * 0.5, left + plot_w, top + plot_h * 0.5, color="#d1d5db", dash="4 4"))
    body.append(_svg_polyline(xs, ys, color="#7c3aed", width=2.0))

    g_len = max(pred_group[1] - pred_group[0], 1)
    gt_rel_start = gt_group[0] - pred_group[0]
    gt_rel_end = gt_group[1] - pred_group[0]
    gt_x = left + plot_w * max(gt_rel_start, 0) / g_len
    gt_w = plot_w * max(gt_rel_end - max(gt_rel_start, 0), 0) / g_len
    body.append(_svg_rect(gt_x, top, gt_w, plot_h, fill="#10b981", opacity=0.08, stroke="#10b981"))

    for t in gt_onsets:
        rel = t - pred_group[0]
        x = left + plot_w * rel / g_len
        body.append(_svg_line(x, top + plot_h - 42, x, top + plot_h, color="#059669", width=1.2))
    for t in pred_onsets:
        rel = t - pred_group[0]
        x = left + plot_w * rel / g_len
        body.append(_svg_line(x, top, x, top + 42, color="#dc2626", width=1.2))

    deltas = [pred_onsets[i] - gt_onsets[i] for i in range(min(len(pred_onsets), len(gt_onsets)))]
    body.append(_svg_text(left, height - 38, f"pred_group={pred_group}  gt_group={gt_group}", size=12))
    body.append(_svg_text(left, height - 18, f"indexed_onset_deltas={deltas}", size=12, color="#6b7280"))
    _write_svg(out_path, width, height, "".join(body))


def main():
    parser = argparse.ArgumentParser(description="Debug one mixed2 case with SVG outputs")
    parser.add_argument("--mixed2_dir", required=True)
    parser.add_argument("--stage2a_ckpt", required=True)
    parser.add_argument("--stage2b_ckpt", required=True)
    parser.add_argument("--sample_rate", type=int, default=190)
    parser.add_argument("--output_dir", default="results/stage2_rebuild/debug_mixed2")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    sessions = discover_sessions(args.mixed2_dir, keyword="mixed2")
    if not sessions:
        raise SystemExit("No mixed2 session found.")
    sess_ref = sessions[0]

    config = PipelineConfig()
    config.signal.sample_rate = args.sample_rate
    pipeline = Stage2Pipeline.from_checkpoints(
        stage2a_ckpt=args.stage2a_ckpt,
        stage2b_ckpt=args.stage2b_ckpt,
        config=config,
        device=args.device,
    )

    session = load_mixed2_session(sess_ref, target_rate_hz=args.sample_rate)
    if session is None:
        raise SystemExit("Failed to load mixed2 GT.")

    results = pipeline.run(session["region_imu"], sample_rate=args.sample_rate)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "session_ref": sess_ref,
        "gt_groups": session["gt_group_boundaries"],
        "pred_groups": results["group_boundaries"],
        "gt_onsets": session["gt_onset_positions"],
        "pred_onsets": [arr.tolist() if hasattr(arr, "tolist") else list(arr) for arr in results["onset_positions"]],
        "gt_chars": ["".join(x) for x in session["gt_chars"]],
        "group_prob_len": int(len(results["group_probs"])),
        "per_group_debug": [],
    }

    for idx, (pred_group, gt_group, pred_onsets, gt_onsets, onset_probs) in enumerate(
        zip(
            results["group_boundaries"],
            session["gt_group_boundaries"],
            results["onset_positions"],
            session["gt_onset_positions"],
            results["onset_probs_per_group"],
        )
    ):
        pred_list = pred_onsets.tolist() if hasattr(pred_onsets, "tolist") else list(pred_onsets)
        deltas = [pred_list[i] - gt_onsets[i] for i in range(min(len(pred_list), len(gt_onsets)))]
        summary["per_group_debug"].append({
            "group_idx": idx,
            "pred_group": list(pred_group),
            "gt_group": list(gt_group),
            "pred_onsets": pred_list,
            "gt_onsets": list(gt_onsets),
            "indexed_onset_deltas": deltas,
        })
        render_group_view(
            out_dir / f"group_{idx}.svg",
            idx,
            np.asarray(onset_probs),
            pred_group,
            gt_group,
            pred_list,
            gt_onsets,
        )

    render_full_timeline(
        out_dir / "full_timeline.svg",
        np.asarray(results["group_probs"]),
        session["gt_group_boundaries"],
        results["group_boundaries"],
        session["gt_onset_positions"],
        [arr.tolist() if hasattr(arr, "tolist") else list(arr) for arr in results["onset_positions"]],
    )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Debug bundle saved to {out_dir}")


if __name__ == "__main__":
    main()
