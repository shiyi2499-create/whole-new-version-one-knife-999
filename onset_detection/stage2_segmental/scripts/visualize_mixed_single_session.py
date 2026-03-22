#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader, discover_sessions
from onset_detection.stage2_episode.utils.decoder import _compute_energy_envelope
from onset_detection.stage2_segmental.data import build_password_episodes, estimate_sample_rate_hz
from onset_detection.stage2_segmental.length_model import compute_region_length_features, load_length_model


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


def _write_svg(path: Path, width: int, height: int, body: str):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />'
        f"{body}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _smooth(values: np.ndarray, win_frames: int) -> np.ndarray:
    if win_frames <= 1:
        return values.astype(np.float64, copy=False)
    kernel = np.ones(int(win_frames), dtype=np.float64) / float(win_frames)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def _cluster_macro_peaks(peaks: np.ndarray, scores: np.ndarray, sample_rate: float, gap_s: float):
    if len(peaks) == 0:
        return []
    max_gap_frames = max(1, int(round(sample_rate * gap_s)))
    groups = []
    cur = [0]
    for i in range(1, len(peaks)):
        if int(peaks[i]) - int(peaks[i - 1]) <= max_gap_frames:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    out = []
    for g in groups:
        idx = np.asarray(g, dtype=np.int64)
        p = peaks[idx]
        s = scores[idx]
        out.append(
            {
                "start_frame": int(p[0]),
                "end_frame": int(p[-1]),
                "num_peaks": int(len(p)),
                "score_sum": float(np.sum(s)),
                "score_mean": float(np.mean(s)),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--length_model", required=True)
    ap.add_argument("--output_svg", required=True)
    ap.add_argument("--session_index", type=int, default=0)
    args = ap.parse_args()

    sessions = sorted(discover_sessions(args.input_dir))
    if not sessions:
        raise SystemExit("No sessions found")
    session_path = sessions[args.session_index]
    loader = SessionLoader(session_path)
    ts, imu = loader.get_imu()
    sr = estimate_sample_rate_hz(ts)
    energy = _compute_energy_envelope(imu, int(round(sr))).astype(np.float64)
    smoothed = _smooth(energy, max(1, int(round(sr * 0.15))))

    q50, q90, q98 = np.quantile(smoothed, [0.50, 0.90, 0.98])
    prominence = max(1e-6, (q90 - q50) * 0.10)
    height = q50 + (q98 - q50) * 0.05
    peaks, props = find_peaks(
        smoothed,
        distance=max(1, int(round(sr * 0.35))),
        prominence=prominence,
        height=height,
    )
    heights = np.asarray(props.get("peak_heights", smoothed[peaks]), dtype=np.float64)
    scores = heights / max(float(np.max(heights)), 1e-8) if len(heights) else np.asarray([], dtype=np.float64)
    clusters = sorted(
        _cluster_macro_peaks(peaks, scores, sr, gap_s=1.6),
        key=lambda x: (x["score_sum"], x["num_peaks"]),
        reverse=True,
    )[:5]

    episodes = build_password_episodes(args.input_dir)
    ep = [e for e in episodes if e.session_path == session_path][0]
    gt_start_s = float((ep.timestamps_ns[0] - ts[0]) / 1e9)
    gt_end_s = float((ep.timestamps_ns[-1] - ts[0]) / 1e9)
    key_times_s = ((ep.key_timestamps_ns - ts[0]) / 1e9).astype(np.float64)

    length_model = load_length_model(args.length_model)
    model, labels, meta = length_model
    cluster_ann = []
    for c in clusters:
        pad = int(round(sr * 1.5))
        lo = max(0, int(c["start_frame"]) - pad)
        hi = min(len(imu), int(c["end_frame"]) + pad + 1)
        feat = compute_region_length_features(
            imu[lo:hi],
            ts[lo:hi],
            feature_mode=str(meta.get("feature_mode", "no_time")),
        ).reshape(1, -1)
        pred = int(model.predict(feat)[0])
        probs = model.predict_proba(feat)[0] if hasattr(model, "predict_proba") else None
        conf = float(np.max(probs)) if probs is not None else 0.0
        cluster_ann.append((c, pred, conf))

    width, height = 1800, 820
    left, right, top, bottom = 70, 30, 45, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    T = max(len(smoothed), 1)
    xs = left + np.linspace(0, plot_w, T)
    y_norm = (smoothed - np.min(smoothed)) / max(np.max(smoothed) - np.min(smoothed), 1e-8)
    ys = top + (1.0 - y_norm) * plot_h

    body = []
    body.append(_svg_text(left, 24, f"Full-stream Mixed Password Visualization: {Path(session_path).name}", size=18))
    body.append(_svg_line(left, top, left, top + plot_h, color="#374151"))
    body.append(_svg_line(left, top + plot_h, left + plot_w, top + plot_h, color="#374151"))
    body.append(_svg_line(left, top + plot_h * 0.5, left + plot_w, top + plot_h * 0.5, color="#d1d5db", dash="4 4"))
    body.append(_svg_polyline(xs, ys, color="#111827", width=1.6))

    gt_x = left + plot_w * gt_start_s / max(time_s := float((ts[-1] - ts[0]) / 1e9), 1e-6)
    gt_w = plot_w * (gt_end_s - gt_start_s) / max(time_s, 1e-6)
    body.append(_svg_rect(gt_x, top, gt_w, plot_h, fill="#10b981", opacity=0.10, stroke="#10b981"))
    body.append(_svg_text(gt_x + 4, top + 16, f"GT password ({ep.password})", size=12, color="#047857"))

    for kt in key_times_s:
        x = left + plot_w * kt / max(time_s, 1e-6)
        body.append(_svg_line(x, top + plot_h - 50, x, top + plot_h, color="#059669", width=1.2))

    palette = ["#ef4444", "#f59e0b", "#3b82f6", "#8b5cf6", "#ec4899"]
    for idx, (c, pred, conf) in enumerate(cluster_ann):
        color = palette[idx % len(palette)]
        s = c["start_frame"] / sr
        e = c["end_frame"] / sr
        x = left + plot_w * s / max(time_s, 1e-6)
        w = plot_w * (e - s) / max(time_s, 1e-6)
        body.append(_svg_rect(x, top + 22 + idx * 18, w, plot_h - 44 - idx * 36, fill=color, opacity=0.08, stroke=color))
        body.append(_svg_text(
            x + 4,
            top + 18 + idx * 20,
            f"C{idx+1}: {s:.1f}-{e:.1f}s | peaks={c['num_peaks']} | len={pred} | conf={conf:.2f}",
            size=11,
            color=color,
        ))

    for p in peaks.tolist():
        x = left + plot_w * ((p / sr)) / max(time_s, 1e-6)
        y = top + (1.0 - y_norm[p]) * plot_h
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.3" fill="#1f2937" />')

    legend_y = height - 48
    body.append(_svg_line(left, legend_y, left + 20, legend_y, color="#111827", width=2))
    body.append(_svg_text(left + 28, legend_y + 4, "Smoothed energy", size=12))
    body.append(_svg_line(left + 170, legend_y, left + 190, legend_y, color="#059669", width=2))
    body.append(_svg_text(left + 198, legend_y + 4, "GT key timestamps", size=12))
    body.append(_svg_rect(left + 360, legend_y - 12, 24, 12, fill="#10b981", opacity=0.10, stroke="#10b981"))
    body.append(_svg_text(left + 392, legend_y + 4, "GT password span", size=12))
    body.append(_svg_text(width - 10, legend_y + 4, f"sample_rate≈{sr:.1f}Hz", size=12, anchor="end", color="#6b7280"))

    out_path = Path(args.output_svg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_svg(out_path, width, height, "".join(body))
    print(str(out_path))


if __name__ == "__main__":
    main()
