#!/usr/bin/env python3
"""
Episode-level end-to-end evaluation:
  Stage 2 episode detection/onset recovery -> Stage 3 password classifier.

This keeps the episode-based task definition:
  - detect password typing episodes
  - recover characters inside each detected episode

Evaluation is done against GT password groups extracted from mixed_training/mixed2.
For each GT episode, we use the matched predicted episode if one exists; otherwise
the prediction is treated as empty.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))

for p in (PKG_ROOT, os.path.join(PROJECT_ROOT, "phase3_password_inception"), PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_password_closure_inception import load_final_inception, topk_strings_from_prob_vectors

from utils.metrics import match_episodes
from run_e2e_episode import load_model, load_gt, run_one_session
from data.loaders import discover_sessions
from configs.config import SignalConfig


DEFAULT_TARGET_RATE_HZ = 190


def window_samples(window_ms: int, target_rate_hz: int = DEFAULT_TARGET_RATE_HZ) -> int:
    return int(round(window_ms / 1000.0 * target_rate_hz))


def resample_window(values: np.ndarray, target_len: int) -> np.ndarray:
    from scipy.signal import resample

    out = resample(values, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def cut_classifier_windows(sensor, onset_times_ns, pre_ms=100, post_ms=200,
                           target_rate_hz=DEFAULT_TARGET_RATE_HZ):
    ts = sensor[:, 0]
    vals = sensor[:, 1:]
    tgt = window_samples(pre_ms + post_ms, target_rate_hz)
    out = []
    for t_ns in onset_times_ns:
        i0 = np.searchsorted(ts, t_ns - pre_ms * 1e6, side="left")
        i1 = np.searchsorted(ts, t_ns + post_ms * 1e6, side="right")
        if i1 - i0 < 4:
            out.append(None)
        else:
            out.append(resample_window(vals[i0:i1], tgt))
    return out


def classify_windows(windows, classifier, means, stds, device):
    valid_idx = [i for i, w in enumerate(windows) if w is not None]
    if not valid_idx:
        return [None] * len(windows)
    X = np.stack([windows[i] for i in valid_idx]).astype(np.float32)
    for ch in range(X.shape[2]):
        X[:, :, ch] = (X[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)
    classifier.eval()
    with torch.no_grad():
        probs = torch.softmax(classifier(torch.from_numpy(X).to(device)), dim=1).cpu().numpy()
    out = [None] * len(windows)
    for bi, oi in enumerate(valid_idx):
        out[oi] = probs[bi]
    return out


def cluster_windows_by_time(onset_times_ns, prob_vecs, cluster_gap_ms=140.0):
    """
    Merge near-duplicate onset candidates before count pruning.

    If several predicted onsets land within a very short time span, they are
    much more likely to be repeated detections of the same physical keypress
    than genuinely distinct characters. Keep only the classifier-strongest
    member of each cluster.
    """
    valid = [(i, onset_times_ns[i], prob_vecs[i]) for i in range(len(prob_vecs)) if prob_vecs[i] is not None]
    if not valid:
        return [], []

    gap_ns = int(cluster_gap_ms * 1e6)
    clusters = []
    cur = [valid[0]]
    for item in valid[1:]:
        if item[1] - cur[-1][1] <= gap_ns:
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]
    clusters.append(cur)

    keep_times = []
    keep_probs = []
    for cluster in clusters:
        best = max(cluster, key=lambda x: float(np.max(x[2])))
        keep_times.append(best[1])
        keep_probs.append(best[2])
    return keep_times, keep_probs


def prune_windows_with_classifier(onset_times_ns, prob_vecs, max_keys=None):
    """
    Drop obviously redundant windows when Stage 2 still over-fires.

    We keep chronology, but choose the strongest classifier-supported windows
    when the predicted count is implausibly large for one password episode.
    """
    clustered_times, clustered_probs = cluster_windows_by_time(onset_times_ns, prob_vecs)
    valid = list(zip(clustered_times, clustered_probs))
    if not valid:
        return [], []

    if max_keys is None or len(valid) <= max_keys:
        return [t for t, _ in valid], [p for _, p in valid]

    scores = np.array([float(np.max(p)) for _, p in valid], dtype=np.float64)
    keep_local = np.argsort(scores)[-max_keys:]
    keep_local = sorted(keep_local)
    return [valid[k][0] for k in keep_local], [valid[k][1] for k in keep_local]


def estimate_plausible_key_upper(onset_times_ns):
    if len(onset_times_ns) <= 2:
        return len(onset_times_ns)
    span_s = max((onset_times_ns[-1] - onset_times_ns[0]) / 1e9, 0.0)
    # Empirically, password-entry episodes in our data are much denser than the
    # raw Stage 2 candidate stream suggests. Use a conservative upper bound so
    # obviously duplicated detections do not survive into character decoding.
    expected = int(round(span_s * 0.95)) + 1
    expected = int(np.clip(expected, 2, 10))
    upper = max(expected + 1, int(round(expected * 1.15)))
    return min(upper, 12)


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


SEQ_HIT_CUTOFFS = (10, 50, 100)


def score_one_episode(onset_times_ns, ref, sensor, classifier, cls_classes,
                      cls_means, cls_stds, device, target_rate_hz):
    windows = cut_classifier_windows(sensor, onset_times_ns, target_rate_hz=target_rate_hz)
    prob_vecs = classify_windows(windows, classifier, cls_means, cls_stds, device)
    pruned_onsets_ns, valid = prune_windows_with_classifier(
        onset_times_ns,
        prob_vecs,
        max_keys=estimate_plausible_key_upper(onset_times_ns),
    )

    result = {
        "reference": ref,
        "hypothesis": "",
        "n_onsets": len(onset_times_ns),
        "n_onsets_used": len(pruned_onsets_ns),
        "n_valid_windows": len(valid),
        "cer": levenshtein(ref, "") / max(len(ref), 1),
        "char_top1": 0.0,
        "char_top3": 0.0,
        "char_top5": 0.0,
    }
    for cutoff in SEQ_HIT_CUTOFFS:
        result[f"seq_top{cutoff}"] = 0

    if not valid:
        return result

    hyp = "".join(cls_classes[int(np.argmax(p))] for p in valid)
    result["hypothesis"] = hyp
    result["cer"] = levenshtein(ref, hyp) / max(len(ref), 1)

    hits = {1: 0, 3: 0, 5: 0}
    for i, ref_ch in enumerate(ref):
        if i >= len(valid):
            break
        ranked = [cls_classes[r] for r in np.argsort(-valid[i])]
        for k in (1, 3, 5):
            if ref_ch in ranked[:k]:
                hits[k] += 1
    denom = max(len(ref), 1)
    result["char_top1"] = hits[1] / denom
    result["char_top3"] = hits[3] / denom
    result["char_top5"] = hits[5] / denom

    try:
        cands = topk_strings_from_prob_vectors(
            np.stack(valid), cls_classes, branch_topk=5, beam_width=max(SEQ_HIT_CUTOFFS)
        )
        cand_strs = [c["candidate"] for c in cands]
        for cutoff in SEQ_HIT_CUTOFFS:
            result[f"seq_top{cutoff}"] = 1 if ref in cand_strs[:cutoff] else 0
    except Exception:
        pass

    return result


def init_aggregate():
    out = {
        "n_sessions": 0,
        "n_episodes": 0,
        "n_chars": 0,
        "char_top1_sum": 0.0,
        "char_top3_sum": 0.0,
        "char_top5_sum": 0.0,
        "cer_edits": 0.0,
        "seq_hits": {cutoff: 0 for cutoff in SEQ_HIT_CUTOFFS},
    }
    return out


def update_aggregate(agg, res):
    ref = res["reference"]
    hyp = res["hypothesis"]
    n = max(len(ref), 1)
    agg["n_episodes"] += 1
    agg["n_chars"] += len(ref)
    agg["char_top1_sum"] += res["char_top1"] * n
    agg["char_top3_sum"] += res["char_top3"] * n
    agg["char_top5_sum"] += res["char_top5"] * n
    agg["cer_edits"] += levenshtein(ref, hyp)
    for cutoff in SEQ_HIT_CUTOFFS:
        agg["seq_hits"][cutoff] += int(res[f"seq_top{cutoff}"])


def finalize_aggregate(agg):
    n_chars = max(agg["n_chars"], 1)
    n_eps = max(agg["n_episodes"], 1)
    return {
        "n_sessions": agg["n_sessions"],
        "n_episodes": agg["n_episodes"],
        "n_chars": agg["n_chars"],
        "char_top1": agg["char_top1_sum"] / n_chars,
        "char_top3": agg["char_top3_sum"] / n_chars,
        "char_top5": agg["char_top5_sum"] / n_chars,
        "cer": agg["cer_edits"] / n_chars,
        **{f"seq_top{cutoff}": agg["seq_hits"][cutoff] / n_eps for cutoff in SEQ_HIT_CUTOFFS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed2_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--classifier-checkpoint", required=True)
    ap.add_argument("--classifier-scaler", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--sample_rate", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage2b-ckpt", default=None)
    ap.add_argument("--typing-threshold", type=float, default=None)
    ap.add_argument("--target-rate-hz", type=int, default=DEFAULT_TARGET_RATE_HZ)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sr = args.sample_rate

    dev = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    model, _mcfg, ecfg = load_model(args.checkpoint, dev)
    scfg = SignalConfig(sample_rate=sr)
    onset_aux = None
    if args.stage2b_ckpt:
        from utils.onset_refine import load_stage2b_refiner
        onset_aux = load_stage2b_refiner(args.stage2b_ckpt, dev)

    classifier, cls_classes, cls_means, cls_stds = load_final_inception(
        args.classifier_checkpoint, args.classifier_scaler, dev
    )

    sessions = discover_sessions(args.mixed2_dir)
    if not sessions and (Path(args.mixed2_dir) / "sensor.csv").exists():
        sessions = [args.mixed2_dir]

    all_data = []
    for sd in sessions:
        data = load_gt(sd, sr)
        if data is not None:
            all_data.append((sd, data))

    print(f"Found {len(all_data)} valid sessions\n")

    e2e_agg = init_aggregate()
    gt_agg = init_aggregate()
    sessions_out = []

    for i, (sd, data) in enumerate(all_data, 1):
        name = Path(sd).name
        print(f"--- Session {i}: {name} ---")
        dec, ev, _preds = run_one_session(
            model, data, sr, scfg, ecfg, dev,
            onset_aux=onset_aux,
            typing_threshold=args.typing_threshold,
        )

        sensor = np.column_stack([data["ts"], data["imu"]]).astype(np.float64)
        gt_eps = data["gt_episodes"]
        match = match_episodes(dec["episodes"], gt_eps)
        matched_by_gt = {gi: pi for pi, gi, _iou in match["matches"]}

        episode_results = []
        for gi, gt_ep in enumerate(gt_eps):
            ref = "".join(gt_ep.get("chars", []))
            pred_ep = dec["episodes"][matched_by_gt[gi]] if gi in matched_by_gt else None
            pred_onsets_ns = []
            if pred_ep is not None:
                pred_onsets_ns = [int(data["ts"][idx]) for idx in pred_ep["onsets"] if 0 <= idx < len(data["ts"])]
            gt_onsets_ns = [int(data["ts"][idx]) for idx in gt_ep["onsets"] if 0 <= idx < len(data["ts"])]

            e2e_res = score_one_episode(
                pred_onsets_ns, ref, sensor, classifier, cls_classes,
                cls_means, cls_stds, dev, args.target_rate_hz
            )
            gt_res = score_one_episode(
                gt_onsets_ns, ref, sensor, classifier, cls_classes,
                cls_means, cls_stds, dev, args.target_rate_hz
            )
            update_aggregate(e2e_agg, e2e_res)
            update_aggregate(gt_agg, gt_res)
            episode_results.append({
                "gt_index": gi,
                "matched_pred_index": matched_by_gt.get(gi, None),
                "reference": ref,
                "e2e": e2e_res,
                "gt_baseline": gt_res,
            })
            print(f"  ep{gi+1}: ref={ref}  e2e={e2e_res['hypothesis']}  gt={gt_res['hypothesis']}")

        e2e_agg["n_sessions"] += 1
        gt_agg["n_sessions"] += 1
        sess_out = {
            "session": name,
            "stage2": {
                "num_pred_episodes": dec["num_episodes"],
                "num_gt_episodes": len(gt_eps),
                "episode_eval": ev,
                "typing_threshold": args.typing_threshold,
            },
            "episode_results": episode_results,
        }
        sessions_out.append(sess_out)
        with open(out / f"{name}_char_results.json", "w") as f:
            json.dump(sess_out, f, indent=2, default=str)

    summary = {
        "e2e_full": finalize_aggregate(e2e_agg),
        "gt_baseline": finalize_aggregate(gt_agg),
        "n_sessions": len(all_data),
        "typing_threshold": args.typing_threshold,
    }
    with open(out / "aggregate_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"AGGREGATE ({summary['n_sessions']} sessions)")
    print("=" * 60)
    ef = summary["e2e_full"]
    gb = summary["gt_baseline"]
    print("\nE2E Full:")
    print(f"  char_top1: {ef['char_top1']*100:.1f}%   top3: {ef['char_top3']*100:.1f}%   top5: {ef['char_top5']*100:.1f}%")
    print(f"  seq_top10: {ef['seq_top10']*100:.1f}%   seq_top50: {ef['seq_top50']*100:.1f}%   seq_top100: {ef['seq_top100']*100:.1f}%")
    print(f"  CER: {ef['cer']*100:.1f}%")
    print("\nGT Baseline:")
    print(f"  char_top1: {gb['char_top1']*100:.1f}%   top3: {gb['char_top3']*100:.1f}%   top5: {gb['char_top5']*100:.1f}%")
    print(f"  seq_top10: {gb['seq_top10']*100:.1f}%   seq_top50: {gb['seq_top50']*100:.1f}%   seq_top100: {gb['seq_top100']*100:.1f}%")
    print(f"  CER: {gb['cer']*100:.1f}%")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
