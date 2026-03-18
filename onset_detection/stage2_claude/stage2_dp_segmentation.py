"""
Stage 2v3: Classifier-in-the-Loop Constrained DP Segmentation
==============================================================

Core insight (inspired by CTC / Viterbi decoding literature):
  Instead of the serial pipeline  onset_detect → group → classify  where
  errors compound at each stage, we let the **already-trained classifier**
  participate in the segmentation decision.

  For every candidate onset time, the classifier produces P(char | window).
  The max class probability acts as a 'keystroke quality' score — real keystrokes
  produce confident predictions, while false alarms (noise / trackpad / Enter)
  produce low-confidence uniform-ish distributions.

  We then formulate the segmentation as a constrained optimization:

    Find exactly  N_pw * L  onsets from the candidate pool,
    partitioned into N_pw groups of exactly L each,
    that maximizes  Σ [ onset_prob(t) * classifier_confidence(t) ]
    subject to:
      - within each group, IKI is in [iki_min, iki_max]
      - between groups, the gap is ≥ min_gap
      - onsets are in temporal order

  This is solved via Dynamic Programming with O(N_candidates * N_pw * L) states,
  which is very fast for our scale (~100 candidates, 5 groups, 8 onsets).

References:
  - CTC (Graves et al., ICML 2006): decode sequence labels from unsegmented input
  - Viterbi algorithm: find optimal path through a trellis under constraints
  - My(o) Armband (Grünerbl et al., IMWUT 2022): EMG/IMU keystroke side-channel

This module is designed as a drop-in replacement for the Stage 2 grouping
logic in password_segment_detector.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ── Candidate Generation ──────────────────────────────────────

@dataclass
class OnsetCandidate:
    """A candidate onset with scores from both the onset detector and classifier."""
    time_s: float
    time_ns: int
    onset_prob: float           # from onset detector
    classifier_entropy: float   # entropy of classifier output (lower = more confident)
    classifier_max_prob: float  # max P(char) from classifier
    classifier_probs: Optional[np.ndarray] = None  # full probability vector
    combined_score: float = 0.0


def generate_onset_candidates(
    sensor: np.ndarray,
    onset_model: torch.nn.Module,
    onset_means: np.ndarray,
    onset_stds: np.ndarray,
    device: torch.device,
    region_start_s: float,
    region_end_s: float,
    window_ms: int,
    stride_ms: int,
    target_rate_hz: int,
    threshold: float = 0.20,       # very low — intentionally permissive
    nms_radius_s: float = 0.08,    # tighter NMS to keep more candidates
    smooth_n: int = 3,
    max_candidates: int = 120,
) -> list[dict]:
    """
    Generate a large pool of onset candidates with low threshold.
    We intentionally over-detect: the DP will select the best subset.
    """
    from onset_utils import detect_peaks, nms_1d
    from password_segment_preprocessor import _iterate_window_chunks

    ts_ns = sensor[:, 0]
    mask = (ts_ns >= region_start_s * 1e9) & (ts_ns <= region_end_s * 1e9)
    if mask.sum() < 10:
        return []

    rsensor = sensor[mask]
    windows, times = [], []
    for c, w in _iterate_window_chunks(rsensor, window_ms, stride_ms, target_rate_hz):
        windows.append((w - onset_means) / onset_stds)
        times.append(c / 1e9)

    if not windows:
        return []

    X = np.stack(windows).astype(np.float32)
    ts_arr = np.asarray(times)

    probs_list = []
    onset_model.eval()
    with torch.no_grad():
        for i in range(0, len(X), 256):
            b = torch.from_numpy(X[i:i + 256]).to(device)
            logits = onset_model(b)
            p_batch = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            probs_list.append(p_batch)

    p = np.concatenate(probs_list)
    raw_peaks = detect_peaks(p, ts_arr, threshold=threshold, smooth_n=smooth_n)
    peaks = nms_1d(raw_peaks, radius_s=nms_radius_s)

    if max_candidates and len(peaks) > max_candidates:
        peaks = sorted(peaks, key=lambda pk: -pk["prob"])[:max_candidates]
        peaks = sorted(peaks, key=lambda pk: pk["time_s"])

    return peaks


def score_candidates_with_classifier(
    sensor: np.ndarray,
    candidates: list[dict],
    classifier: torch.nn.Module,
    cls_means: np.ndarray,
    cls_stds: np.ndarray,
    device: torch.device,
    target_rate_hz: int,
    pre_ms: int = 100,
    post_ms: int = 200,
) -> list[OnsetCandidate]:
    """
    For each onset candidate, cut a classifier window, run inference,
    and compute both the max class probability and entropy.
    """
    from onset_preprocessor import resample_window, window_samples

    ts = sensor[:, 0]
    vals = sensor[:, 1:]
    tgt_len = window_samples(pre_ms + post_ms, target_rate_hz)

    scored = []
    batch_windows = []
    batch_indices = []

    for idx, pk in enumerate(candidates):
        t_ns = int(pk["time_s"] * 1e9)
        i0 = np.searchsorted(ts, t_ns - pre_ms * 1e6, side="left")
        i1 = np.searchsorted(ts, t_ns + post_ms * 1e6, side="right")

        if i1 - i0 < 4:
            scored.append(OnsetCandidate(
                time_s=pk["time_s"],
                time_ns=t_ns,
                onset_prob=pk["prob"],
                classifier_entropy=10.0,  # high entropy = bad
                classifier_max_prob=0.0,
                combined_score=0.0,
            ))
            continue

        win = resample_window(vals[i0:i1], tgt_len)
        batch_windows.append(win)
        batch_indices.append(idx)
        scored.append(None)  # placeholder

    # Batch classify
    if batch_windows:
        X = np.stack(batch_windows).astype(np.float32)
        for ch in range(X.shape[2]):
            X[:, :, ch] = (X[:, :, ch] - cls_means[ch]) / (cls_stds[ch] + 1e-10)

        classifier.eval()
        with torch.no_grad():
            logits = classifier(torch.from_numpy(X).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for batch_idx, orig_idx in enumerate(batch_indices):
            pk = candidates[orig_idx]
            p_vec = probs[batch_idx]
            max_p = float(np.max(p_vec))

            # Entropy: lower = more confident = more likely a real keystroke
            entropy = float(-np.sum(p_vec * np.log(np.maximum(p_vec, 1e-12))))
            # Uniform entropy for 36 classes: log(36) ≈ 3.58
            max_entropy = math.log(len(p_vec))
            norm_entropy = entropy / max(max_entropy, 1e-6)

            # Combined score: high onset_prob + high classifier confidence
            onset_p = pk["prob"]
            combined = onset_p * (1.0 - norm_entropy) * (0.3 + 0.7 * max_p)

            scored[orig_idx] = OnsetCandidate(
                time_s=pk["time_s"],
                time_ns=int(pk["time_s"] * 1e9),
                onset_prob=onset_p,
                classifier_entropy=entropy,
                classifier_max_prob=max_p,
                classifier_probs=p_vec,
                combined_score=combined,
            )

    # Fill any remaining None (shouldn't happen, but safety)
    result = []
    for idx, pk in enumerate(candidates):
        if scored[idx] is None:
            scored[idx] = OnsetCandidate(
                time_s=pk["time_s"],
                time_ns=int(pk["time_s"] * 1e9),
                onset_prob=pk["prob"],
                classifier_entropy=10.0,
                classifier_max_prob=0.0,
                combined_score=0.0,
            )
        result.append(scored[idx])

    return result


# ── Constrained DP Segmentation ──────────────────────────────

def dp_segment_passwords(
    candidates: list[OnsetCandidate],
    n_passwords: int = 5,
    password_len: int = 8,
    iki_min_s: float = 0.08,
    iki_max_s: float = 1.5,
    min_inter_password_gap_s: float = 0.5,
    gap_bonus_weight: float = 0.3,
) -> list[list[OnsetCandidate]]:
    """
    Find the globally optimal assignment of candidates to N_pw groups of L each,
    using Dynamic Programming.

    State: dp[i][g][k] = best total score achievable using candidates[0..i],
           having completed g full groups and k onsets in the current group,
           where candidate i is the last selected onset.

    Transitions:
      - SAME GROUP:  dp[i][g][k] <- dp[j][g][k-1] + score(i)
                     where iki_min <= (t_i - t_j) <= iki_max
      - NEW GROUP:   dp[i][g][0] <- dp[j][g-1][L-1] + score(i) + gap_bonus
                     where (t_i - t_j) >= min_inter_password_gap

    Total onsets to select: n_passwords * password_len
    """
    N = len(candidates)
    L = password_len
    G = n_passwords

    if N < G * L:
        # Not enough candidates
        return _fallback_equal_split(candidates, G, L)

    NEG_INF = -1e18

    # dp[i][g][k] = best score ending at candidate i, with g complete groups, k onsets in current group
    # k ranges from 0 to L-1 (0 = first onset in group, L-1 = last onset in group)
    dp = np.full((N, G + 1, L), NEG_INF, dtype=np.float64)
    parent = np.full((N, G + 1, L), -1, dtype=np.int32)  # backtrack pointer

    times = np.array([c.time_s for c in candidates], dtype=np.float64)
    scores = np.array([c.combined_score for c in candidates], dtype=np.float64)

    # Initialize: candidate i is the first onset (k=0) of the first group (g=0)
    for i in range(N):
        dp[i, 0, 0] = scores[i]

    # Fill DP
    for i in range(1, N):
        t_i = times[i]
        s_i = scores[i]

        for j in range(i - 1, -1, -1):
            t_j = times[j]
            dt = t_i - t_j

            if dt < iki_min_s * 0.5:
                continue  # too close, skip
            if dt > iki_max_s * 10:
                break  # too far back, no point checking further

            # Transition 1: SAME GROUP (j and i in same group)
            if iki_min_s <= dt <= iki_max_s:
                for g in range(G):
                    for k in range(1, L):
                        new_score = dp[j, g, k - 1] + s_i
                        if new_score > dp[i, g, k]:
                            dp[i, g, k] = new_score
                            parent[i, g, k] = j

            # Transition 2: NEW GROUP (j is last onset of previous group, i is first of new)
            if dt >= min_inter_password_gap_s:
                gap_bonus = gap_bonus_weight * min(dt / min_inter_password_gap_s, 3.0)
                for g in range(1, G + 1):
                    new_score = dp[j, g - 1, L - 1] + s_i + gap_bonus
                    if new_score > dp[i, g, 0]:
                        dp[i, g, 0] = new_score
                        parent[i, g, 0] = j

    # Find best ending point: last onset of the last group
    best_score = NEG_INF
    best_end = -1
    for i in range(N):
        if dp[i, G - 1, L - 1] > best_score:
            best_score = dp[i, G - 1, L - 1]
            best_end = i

    # Also check if we completed all G groups (g=G with k=L-1 means wrong indexing)
    # Actually: g=G-1, k=L-1 means we're in the last (G-th) group, at the last onset
    # But our indexing: g counts COMPLETED groups before current.
    # So completing all G groups means: the last onset has g=G-1, k=L-1
    # which means: G-1 groups already completed, and k=L-1 (last onset of current = G-th group)
    # This is correct.

    if best_end < 0 or best_score <= NEG_INF / 2:
        return _fallback_equal_split(candidates, G, L)

    # Backtrack
    path = []
    i = best_end
    g = G - 1
    k = L - 1

    while i >= 0:
        path.append((i, g, k))
        prev_i = int(parent[i, g, k])
        if prev_i < 0:
            break

        if k > 0:
            # Same group, previous onset
            k -= 1
        elif g > 0:
            # First onset of this group; previous was last of previous group
            g -= 1
            k = L - 1
        else:
            # We're at the very first onset
            break
        i = prev_i

    path.reverse()

    if len(path) != G * L:
        return _fallback_equal_split(candidates, G, L)

    # Build groups
    groups = []
    for g_idx in range(G):
        group = []
        for k_idx in range(L):
            flat_idx = g_idx * L + k_idx
            if flat_idx < len(path):
                cand_idx = path[flat_idx][0]
                group.append(candidates[cand_idx])
        groups.append(group)

    return groups


def _fallback_equal_split(
    candidates: list[OnsetCandidate],
    n_groups: int,
    group_len: int,
) -> list[list[OnsetCandidate]]:
    """Fallback: pick top candidates by combined_score and split equally."""
    total_needed = n_groups * group_len
    if not candidates:
        return [[] for _ in range(n_groups)]

    ranked = sorted(candidates, key=lambda c: -c.combined_score)
    selected = sorted(ranked[:total_needed], key=lambda c: c.time_s)

    groups = []
    for g in range(n_groups):
        start = g * group_len
        end = start + group_len
        groups.append(selected[start:end] if end <= len(selected) else selected[start:])

    return groups


# ── Main Entry Point ──────────────────────────────────────────

def run_stage2_dp(
    sensor: np.ndarray,
    coarse_regions: list,  # list of CoarseRegion
    onset_model: torch.nn.Module,
    onset_means: np.ndarray,
    onset_stds: np.ndarray,
    onset_meta: dict,
    classifier: torch.nn.Module,
    cls_classes: np.ndarray,
    cls_means: np.ndarray,
    cls_stds: np.ndarray,
    device: torch.device,
    n_passwords: int = 5,
    password_len: int = 8,
    onset_threshold: float = 0.20,
    onset_nms_radius_s: float = 0.08,
    max_candidates: int = 120,
    iki_min_s: float = 0.08,
    iki_max_s: float = 1.5,
    min_inter_password_gap_s: float = 0.5,
) -> tuple[list[list[int]], list[list[Optional[np.ndarray]]], dict]:
    """
    Stage 2v3: Classifier-in-the-Loop Constrained DP Segmentation.

    Returns:
        password_groups_ns: list of onset time lists (in nanoseconds), one per password
        password_prob_vecs: list of classifier probability vector lists, one per password
        debug: debug info dict
    """
    target_rate_hz = onset_meta["target_rate_hz"]
    window_ms = onset_meta["window_ms"]
    stride_ms = onset_meta["stride_ms"]

    # Step 1: Generate many candidates across all coarse regions
    all_candidates_raw = []
    for region in coarse_regions:
        peaks = generate_onset_candidates(
            sensor, onset_model, onset_means, onset_stds, device,
            region.start_s, region.end_s,
            window_ms, stride_ms, target_rate_hz,
            threshold=onset_threshold,
            nms_radius_s=onset_nms_radius_s,
            max_candidates=max_candidates,
        )
        all_candidates_raw.extend(peaks)

    all_candidates_raw = sorted(all_candidates_raw, key=lambda pk: pk["time_s"])
    n_raw = len(all_candidates_raw)

    # Step 2: Score each candidate with the classifier
    scored_candidates = score_candidates_with_classifier(
        sensor, all_candidates_raw,
        classifier, cls_means, cls_stds, device,
        target_rate_hz,
    )

    # Step 3: DP segmentation
    groups = dp_segment_passwords(
        scored_candidates,
        n_passwords=n_passwords,
        password_len=password_len,
        iki_min_s=iki_min_s,
        iki_max_s=iki_max_s,
        min_inter_password_gap_s=min_inter_password_gap_s,
    )

    # Step 4: Extract results
    password_groups_ns = []
    password_prob_vecs = []
    for group in groups:
        password_groups_ns.append([c.time_ns for c in group])
        password_prob_vecs.append([c.classifier_probs for c in group])

    # Debug info
    debug = {
        "method": "dp_classifier_in_the_loop",
        "n_raw_candidates": n_raw,
        "n_scored_candidates": len(scored_candidates),
        "n_groups": len(groups),
        "groups": [],
    }
    for g_idx, group in enumerate(groups):
        g_info = {
            "n_onsets": len(group),
            "onset_times_s": [c.time_s for c in group],
            "onset_probs": [c.onset_prob for c in group],
            "classifier_max_probs": [c.classifier_max_prob for c in group],
            "combined_scores": [c.combined_score for c in group],
        }
        if len(group) >= 2:
            ikis = [group[i + 1].time_s - group[i].time_s for i in range(len(group) - 1)]
            g_info["ikis_s"] = ikis
            g_info["median_iki_s"] = float(np.median(ikis))
        debug["groups"].append(g_info)

    if len(groups) >= 2:
        inter_gaps = []
        for i in range(len(groups) - 1):
            if groups[i] and groups[i + 1]:
                inter_gaps.append(groups[i + 1][0].time_s - groups[i][-1].time_s)
        debug["inter_password_gaps_s"] = inter_gaps

    # Score distribution summary
    all_combined = [c.combined_score for c in scored_candidates]
    selected_combined = [c.combined_score for g in groups for c in g]
    if all_combined:
        debug["score_stats"] = {
            "all_mean": float(np.mean(all_combined)),
            "all_median": float(np.median(all_combined)),
            "selected_mean": float(np.mean(selected_combined)) if selected_combined else 0.0,
            "selected_median": float(np.median(selected_combined)) if selected_combined else 0.0,
        }

    return password_groups_ns, password_prob_vecs, debug
