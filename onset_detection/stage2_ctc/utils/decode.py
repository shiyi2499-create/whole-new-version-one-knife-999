"""
CTC decoding utilities.

This file now contains two decode families:
1. Standard CTC greedy / prefix beam search.
2. A rhythm-constrained posterior decoder that treats frame-level character
   posteriors as the strongest signal and explicitly penalizes overlong,
   over-dense strings.

The second path is intentionally pragmatic: current models often learn useful
local character evidence before their full-sequence CTC behavior stabilizes.
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from utils.vocab import IDX_TO_CHAR, BLANK_IDX, NUM_CLASSES


def greedy_decode(log_probs: np.ndarray) -> str:
    """
    Greedy CTC decode.

    Args:
        log_probs: [C, T] or [T, C] log-probabilities.
                   Auto-detected: if shape[0] == NUM_CLASSES, assume [C, T].
    Returns:
        Decoded string.
    """
    if log_probs.shape[0] == NUM_CLASSES:
        # [C, T] -> [T]
        best = np.argmax(log_probs, axis=0)
    else:
        # [T, C] -> [T]
        best = np.argmax(log_probs, axis=1)

    # CTC collapse: merge repeated, remove blanks
    decoded = []
    prev = -1
    for idx in best:
        if idx != prev:
            if idx != BLANK_IDX:
                decoded.append(int(idx))
        prev = idx

    return ''.join(IDX_TO_CHAR.get(i, '?') for i in decoded)


def prefix_beam_search(log_probs: np.ndarray, beam_width: int = 20,
                       blank: int = BLANK_IDX) -> List[Dict]:
    """
    Prefix beam search CTC decoder.

    Args:
        log_probs: [C, T] log-probabilities (log-softmax output).
        beam_width: number of beams to keep.
        blank: blank token index.

    Returns:
        List of {'candidate': str, 'score': float}, sorted by score descending.
    """
    C, T = log_probs.shape

    # State: dict mapping prefix (tuple of ints) -> (p_blank, p_non_blank)
    # p_blank = log-prob of paths ending in blank
    # p_non_blank = log-prob of paths ending in non-blank
    NEG_INF = -1e18

    beam = {(): (0.0, NEG_INF)}  # empty prefix, starts with "blank"

    for t in range(T):
        next_beam = defaultdict(lambda: (NEG_INF, NEG_INF))

        # Prune current beam
        scored = []
        for prefix, (pb, pnb) in beam.items():
            scored.append((prefix, pb, pnb, np.logaddexp(pb, pnb)))
        scored.sort(key=lambda x: -x[3])
        scored = scored[:beam_width]

        for prefix, pb, pnb, _ in scored:
            p_total = np.logaddexp(pb, pnb)

            # Extension by blank
            key = prefix
            old_pb, old_pnb = next_beam[key]
            new_pb = p_total + log_probs[blank, t]
            next_beam[key] = (np.logaddexp(old_pb, new_pb), old_pnb)

            # Extension by each character
            for c in range(C):
                if c == blank:
                    continue
                key = prefix + (c,)
                old_pb_c, old_pnb_c = next_beam[key]

                if prefix and prefix[-1] == c:
                    # Same char as last: only extend from blank-ending paths
                    new_pnb = pb + log_probs[c, t]
                    next_beam[key] = (old_pb_c, np.logaddexp(old_pnb_c, new_pnb))

                    # Also keep the non-extended prefix (collapse)
                    key2 = prefix
                    old_pb2, old_pnb2 = next_beam[key2]
                    new_pnb2 = pnb + log_probs[c, t]
                    next_beam[key2] = (old_pb2, np.logaddexp(old_pnb2, new_pnb2))
                else:
                    new_pnb = p_total + log_probs[c, t]
                    next_beam[key] = (old_pb_c, np.logaddexp(old_pnb_c, new_pnb))

        beam = dict(next_beam)

    # Final scoring
    results = []
    for prefix, (pb, pnb) in beam.items():
        score = float(np.logaddexp(pb, pnb))
        text = ''.join(IDX_TO_CHAR.get(i, '?') for i in prefix)
        # Filter out <unk> and <blank> from output string
        text = text.replace('<blank>', '').replace('<unk>', '')
        results.append({'candidate': text, 'score': score})

    results.sort(key=lambda x: -x['score'])
    return results[:beam_width]


def topk_strings(log_probs: np.ndarray, beam_width: int = 50,
                 top_k: int = 10) -> List[str]:
    """Convenience: return top-k decoded strings."""
    results = prefix_beam_search(log_probs, beam_width=beam_width)
    return [r['candidate'] for r in results[:top_k]]


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(xp, kernel, mode='valid')


def _find_local_maxima(x: np.ndarray, threshold: float,
                       min_separation: int) -> List[int]:
    peaks: List[int] = []
    n = len(x)
    last_keep = -10**9
    for i in range(1, n - 1):
        if x[i] < threshold:
            continue
        if x[i] < x[i - 1] or x[i] < x[i + 1]:
            continue
        if i - last_keep < min_separation:
            if peaks and x[i] > x[peaks[-1]]:
                peaks[-1] = i
                last_keep = i
            continue
        peaks.append(i)
        last_keep = i
    return peaks


def _log_gap_penalty(gap: int, target_gap: float, sigma: float) -> float:
    gap = max(float(gap), 1.0)
    target_gap = max(float(target_gap), 1.0)
    z = (np.log(gap) - np.log(target_gap)) / max(sigma, 1e-6)
    return -0.5 * float(z * z)


def rhythm_constrained_decode(
    frame_probs: np.ndarray,
    median_iki_frames: float = 258.0,
    min_chars: int = 4,
    max_chars: int = 12,
    smooth_win: int = 7,
    candidate_min_gap_frames: int = 12,
    min_select_gap_frames: int = 75,
    peak_quantile: float = 0.88,
    min_peak_threshold: float = 0.08,
    gap_sigma: float = 0.45,
    score_weight: float = 6.0,
    gap_weight: float = 1.8,
    len_weight: float = 5.0,
    tail_weight: float = 2.2,
    refine_radius: int = 8,
    char_topk: int = 6,
    char_bias_weight: float = 0.7,
    repeat_penalty: float = 0.8,
    global_repeat_penalty: float = 0.18,
) -> Dict:
    """
    Decode a sequence by selecting a sparse, monotonic subset of posterior peaks.

    The model's frame-level posterior is treated as the primary signal. We:
    1. Find candidate non-blank peaks.
    2. Select a monotonic subset with explicit rhythm and length priors.
    3. Read out characters at the selected frames.

    This is designed to combat the current failure mode where CTC beam search
    emits absurdly long repetitive strings despite useful local char evidence.
    """
    if frame_probs.shape[0] == NUM_CLASSES:
        probs_tc = frame_probs.T  # [T, C]
    else:
        probs_tc = frame_probs

    T, C = probs_tc.shape
    nonblank = probs_tc[:, 1:]
    best_char_local = np.argmax(nonblank, axis=1) + 1
    best_char_prob = np.max(nonblank, axis=1)
    blank_prob = probs_tc[:, BLANK_IDX]
    signal = best_char_prob * (1.0 - blank_prob)
    signal = _moving_average(signal.astype(np.float32), smooth_win)

    expected_len = int(round(T / max(median_iki_frames, 1.0)))
    expected_len = int(np.clip(expected_len, min_chars, max_chars))

    threshold = max(
        min_peak_threshold,
        float(np.quantile(signal, peak_quantile)) * 0.6,
    )
    peaks = _find_local_maxima(
        signal,
        threshold=threshold,
        min_separation=candidate_min_gap_frames,
    )

    if not peaks:
        return {
            'candidate': '',
            'score': -1e9,
            'positions': [],
            'expected_len': expected_len,
            'num_candidates': 0,
        }

    # Keep strongest candidates only; enough headroom for DP, but not thousands.
    peak_scores = np.array([signal[p] for p in peaks], dtype=np.float32)
    max_keep = max(expected_len * 6, 24)
    if len(peaks) > max_keep:
        keep_idx = np.argsort(-peak_scores)[:max_keep]
        keep_idx = np.sort(keep_idx)
        peaks = [peaks[i] for i in keep_idx]
        peak_scores = peak_scores[keep_idx]

    # Local character refinement near each candidate.
    refined_pos = []
    refined_char = []
    refined_score = []
    char_margin = []
    for p in peaks:
        lo = max(0, p - refine_radius)
        hi = min(T, p + refine_radius + 1)
        local = probs_tc[lo:hi, 1:]
        flat = np.argmax(local)
        rel_t, rel_c = np.unravel_index(flat, local.shape)
        t_sel = lo + rel_t
        c_sel = rel_c + 1
        sorted_local = np.sort(local[rel_t])[::-1]
        margin = float(sorted_local[0] - sorted_local[1]) if len(sorted_local) > 1 else float(sorted_local[0])
        refined_pos.append(t_sel)
        refined_char.append(c_sel)
        refined_score.append(float(probs_tc[t_sel, c_sel] * (1.0 - probs_tc[t_sel, BLANK_IDX])))
        char_margin.append(margin)

    candidates = list(zip(refined_pos, refined_char, refined_score, char_margin))
    candidates.sort(key=lambda x: x[0])

    N = len(candidates)
    max_len = min(max_chars, N)
    min_len = min(min_chars, max_len)
    target_len_lo = max(min_len, expected_len - 1)
    target_len_hi = min(max_len, expected_len + 1)
    target_gap = median_iki_frames

    dp = [[-1e18] * N for _ in range(max_len + 1)]
    prev = [[-1] * N for _ in range(max_len + 1)]

    for j, (pos, char_idx, sc, margin) in enumerate(candidates):
        start_pen = min(pos / max(target_gap, 1.0), 3.0)
        dp[1][j] = score_weight * (np.log(max(sc, 1e-8)) + 0.5 * margin) - 0.9 * start_pen

    for k in range(2, max_len + 1):
        for j in range(N):
            pos_j, char_j, sc_j, margin_j = candidates[j]
            best_score = -1e18
            best_i = -1
            for i in range(j):
                pos_i, _, _, _ = candidates[i]
                gap = pos_j - pos_i
                if gap < min_select_gap_frames:
                    continue
                score = dp[k - 1][i]
                if score <= -1e17:
                    continue
                score += score_weight * (np.log(max(sc_j, 1e-8)) + 0.5 * margin_j)
                score += gap_weight * _log_gap_penalty(gap, target_gap, gap_sigma)
                if score > best_score:
                    best_score = score
                    best_i = i
            dp[k][j] = best_score
            prev[k][j] = best_i

    best_total = -1e18
    best_end = (-1, -1)
    for k in range(target_len_lo, target_len_hi + 1):
        len_pen = -len_weight * ((k - expected_len) / max(1.0, expected_len * 0.20)) ** 2
        for j in range(N):
            score = dp[k][j]
            if score <= -1e17:
                continue
            pos_j = candidates[j][0]
            end_pen = min((T - 1 - pos_j) / max(target_gap, 1.0), 3.5)
            span = max(pos_j - candidates[0][0], 1)
            span_ratio = span / max(T - 1, 1)
            total = score + len_pen - tail_weight * end_pen + 1.2 * span_ratio
            if total > best_total:
                best_total = total
                best_end = (k, j)

    if best_end == (-1, -1):
        return {
            'candidate': '',
            'score': -1e9,
            'positions': [],
            'expected_len': expected_len,
            'num_candidates': len(candidates),
        }

    k, j = best_end
    chosen = []
    while j >= 0 and k >= 1:
        chosen.append(candidates[j])
        j = prev[k][j]
        k -= 1
    chosen.reverse()

    # Character assignment: do not trust per-peak top1 directly.
    # The current model often has strong global bias toward a few classes
    # (for example 'a' / '2'). We therefore calibrate local scores by the
    # episode-wide class prior and search over top-k characters per chosen peak.
    char_prior = np.mean(probs_tc[:, 1:], axis=0) + 1e-8
    seq_beam: List[Tuple[str, float, Dict[str, int]]] = [('', 0.0, defaultdict(int))]
    debug_topk = []
    for pos, _, _, _ in chosen:
        lo = max(0, pos - refine_radius)
        hi = min(T, pos + refine_radius + 1)
        local = probs_tc[lo:hi, 1:]  # [L, 37]
        # Calibrated local score per class
        local_best = np.max(local, axis=0)
        adjusted = np.log(np.maximum(local_best, 1e-8)) - char_bias_weight * np.log(char_prior)
        topk_idx = np.argsort(-adjusted)[:char_topk]
        local_choices = []
        for idx in topk_idx:
            c_idx = int(idx + 1)
            local_choices.append((IDX_TO_CHAR.get(c_idx, '?'), c_idx, float(adjusted[idx])))
        debug_topk.append({'pos': int(pos), 'choices': local_choices})

        next_beam: List[Tuple[str, float, Dict[str, int]]] = []
        for prefix, prefix_score, counts in seq_beam:
            for _, c_idx, c_score in local_choices:
                ch = IDX_TO_CHAR.get(c_idx, '?')
                penalty = 0.0
                if prefix and prefix[-1] == ch:
                    penalty -= repeat_penalty
                penalty -= global_repeat_penalty * counts.get(ch, 0)
                new_counts = dict(counts)
                new_counts[ch] = new_counts.get(ch, 0) + 1
                next_beam.append((prefix + ch, prefix_score + c_score + penalty, new_counts))
        next_beam.sort(key=lambda x: -x[1])
        seq_beam = next_beam[:24]

    chars = seq_beam[0][0] if seq_beam else ''.join(IDX_TO_CHAR.get(c, '?') for _, c, _, _ in chosen)
    return {
        'candidate': chars,
        'score': float(best_total),
        'positions': [int(p) for p, _, _, _ in chosen],
        'expected_len': expected_len,
        'num_candidates': len(candidates),
        'candidate_chars': [IDX_TO_CHAR.get(c, '?') for _, c, _, _ in chosen],
        'char_choice_debug': debug_topk,
    }
