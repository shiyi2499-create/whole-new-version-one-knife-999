"""
Evaluation metrics for frame-level CTC password decoding.

Includes:
  - Character Error Rate (CER) via Levenshtein distance
  - Per-position top-k accuracy (using GT onset timestamps)
  - Sequence-level exact match / top-k hit rate
"""
import numpy as np
from typing import List, Dict, Tuple, Optional

from utils.vocab import IDX_TO_CHAR, BLANK_IDX, NUM_CLASSES


def levenshtein(a: str, b: str) -> int:
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
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def char_topk_at_gt_positions(
    frame_probs: np.ndarray,
    gt_onset_frames: List[int],
    gt_chars: List[str],
    ks: Tuple[int, ...] = (1, 3, 5),
) -> Dict[str, float]:
    """
    Evaluate character top-k accuracy at ground-truth onset positions.

    This is the oracle metric: how good is the model's character prediction
    at the exact frame where each keystroke happened?

    Args:
        frame_probs: [C, T] or [T, C] softmax probabilities
        gt_onset_frames: list of frame indices for each GT keystroke
        gt_chars: list of GT characters
        ks: top-k values to evaluate

    Returns:
        dict with char_top1, char_top3, char_top5
    """
    from utils.vocab import CHAR_TO_IDX

    if frame_probs.shape[0] == NUM_CLASSES:
        # [C, T] -> [T, C]
        frame_probs = frame_probs.T

    T, C = frame_probs.shape
    hits = {k: 0 for k in ks}
    total = 0

    for frame_idx, char in zip(gt_onset_frames, gt_chars):
        if frame_idx < 0 or frame_idx >= T:
            continue
        char_lower = char.lower()
        if char_lower not in CHAR_TO_IDX:
            continue

        gt_idx = CHAR_TO_IDX[char_lower]
        probs = frame_probs[frame_idx]
        ranked = np.argsort(-probs)

        for k in ks:
            if gt_idx in ranked[:k]:
                hits[k] += 1
        total += 1

    denom = max(total, 1)
    return {f'char_top{k}': hits[k] / denom for k in ks}


def evaluate_episode(
    reference: str,
    hypothesis: str,
    frame_probs: Optional[np.ndarray] = None,
    gt_onset_frames: Optional[List[int]] = None,
    gt_chars: Optional[List[str]] = None,
    beam_candidates: Optional[List[str]] = None,
    seq_hit_cutoffs: Tuple[int, ...] = (10, 50, 100),
) -> Dict:
    """
    Full evaluation of one episode.

    Args:
        reference: GT password string
        hypothesis: greedy-decoded string
        frame_probs: [C, T] softmax probs (optional, for char_topk)
        gt_onset_frames: GT onset frame indices (optional, for char_topk)
        gt_chars: GT character list (optional, for char_topk)
        beam_candidates: list of beam search results (optional)
        seq_hit_cutoffs: sequence-level hit cutoffs

    Returns:
        dict with cer, char_topk, seq_topk metrics
    """
    result = {
        'reference': reference,
        'hypothesis': hypothesis,
        'cer': cer(reference, hypothesis),
        'ref_len': len(reference),
        'hyp_len': len(hypothesis),
    }

    # Per-position char accuracy at GT onsets
    if frame_probs is not None and gt_onset_frames is not None and gt_chars is not None:
        topk = char_topk_at_gt_positions(
            frame_probs, gt_onset_frames, gt_chars
        )
        result.update(topk)

    # Sequence-level exact match in beam results
    if beam_candidates is not None:
        for cutoff in seq_hit_cutoffs:
            result[f'seq_top{cutoff}'] = int(
                reference in beam_candidates[:cutoff]
            )

    return result


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate per-episode results into summary metrics."""
    n = len(results)
    if n == 0:
        return {}

    total_chars = sum(r['ref_len'] for r in results)
    total_edits = sum(
        levenshtein(r['reference'], r['hypothesis']) for r in results
    )

    agg = {
        'n_episodes': n,
        'n_chars': total_chars,
        'cer': total_edits / max(total_chars, 1),
        'avg_cer': np.mean([r['cer'] for r in results]),
    }

    # Aggregate char_topk (weighted by ref_len)
    for k in (1, 3, 5):
        key = f'char_top{k}'
        if key in results[0]:
            weighted_sum = sum(r[key] * r['ref_len'] for r in results)
            agg[key] = weighted_sum / max(total_chars, 1)

    # Aggregate seq_topk
    for cutoff in (10, 50, 100):
        key = f'seq_top{cutoff}'
        if key in results[0]:
            agg[key] = sum(r[key] for r in results) / n

    return agg
