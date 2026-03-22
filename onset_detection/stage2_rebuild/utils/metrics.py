"""
Evaluation metrics for the full pipeline.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.optimize import linear_sum_assignment


# ============================================================
# Stage 2A Metrics: Group Segmentation
# ============================================================

def compute_group_iou(pred_groups: List[Tuple[int, int]],
                      gt_groups: List[Tuple[int, int]]) -> Dict:
    """
    Compute IoU between predicted and GT groups using Hungarian matching.

    Args:
        pred_groups: list of (start, end) in samples
        gt_groups: list of (start, end) in samples

    Returns:
        dict with 'mean_iou', 'per_group_iou', 'group_count_acc',
        'mean_boundary_error_samples'
    """
    n_pred = len(pred_groups)
    n_gt = len(gt_groups)

    result = {
        'pred_count': n_pred,
        'gt_count': n_gt,
        'group_count_correct': n_pred == n_gt,
    }

    if n_pred == 0 or n_gt == 0:
        result['mean_iou'] = 0.0
        result['per_group_iou'] = []
        result['mean_boundary_error_samples'] = float('inf')
        return result

    # Build IoU cost matrix
    cost_matrix = np.zeros((n_pred, n_gt))
    for i, (ps, pe) in enumerate(pred_groups):
        for j, (gs, ge) in enumerate(gt_groups):
            intersection = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            cost_matrix[i, j] = intersection / max(union, 1)

    # Hungarian matching (maximize IoU = minimize -IoU)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    ious = [cost_matrix[r, c] for r, c in zip(row_ind, col_ind)]
    boundary_errors = []
    for r, c in zip(row_ind, col_ind):
        ps, pe = pred_groups[r]
        gs, ge = gt_groups[c]
        boundary_errors.append(abs(ps - gs))
        boundary_errors.append(abs(pe - ge))

    result['mean_iou'] = float(np.mean(ious)) if ious else 0.0
    result['per_group_iou'] = ious
    result['mean_boundary_error_samples'] = float(np.mean(boundary_errors)) if boundary_errors else float('inf')

    return result


# ============================================================
# Stage 2B Metrics: Onset Detection
# ============================================================

def compute_onset_metrics(pred_onsets: np.ndarray,
                          gt_onsets: np.ndarray,
                          tolerance_samples: int = 5,
                          sample_rate: int = 100
                          ) -> Dict:
    """
    Compute onset detection precision, recall, F1 at a given tolerance.

    Args:
        pred_onsets: array of predicted onset sample indices
        gt_onsets: array of GT onset sample indices
        tolerance_samples: matching tolerance in samples
        sample_rate: for converting to ms in output

    Returns:
        dict with precision, recall, f1, mean_abs_error_ms
    """
    pred_onsets = np.sort(np.array(pred_onsets))
    gt_onsets = np.sort(np.array(gt_onsets))

    tolerance_ms = tolerance_samples / sample_rate * 1000

    if len(pred_onsets) == 0 or len(gt_onsets) == 0:
        return {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'mean_abs_error_ms': float('inf'),
            'tolerance_ms': tolerance_ms,
            'n_pred': len(pred_onsets), 'n_gt': len(gt_onsets),
        }

    # Greedy matching: for each GT onset, find nearest unmatched pred
    matched_pred = set()
    matched_gt = set()
    errors = []

    # Build distance matrix and use Hungarian
    dist_matrix = np.abs(pred_onsets[:, None] - gt_onsets[None, :])
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= tolerance_samples:
            matched_pred.add(r)
            matched_gt.add(c)
            errors.append(dist_matrix[r, c])

    tp = len(matched_gt)
    precision = tp / len(pred_onsets) if len(pred_onsets) > 0 else 0.0
    recall = tp / len(gt_onsets) if len(gt_onsets) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_error_ms = float(np.mean(errors)) / sample_rate * 1000 if errors else float('inf')

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_abs_error_ms': mean_error_ms,
        'tolerance_ms': tolerance_ms,
        'n_pred': len(pred_onsets),
        'n_gt': len(gt_onsets),
        'n_matched': tp,
    }


# ============================================================
# E2E Metrics
# ============================================================

def compute_e2e_metrics(pred_chars: List[List[str]],
                        gt_chars: List[List[str]],
                        top_k_preds: Optional[List[List[List[str]]]] = None
                        ) -> Dict:
    """
    Compute end-to-end character-level metrics.

    Args:
        pred_chars: list of 5 lists, each with 8 predicted characters
        gt_chars: list of 5 lists, each with 8 GT characters
        top_k_preds: optional, list of 5 lists of 8 lists of top-k predictions

    Returns:
        dict with char_top1, char_top3, char_top5, CER, exact_match_count
    """
    assert len(pred_chars) == len(gt_chars), \
        f"Mismatch: {len(pred_chars)} pred vs {len(gt_chars)} gt groups"

    total_chars = 0
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    exact_matches = 0

    for g in range(len(gt_chars)):
        gt_group = gt_chars[g]
        pred_group = pred_chars[g]

        group_correct = True
        for k in range(min(len(gt_group), len(pred_group))):
            total_chars += 1
            gt_c = gt_group[k].lower()
            pred_c = pred_group[k].lower()

            if pred_c == gt_c:
                correct_top1 += 1
            else:
                group_correct = False

            # Top-k
            if top_k_preds is not None and g < len(top_k_preds) and k < len(top_k_preds[g]):
                topk = [c.lower() for c in top_k_preds[g][k]]
                if gt_c in topk[:3]:
                    correct_top3 += 1
                if gt_c in topk[:5]:
                    correct_top5 += 1
            else:
                if pred_c == gt_c:
                    correct_top3 += 1
                    correct_top5 += 1

        if group_correct and len(gt_group) == len(pred_group):
            exact_matches += 1

    n = max(total_chars, 1)
    cer = 1.0 - correct_top1 / n

    return {
        'char_top1': correct_top1 / n,
        'char_top3': correct_top3 / n,
        'char_top5': correct_top5 / n,
        'CER': cer,
        'exact_match_count': exact_matches,
        'total_groups': len(gt_chars),
        'total_chars': total_chars,
    }


def compute_full_report(stage2a_metrics: Dict,
                        stage2b_metrics_per_group: List[Dict],
                        e2e_metrics: Dict,
                        sample_rate: int = 100) -> str:
    """Format a readable evaluation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("FULL PIPELINE EVALUATION REPORT")
    lines.append("=" * 60)

    lines.append("\n--- Stage 2A: Group Segmentation ---")
    lines.append(f"  Predicted groups: {stage2a_metrics['pred_count']}")
    lines.append(f"  GT groups:        {stage2a_metrics['gt_count']}")
    lines.append(f"  Count correct:    {stage2a_metrics['group_count_correct']}")
    lines.append(f"  Mean IoU:         {stage2a_metrics['mean_iou']:.4f}")
    bnd_ms = stage2a_metrics['mean_boundary_error_samples'] / sample_rate * 1000
    lines.append(f"  Mean boundary err: {bnd_ms:.1f} ms")

    lines.append("\n--- Stage 2B: Onset Detection (per group) ---")
    for i, m in enumerate(stage2b_metrics_per_group):
        lines.append(f"  Group {i}: P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f} MAE={m['mean_abs_error_ms']:.1f}ms "
                      f"({m['n_matched']}/{m['n_gt']} matched)")

    if stage2b_metrics_per_group:
        avg_f1 = np.mean([m['f1'] for m in stage2b_metrics_per_group])
        avg_recall = np.mean([m['recall'] for m in stage2b_metrics_per_group])
        lines.append(f"  Average: F1={avg_f1:.3f} Recall={avg_recall:.3f}")

    lines.append("\n--- E2E: Character Recovery ---")
    lines.append(f"  char_top1:     {e2e_metrics['char_top1']:.4f} ({e2e_metrics['char_top1']*100:.1f}%)")
    lines.append(f"  char_top3:     {e2e_metrics.get('char_top3', 0):.4f}")
    lines.append(f"  char_top5:     {e2e_metrics.get('char_top5', 0):.4f}")
    lines.append(f"  CER:           {e2e_metrics['CER']:.4f} ({e2e_metrics['CER']*100:.1f}%)")
    lines.append(f"  Exact matches: {e2e_metrics['exact_match_count']}/{e2e_metrics['total_groups']}")

    lines.append("=" * 60)
    return "\n".join(lines)
