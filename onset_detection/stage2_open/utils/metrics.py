"""
Evaluation metrics for open/variable-length pipeline.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional


def match_groups(pred_groups: List[Dict], gt_groups: List[Dict]) -> Dict:
    """
    Match predicted groups to GT groups via Hungarian on IoU.
    Groups are dicts with 'start', 'end'.
    """
    n_p = len(pred_groups)
    n_g = len(gt_groups)
    res = {'pred_count': n_p, 'gt_count': n_g}

    if n_p == 0 or n_g == 0:
        res.update({'mean_iou': 0.0, 'matches': [], 'unmatched_gt': list(range(n_g)),
                     'unmatched_pred': list(range(n_p))})
        return res

    cost = np.zeros((n_p, n_g))
    for i in range(n_p):
        ps, pe = pred_groups[i]['start'], pred_groups[i]['end']
        for j in range(n_g):
            gs, ge = gt_groups[j]['start'], gt_groups[j]['end']
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            cost[i, j] = inter / max(union, 1)

    ri, ci = linear_sum_assignment(-cost)
    matches = []
    for r, c in zip(ri, ci):
        if cost[r, c] > 0.1:  # minimum IoU to count as match
            matches.append((r, c, float(cost[r, c])))

    matched_p = {m[0] for m in matches}
    matched_g = {m[1] for m in matches}

    ious = [m[2] for m in matches]
    res['mean_iou'] = float(np.mean(ious)) if ious else 0.0
    res['matches'] = matches
    res['unmatched_gt'] = [j for j in range(n_g) if j not in matched_g]
    res['unmatched_pred'] = [i for i in range(n_p) if i not in matched_p]
    return res


def onset_metrics(pred_onsets, gt_onsets, tol_samples=5, sr=100):
    """Precision/recall/F1 for onset detection with tolerance."""
    pred = np.sort(np.array(pred_onsets, dtype=np.int64))
    gt = np.sort(np.array(gt_onsets, dtype=np.int64))

    if len(pred) == 0 or len(gt) == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0,
                'mae_ms': float('inf'), 'n_pred': len(pred), 'n_gt': len(gt)}

    dist = np.abs(pred[:, None] - gt[None, :])
    ri, ci = linear_sum_assignment(dist)
    tp = sum(1 for r, c in zip(ri, ci) if dist[r, c] <= tol_samples)
    errors = [dist[r, c] for r, c in zip(ri, ci) if dist[r, c] <= tol_samples]

    p = tp / len(pred) if pred.size else 0
    r = tp / len(gt) if gt.size else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    mae = float(np.mean(errors)) / sr * 1000 if errors else float('inf')

    return {'precision': p, 'recall': r, 'f1': f1, 'mae_ms': mae,
            'n_pred': len(pred), 'n_gt': len(gt), 'n_matched': tp}


def frame_accuracy(pred_labels, gt_labels, mask=None):
    """Per-class and overall frame accuracy."""
    pred = np.array(pred_labels)
    gt = np.array(gt_labels)
    if mask is not None:
        m = np.array(mask, dtype=bool)
        pred = pred[m]
        gt = gt[m]
    if len(gt) == 0:
        return {}

    overall = float((pred == gt).mean())
    per_class = {}
    for c in range(3):
        idx = gt == c
        if idx.sum() > 0:
            per_class[c] = float((pred[idx] == gt[idx]).mean())
        else:
            per_class[c] = float('nan')

    return {'overall': overall, 'per_class': per_class}


def full_eval(pred_groups, gt_groups, sr=100, tol_ms=50):
    """
    Full evaluation: group matching → per-matched-group onset metrics.
    pred_groups/gt_groups: list of dicts with 'start', 'end', 'onsets'.
    """
    tol = int(tol_ms / 1000 * sr)
    gm = match_groups(pred_groups, gt_groups)

    onset_results = []
    for pi, gi, iou in gm['matches']:
        om = onset_metrics(pred_groups[pi]['onsets'], gt_groups[gi]['onsets'], tol, sr)
        om['group_iou'] = iou
        om['pred_nkeys'] = pred_groups[pi].get('num_keys', len(pred_groups[pi]['onsets']))
        om['gt_nkeys'] = gt_groups[gi].get('num_keys', len(gt_groups[gi]['onsets']))
        onset_results.append(om)

    return {
        'group_match': gm,
        'onset_per_group': onset_results,
        'avg_onset_f1': float(np.mean([o['f1'] for o in onset_results])) if onset_results else 0,
        'avg_onset_recall': float(np.mean([o['recall'] for o in onset_results])) if onset_results else 0,
        'avg_group_iou': gm['mean_iou'],
    }


def format_report(eval_result: Dict, frame_acc: Dict = None) -> str:
    lines = ["=" * 60, "OPEN STAGE 2 EVALUATION", "=" * 60]
    gm = eval_result['group_match']
    lines.append(f"\nGroups: pred={gm['pred_count']}  gt={gm['gt_count']}  "
                 f"matched={len(gm['matches'])}  IoU={gm['mean_iou']:.3f}")
    lines.append(f"  unmatched GT: {gm['unmatched_gt']}")
    lines.append(f"  unmatched pred: {gm['unmatched_pred']}")

    lines.append("\nPer-group onset metrics:")
    for i, om in enumerate(eval_result['onset_per_group']):
        lines.append(f"  Group {i}: IoU={om['group_iou']:.3f}  "
                     f"keys pred/gt={om['pred_nkeys']}/{om['gt_nkeys']}  "
                     f"F1={om['f1']:.3f} R={om['recall']:.3f} P={om['precision']:.3f} "
                     f"MAE={om['mae_ms']:.1f}ms")

    lines.append(f"\nAvg onset F1: {eval_result['avg_onset_f1']:.3f}")
    lines.append(f"Avg onset recall: {eval_result['avg_onset_recall']:.3f}")

    if frame_acc:
        lines.append(f"\nFrame accuracy: {frame_acc['overall']:.4f}")
        for c, name in [(0, 'gap'), (1, 'keystroke'), (2, 'separator')]:
            v = frame_acc['per_class'].get(c, float('nan'))
            lines.append(f"  {name}: {v:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)
