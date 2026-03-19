"""
Evaluation metrics for episode-based pipeline.

Adapted from stage2_open/utils/metrics.py:
  - 'groups' → 'episodes'
  - Frame accuracy is 2-class (silence/typing) instead of 3-class
  - Added episode-level detection metrics (how many GT episodes were found)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict


def match_episodes(pred_episodes: List[Dict], gt_episodes: List[Dict],
                   min_iou: float = 0.1) -> Dict:
    """
    Match predicted episodes to GT episodes via Hungarian on IoU.
    Episodes are dicts with 'start', 'end'.
    """
    n_p = len(pred_episodes)
    n_g = len(gt_episodes)
    res = {'pred_count': n_p, 'gt_count': n_g}

    if n_p == 0 or n_g == 0:
        res.update({'mean_iou': 0.0, 'matches': [],
                     'unmatched_gt': list(range(n_g)),
                     'unmatched_pred': list(range(n_p)),
                     'detection_rate': 0.0})
        return res

    cost = np.zeros((n_p, n_g))
    for i in range(n_p):
        ps, pe = pred_episodes[i]['start'], pred_episodes[i]['end']
        for j in range(n_g):
            gs, ge = gt_episodes[j]['start'], gt_episodes[j]['end']
            inter = max(0, min(pe, ge) - max(ps, gs))
            union = max(pe, ge) - min(ps, gs)
            cost[i, j] = inter / max(union, 1)

    ri, ci = linear_sum_assignment(-cost)
    matches = []
    for r, c in zip(ri, ci):
        if cost[r, c] > min_iou:
            matches.append((r, c, float(cost[r, c])))

    matched_p = {m[0] for m in matches}
    matched_g = {m[1] for m in matches}

    ious = [m[2] for m in matches]
    res['mean_iou'] = float(np.mean(ious)) if ious else 0.0
    res['matches'] = matches
    res['unmatched_gt'] = [j for j in range(n_g) if j not in matched_g]
    res['unmatched_pred'] = [i for i in range(n_p) if i not in matched_p]

    # Episode detection rate: how many GT episodes were matched?
    res['detection_rate'] = len(matches) / max(n_g, 1)
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


def frame_accuracy_2class(pred_labels, gt_labels, mask=None):
    """Per-class and overall frame accuracy for 2-class (silence/typing)."""
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
    names = {0: 'silence', 1: 'typing'}
    for c in range(2):
        idx = gt == c
        if idx.sum() > 0:
            per_class[names[c]] = float((pred[idx] == gt[idx]).mean())
        else:
            per_class[names[c]] = float('nan')

    return {'overall': overall, 'per_class': per_class}


def full_eval(pred_episodes, gt_episodes, sr=100, tol_ms=50):
    """
    Full evaluation: episode matching → per-matched-episode onset metrics.
    """
    tol = int(tol_ms / 1000 * sr)
    em = match_episodes(pred_episodes, gt_episodes)

    onset_results = []
    for pi, gi, iou in em['matches']:
        om = onset_metrics(pred_episodes[pi]['onsets'],
                          gt_episodes[gi]['onsets'], tol, sr)
        om['episode_iou'] = iou
        om['pred_nkeys'] = pred_episodes[pi].get('num_keys',
                                                  len(pred_episodes[pi]['onsets']))
        om['gt_nkeys'] = gt_episodes[gi].get('num_keys',
                                              len(gt_episodes[gi]['onsets']))
        onset_results.append(om)

    return {
        'episode_match': em,
        'onset_per_episode': onset_results,
        'avg_onset_f1': float(np.mean([o['f1'] for o in onset_results])) if onset_results else 0,
        'avg_onset_recall': float(np.mean([o['recall'] for o in onset_results])) if onset_results else 0,
        'avg_episode_iou': em['mean_iou'],
        'episode_detection_rate': em['detection_rate'],
    }


def format_report(eval_result: Dict, frame_acc: Dict = None) -> str:
    lines = ["=" * 60, "EPISODE-BASED STAGE 2 EVALUATION", "=" * 60]
    em = eval_result['episode_match']
    lines.append(f"\nEpisodes: pred={em['pred_count']}  gt={em['gt_count']}  "
                 f"matched={len(em['matches'])}  "
                 f"detection_rate={em['detection_rate']:.3f}  "
                 f"IoU={em['mean_iou']:.3f}")
    lines.append(f"  unmatched GT: {em['unmatched_gt']}")
    lines.append(f"  unmatched pred: {em['unmatched_pred']}")

    lines.append("\nPer-episode onset metrics:")
    for i, om in enumerate(eval_result['onset_per_episode']):
        lines.append(f"  Ep {i}: IoU={om['episode_iou']:.3f}  "
                     f"keys pred/gt={om['pred_nkeys']}/{om['gt_nkeys']}  "
                     f"F1={om['f1']:.3f} R={om['recall']:.3f} P={om['precision']:.3f} "
                     f"MAE={om['mae_ms']:.1f}ms")

    lines.append(f"\nAvg onset F1: {eval_result['avg_onset_f1']:.3f}")
    lines.append(f"Avg onset recall: {eval_result['avg_onset_recall']:.3f}")
    lines.append(f"Episode detection rate: {eval_result['episode_detection_rate']:.3f}")

    if frame_acc:
        lines.append(f"\nFrame accuracy: {frame_acc['overall']:.4f}")
        for name, v in frame_acc.get('per_class', {}).items():
            lines.append(f"  {name}: {v:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)
