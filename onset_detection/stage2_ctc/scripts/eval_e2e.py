#!/usr/bin/env python3
"""
End-to-end evaluation of the Frame-CTC model on real mixed_training / mixed2 sessions.

For each session:
  1. Extract password episodes from activity log + key events
  2. Run frame-level CTC model on each episode's IMU
  3. Greedy + beam search decode
  4. Evaluate CER, char_topk (at GT onset positions), sequence hit rate

Also computes the "GT oracle" comparison: char_topk at GT positions shows how
well the model discriminates characters when you look at the right frames.

Usage:
    python scripts/eval_e2e.py \
        --mixed_dir data/raw/mixed_training \
        --checkpoint runs/stage2_ctc/best.pt \
        --output_dir results/stage2_ctc_eval \
        --device mps
"""
import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
REPO_ROOT = os.path.dirname(os.path.dirname(PKG_ROOT))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.frame_ctc import FrameCTCModel
from configs.config import ModelConfig, SignalConfig
from data.loaders import SessionLoader, discover_sessions
from utils.signal_processing import preprocess
from utils.decode import greedy_decode, prefix_beam_search, rhythm_constrained_decode
from utils.metrics import (levenshtein, cer, char_topk_at_gt_positions,
                           evaluate_episode, aggregate_results)
from utils.vocab import CHAR_TO_IDX, IDX_TO_CHAR, BLANK_IDX, NUM_CLASSES
from phase3_password_inception.run_password_closure_inception import (
    load_final_inception,
    topk_strings_from_prob_vectors,
)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg_dict = ckpt.get('model_cfg', {})
    if isinstance(mcfg_dict, dict):
        mcfg = ModelConfig(**mcfg_dict)
    else:
        mcfg = mcfg_dict  # already a ModelConfig

    model = FrameCTCModel(
        in_ch=mcfg.input_channels,
        hidden=mcfg.hidden_channels,
        num_layers=mcfg.num_layers,
        kernel=mcfg.kernel_size,
        dropout=mcfg.dropout,
        num_classes=mcfg.num_classes,
    ).to(device)

    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model from {ckpt_path} "
          f"(ep={ckpt.get('epoch', '?')}, "
          f"val_cer={ckpt.get('val_cer', '?')})")
    return model, mcfg


def run_episode(model, episode_imu, device, sample_rate=100,
                beam_width=50,
                rhythm_median_iki_frames: float = 258.0,
                rhythm_min_chars: int = 4,
                rhythm_max_chars: int = 12,
                run_beam: bool = True):
    """
    Run model on one episode and decode.

    Args:
        model: FrameCTCModel
        episode_imu: [T, 6] raw IMU
        device: torch device
        beam_width: beam search width

    Returns:
        dict with greedy, beam results, and frame posteriors
    """
    proc, _ = preprocess(episode_imu, sample_rate, add_mag=True, norm=True)
    x = torch.from_numpy(proc.T).float().unsqueeze(0).to(device)  # [1, C, T]

    with torch.no_grad():
        logits = model(x)  # [1, num_classes, T]
        log_probs = F.log_softmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

    lp = log_probs[0].cpu().numpy()   # [C, T]
    sp = probs[0].cpu().numpy()        # [C, T]

    # Greedy decode
    hyp_greedy = greedy_decode(lp)

    # Beam search (optional; expensive on long episodes)
    if run_beam and beam_width > 0:
        beam_results = prefix_beam_search(lp, beam_width=beam_width)
        hyp_beam = beam_results[0]['candidate'] if beam_results else ''
        beam_candidates = [r['candidate'] for r in beam_results]
    else:
        hyp_beam = ''
        beam_candidates = []

    # Rhythm-constrained decode from frame posterior
    constrained = rhythm_constrained_decode(
        sp,
        median_iki_frames=rhythm_median_iki_frames,
        min_chars=rhythm_min_chars,
        max_chars=rhythm_max_chars,
    )
    hyp_constrained = constrained['candidate']

    return {
        'greedy': hyp_greedy,
        'beam_top1': hyp_beam,
        'beam_candidates': beam_candidates,
        'constrained_top1': hyp_constrained,
        'constrained_debug': constrained,
        'frame_probs': sp,      # [C, T] softmax
        'frame_log_probs': lp,  # [C, T] log-softmax
    }


def _extract_classifier_window(
    episode_imu: np.ndarray,
    center_frame: int,
    sample_rate: int,
    pre_ms: float,
    post_ms: float,
    target_len: int,
):
    pre_frames = int(round(pre_ms / 1000.0 * sample_rate))
    post_frames = int(round(post_ms / 1000.0 * sample_rate))
    lo = max(0, int(center_frame) - pre_frames)
    hi = min(len(episode_imu), int(center_frame) + post_frames)
    if hi - lo < 2:
        return None
    window = episode_imu[lo:hi].astype(np.float32)
    out = resample(window, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def _refine_with_classifier(
    episode_imu: np.ndarray,
    positions: list[int],
    frame_probs: np.ndarray,
    cls_model,
    cls_classes: np.ndarray,
    cls_means: np.ndarray,
    cls_stds: np.ndarray,
    cls_device: torch.device,
    sample_rate: int,
    target_len: int,
    shift_radius_frames: int = 8,
    shift_step_frames: int = 2,
    branch_topk: int = 5,
    beam_width: int = 100,
    cls_weight: float = 1.0,
    frame_weight: float = 0.35,
):
    prob_vectors = []
    debug = []

    if frame_probs.shape[0] == NUM_CLASSES:
        probs_tc = frame_probs.T
    else:
        probs_tc = frame_probs

    for pos in positions:
        best_shift = 0
        best_vec = None
        best_score = -1e18
        frame_local = probs_tc[max(0, pos - shift_radius_frames): min(len(probs_tc), pos + shift_radius_frames + 1), 1:]
        frame_local_best = np.max(frame_local, axis=0) if len(frame_local) else np.zeros(NUM_CLASSES - 1, dtype=np.float32)

        for shift in range(-shift_radius_frames, shift_radius_frames + 1, max(1, shift_step_frames)):
            center = pos + shift
            window = _extract_classifier_window(
                episode_imu,
                center_frame=center,
                sample_rate=sample_rate,
                pre_ms=100.0,
                post_ms=200.0,
                target_len=target_len,
            )
            if window is None:
                continue

            w = window.copy()
            for ch in range(w.shape[1]):
                w[:, ch] = (w[:, ch] - cls_means[ch]) / (cls_stds[ch] + 1e-10)
            with torch.no_grad():
                xb = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(cls_device)
                logits = cls_model(xb)
                cls_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            fused = np.log(np.maximum(cls_probs, 1e-8)).astype(np.float32)
            for i, ch in enumerate(cls_classes.tolist()):
                frame_idx = CHAR_TO_IDX.get(str(ch))
                if frame_idx is None or frame_idx == BLANK_IDX:
                    continue
                fused[i] = (
                    cls_weight * np.log(max(float(cls_probs[i]), 1e-8))
                    + frame_weight * np.log(max(float(frame_local_best[frame_idx - 1]), 1e-8))
                )

            score = float(np.max(fused))
            if score > best_score:
                best_score = score
                best_shift = shift
                best_vec = fused

        if best_vec is None:
            continue

        # convert fused log-scores back into a pseudo-probability vector for beaming
        fused_probs = np.exp(best_vec - np.max(best_vec))
        fused_probs = fused_probs / np.maximum(fused_probs.sum(), 1e-12)
        prob_vectors.append(fused_probs.astype(np.float32))

        top_idx = np.argsort(fused_probs)[::-1][:branch_topk]
        debug.append({
            'pos': int(pos),
            'best_shift': int(best_shift),
            'top_chars': [
                {'char': str(cls_classes[i]), 'score': float(fused_probs[i])}
                for i in top_idx
            ],
        })

    if not prob_vectors:
        return {
            'candidate': '',
            'candidates': [],
            'debug': debug,
        }

    candidates = topk_strings_from_prob_vectors(
        prob_vectors,
        cls_classes,
        branch_topk=branch_topk,
        beam_width=beam_width,
    )
    return {
        'candidate': candidates[0]['candidate'] if candidates else '',
        'candidates': candidates,
        'debug': debug,
    }


def extract_episodes(session_path: str, sample_rate: int = 100,
                     margin_ms: float = 300.0):
    """
    Extract password episodes from a session.

    Returns list of dicts with:
        imu: [T, 6]
        gt_chars: list of str
        gt_onset_frames: list of int (episode-local)
        password: str
    """
    loader = SessionLoader(session_path)
    ts, imu = loader.get_imu()
    if len(ts) == 0:
        return []

    groups = loader.split_password_groups_from_enters()
    if not groups:
        return []

    margin_frames = int(margin_ms / 1000.0 * sample_rate)
    episodes = []

    for group in groups:
        keys = group['keys']
        if not keys:
            continue

        key_frames = []
        key_chars = []
        for event in keys:
            fi = int(np.searchsorted(ts, event['ts']))
            fi = min(max(fi, 0), len(ts) - 1)
            key_frames.append(fi)
            key_chars.append(event['key'].lower())

        ep_start = max(0, key_frames[0] - margin_frames)
        ep_end = min(len(ts), key_frames[-1] + margin_frames)

        episode_imu = imu[ep_start:ep_end].astype(np.float32)
        if len(episode_imu) < 10:
            continue

        # Local frame indices
        local_frames = [f - ep_start for f in key_frames]
        password = ''.join(c for c in key_chars
                          if c not in ('enter', 'return', 'backspace'))

        episodes.append({
            'imu': episode_imu,
            'gt_chars': key_chars,
            'gt_onset_frames': local_frames,
            'password': password,
        })

    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mixed_dir', required=True,
                    help='mixed_training or mixed2 directory')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--beam_width', type=int, default=50)
    ap.add_argument('--margin_ms', type=float, default=300.0)
    ap.add_argument('--decode_mode', choices=('beam', 'constrained'),
                    default='constrained')
    ap.add_argument('--median_iki_frames', type=float, default=258.0)
    ap.add_argument('--min_chars', type=int, default=4)
    ap.add_argument('--max_chars', type=int, default=12)
    ap.add_argument('--classifier_checkpoint', default='')
    ap.add_argument('--classifier_scaler', default='')
    ap.add_argument('--classifier_shift_radius_frames', type=int, default=8)
    ap.add_argument('--classifier_shift_step_frames', type=int, default=2)
    ap.add_argument('--classifier_branch_topk', type=int, default=5)
    ap.add_argument('--classifier_beam_width', type=int, default=100)
    ap.add_argument('--classifier_cls_weight', type=float, default=1.0)
    ap.add_argument('--classifier_frame_weight', type=float, default=0.35)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    model, mcfg = load_model(args.checkpoint, device)
    cls_bundle = None
    if args.classifier_checkpoint and args.classifier_scaler:
        cls_ckpt = torch.load(args.classifier_checkpoint, map_location='cpu', weights_only=False)
        cls_model, cls_classes, cls_means, cls_stds = load_final_inception(
            args.classifier_checkpoint, args.classifier_scaler, device
        )
        cls_bundle = (
            cls_model, cls_classes, cls_means, cls_stds, device,
            int(cls_ckpt['n_timesteps'])
        )
        print(f"Loaded auxiliary classifier ({len(cls_classes)} classes)")

    sessions = discover_sessions(args.mixed_dir)
    print(f"Found {len(sessions)} sessions\n")

    all_results = []
    session_outputs = []

    for si, sp in enumerate(sessions, 1):
        name = Path(sp).name
        print(f"--- Session {si}: {name} ---")

        episodes = extract_episodes(sp, args.sample_rate, args.margin_ms)
        if not episodes:
            print("  (no episodes)")
            continue

        ep_results = []
        for ei, ep in enumerate(episodes):
            decoded = run_episode(
                model, ep['imu'], device,
                sample_rate=args.sample_rate,
                beam_width=args.beam_width,
                rhythm_median_iki_frames=args.median_iki_frames,
                rhythm_min_chars=args.min_chars,
                rhythm_max_chars=args.max_chars,
                run_beam=(args.decode_mode == 'beam'),
            )

            ref = ep['password']
            hyp = (
                decoded['constrained_top1']
                if args.decode_mode == 'constrained'
                else decoded['beam_top1']
            )
            classifier_refine = None
            if cls_bundle is not None and args.decode_mode == 'constrained':
                positions = decoded['constrained_debug'].get('positions', [])
                if positions:
                    cls_model, cls_classes, cls_means, cls_stds, cls_device, cls_target_len = cls_bundle
                    classifier_refine = _refine_with_classifier(
                        ep['imu'],
                        positions,
                        decoded['frame_probs'],
                        cls_model,
                        cls_classes,
                        cls_means,
                        cls_stds,
                        cls_device,
                        sample_rate=args.sample_rate,
                        target_len=cls_target_len,
                        shift_radius_frames=args.classifier_shift_radius_frames,
                        shift_step_frames=args.classifier_shift_step_frames,
                        branch_topk=args.classifier_branch_topk,
                        beam_width=args.classifier_beam_width,
                        cls_weight=args.classifier_cls_weight,
                        frame_weight=args.classifier_frame_weight,
                    )
                    if classifier_refine.get('candidate'):
                        hyp = classifier_refine['candidate']

            # Filter GT chars for evaluation (remove enter/backspace)
            eval_chars = [c for c in ep['gt_chars']
                         if c not in ('enter', 'return', 'backspace')]
            eval_frames = [f for f, c in zip(ep['gt_onset_frames'], ep['gt_chars'])
                          if c not in ('enter', 'return', 'backspace')]

            result = evaluate_episode(
                reference=ref,
                hypothesis=hyp,
                frame_probs=decoded['frame_probs'],
                gt_onset_frames=eval_frames,
                gt_chars=eval_chars,
                beam_candidates=decoded['beam_candidates'],
            )
            result['greedy'] = decoded['greedy']
            result['beam_top1'] = decoded['beam_top1']
            result['constrained_top1'] = decoded['constrained_top1']
            result['constrained_debug'] = decoded['constrained_debug']
            if classifier_refine is not None:
                result['classifier_refine'] = classifier_refine

            ep_results.append(result)
            all_results.append(result)

            print(f"  ep{ei+1}: ref={ref}  hyp={hyp}  greedy={decoded['greedy']}  "
                  f"CER={result['cer']:.2f}")

        session_out = {
            'session': name,
            'num_episodes': len(episodes),
            'episodes': ep_results,
        }
        session_outputs.append(session_out)

        with open(out / f"{name}_results.json", 'w') as f:
            json.dump(session_out, f, indent=2, default=str)

    # ── Aggregate ──
    if all_results:
        agg = aggregate_results(all_results)
        agg['n_sessions'] = len(session_outputs)

        print(f"\n{'='*60}")
        print(f"AGGREGATE ({agg['n_sessions']} sessions, "
              f"{agg['n_episodes']} episodes, {agg['n_chars']} chars)")
        print(f"{'='*60}")
        print(f"  CER:       {agg['cer']*100:.1f}%")
        if 'char_top1' in agg:
            print(f"  char_top1: {agg['char_top1']*100:.1f}%")
            print(f"  char_top3: {agg['char_top3']*100:.1f}%")
            print(f"  char_top5: {agg['char_top5']*100:.1f}%")
        if 'seq_top10' in agg:
            print(f"  seq_top10:  {agg['seq_top10']*100:.1f}%")
            print(f"  seq_top50:  {agg['seq_top50']*100:.1f}%")
            print(f"  seq_top100: {agg['seq_top100']*100:.1f}%")

        with open(out / 'aggregate_results.json', 'w') as f:
            json.dump(agg, f, indent=2)
        print(f"\nSaved -> {out}")
    else:
        print("No results to aggregate.")


if __name__ == '__main__':
    main()
