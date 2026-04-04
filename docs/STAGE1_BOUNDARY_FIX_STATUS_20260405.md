# Stage1 Boundary Fix Status — April 5, 2026

## Current Solution: Approach D (inference-time padding)

After Stage1 posthoc segmentation, extend each detected segment by 0.8s on both sides
before passing to Stage2/Stage3.

Result on 5-sample still-probe:
- CER: 0.0222 (1 char error in 45 total chars)
- Exact match: 4/5
- Top-100: 5/5
- Matches the native GT-assisted CER on this set

## Why This Works

Stage1 two-stage model (clean baseline -> mixed adaptation) detects correct segments
with pmax=1.0 and correct durations, but boundaries are around 0.5-0.7s too tight,
clipping 1 boundary keystroke in 3/5 samples.

The remaining full-auto gap was not pure Stage1 failure:

- native two-stage full auto: CER 0.1989
- same Stage1 crop + GT Stage2: CER 0.0917
- same pipeline + 0.8s segment padding: CER 0.0222

So the practical conclusion is:

- Stage2 remains one major bottleneck
- Stage1 boundary/context tightness is also a confirmed major bottleneck
- 0.8s inference padding resolves most of the still-probe error immediately

## Future Improvement: Approach A (wider label padding, retraining)

For a cleaner model-based fix:

1. Increase label padding:
   - Recommended first try: pre_pad_ms=800, post_pad_ms=1200
   - Conservative: pre_pad_ms=500, post_pad_ms=800
   - Aggressive: pre_pad_ms=1200, post_pad_ms=1500

2. Retrain two-step chain:
   - Step 1: clean baseline (800Hz password + onset_negative), random init, depth=6, trainmax=34000
   - Step 2: mixed adaptation from Step 1 checkpoint, 27 sessions (21/6 split)

3. Evaluate on still-probe without any inference-time padding
   - Target: CER <= 0.0222 without post-hoc segment padding

## Key Files

- Training script:
  - `onset_detection/stage2_segmental/scripts/train_eval_stage1_dense_labeling.py`
- Current checkpoint:
  - `results/800hz_stage1_v14_adapt_from_clean_baseline_depth6_trainmax34000_20260405`
- onset_negative data:
  - `data/raw/800hz/onset_negative_workstation`
- mixed split:
  - `data/raw/800hz_stage1_retry20plus7_split_20260403/train_sessions.txt`
  - `data/raw/800hz_stage1_retry20plus7_split_20260403/val_sessions.txt`
- experimental native manifest:
  - `demo_inference_api/inference/checkpoints_800hz_fullauto_stage12_twostage/CHECKPOINT_MANIFEST.json`

## Other Approaches Considered

- Approach B: soft/tapered labels at boundaries
  - not pursued yet; more complex and unnecessary given D results
- Approach C: lower posthoc threshold
  - not pursued yet; D was sufficient to validate the boundary/context hypothesis
