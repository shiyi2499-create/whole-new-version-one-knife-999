# Stage3 Per-Window Normalization Status 2026-03-27

## Goal
Test whether replacing Stage3 global channel-scaler normalization with per-window z-normalization would reduce the residual still-password-still resolution gap after Stage1 coarse merge.

## Change
- Keep Stage3 window at 100ms / 200ms
- Keep model architecture unchanged
- Replace global normalization with per-window normalization during:
  - base training
  - mixed adaptation
  - inference / eval
- Keep scaler files for backward compatibility
- New checkpoints store `norm_mode=per_window`

## Results

### still probe GT
- baseline fixed CER: 0.334444
- per-window fixed CER: 0.531667
- baseline overlap CER: 0.398889
- per-window overlap CER: 0.546111

### mixed holdout
- historical baseline adapted CER: 0.258427
- per-window adapted CER: 0.434783

### fair6
- baseline gt_keyframes_fixed CER: 0.269663
- per-window gt_keyframes_fixed CER: 0.438202
- baseline stage1_bestpred_overlap CER: 0.269663
- per-window stage1_bestpred_overlap CER: 0.449438

## Interpretation
This is a negative result. Per-window normalization hurts both the original mixed/fair6 distribution and the still-password-still probe setting.

## Current conclusion
- Stage1 coarse merge remains useful
- Wider fixed windows were not helpful
- Per-window z-normalization is also not helpful
- The next Stage3 step should target local neighbor-key resolution more directly, not broader context or normalization changes
