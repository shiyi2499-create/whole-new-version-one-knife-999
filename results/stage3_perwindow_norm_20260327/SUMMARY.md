# Stage3 Per-Window Normalization Summary

## Setup
- Window: 100ms / 200ms (unchanged)
- Change: Stage3 training + adaptation + inference switched from global channel scaler normalization to per-window z-normalization
- New checkpoints store `norm_mode=per_window`

## Key results

### still probe GT
- baseline fixed CER: 0.334444
- per-window fixed CER: 0.531667
- baseline overlap CER: 0.398889
- per-window overlap CER: 0.546111

### mixed holdout adaptation
- historical baseline adapted CER: 0.258427
- per-window adapted CER: 0.434783

### fair6
- baseline gt_keyframes_fixed CER: 0.269663
- per-window gt_keyframes_fixed CER: 0.438202
- baseline stage1_bestpred_overlap CER: 0.269663
- per-window stage1_bestpred_overlap CER: 0.449438

## Conclusion
Per-window z-normalization is not the fix for the residual Stage3 resolution gap.
It significantly degrades still probe, mixed holdout, and fair6.
