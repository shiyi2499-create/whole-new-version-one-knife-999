# Stage3 diff12ch Summary (2026-03-27)

## Setup
- Window: pre=100ms, post=200ms
- Norm mode: global
- Input mode: raw + diff1 = 12 channels
- Base train on password len8/len9/len10 free-type windows
- Mixed adapt on mixed_training + mixed_single_training + mixed_retry_training + mixed_single_len9 + mixed_retry_len9
- Holdout sessions: 6 mixed sessions (same holdout set as prior Stage3 ablations)

## Base model
- report: `results/stage3_diff12ch_20260327/inception_password_diff12ch_base_report.json`
- char_top1 = 0.9792156862745098
- CER = 0.020784313725490194

## Mixed holdout
- report: `results/stage3_diff12ch_20260327/inception_password_diff12ch_mixedadapt_report.json`
- zero-shot CER = 0.37681159420289856
- adapted CER = 0.2898550724637681
- historical baseline adapted CER = 0.258427
- interpretation: diff12ch helps relative to zero-shot, but is still worse than the current baseline mixed-adapt Stage3.

## Still probe GT
- report: `results/stage3_diff12ch_20260327/still_probe_gt_eval/report.json`
- baseline fixed CER = 0.33444444444444443
- diff12ch fixed CER = 0.23500000000000001
- baseline overlap CER = 0.3988888888888889
- diff12ch overlap CER = 0.23500000000000001
- fixed exact = 1/5
- overlap exact = 1/5
- interpretation: this is a clear gain on the still-password-still target scenario.

## fair6
- report: `results/stage3_diff12ch_20260327/fair6_eval/report.json`
- baseline gt_keyframes_fixed CER = 0.2696629213483146
- diff12ch gt_keyframes_fixed CER = 0.29213483146067415
- baseline stage1_bestpred_overlap CER = 0.2696629213483146
- diff12ch stage1_bestpred_overlap CER = 0.3258426966292135
- interpretation: diff12ch improves still, but degrades the fair6 mainline automatic path.

## Bottom line
- diff12ch is the first Stage3 change after coarse merge that clearly improves the still-password-still target scenario.
- It does not dominate the existing mainline: fair6 and mixed holdout both get worse relative to the current baseline.
- So this is best understood as a target-scenario optimization, not a universal Stage3 replacement.
