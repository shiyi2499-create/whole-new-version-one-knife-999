# Stage3 Window Ablation Status 2026-03-27

## Goal
Test whether wider Stage3 windows improve residual neighbor-key / resolution errors after Stage1 coarse-merge stabilized burst selection.

## Configs
- Baseline: pre=100ms, post=200ms, n_timesteps=57
- Wider-1: pre=125ms, post=250ms, n_timesteps=71
- Wider-2: pre=150ms, post=300ms, n_timesteps=85

## Code changes
The Stage3 window parameter chain is now configurable end-to-end:
- `phase3_password_inception/run_password_closure_inception.py`
- `phase3_password_inception/rebuild_merged_wider.py`
- `adapt_password_mixed_inception.py`
- `onset_detection/stage2_segmental/scripts/eval_stage123_end_to_end_strongstage2.py`
- `clean_password_eval/eval_still_password_probe_gt_stage3.py`
- `demo_inference_api/inference/pipeline_inference.py`

Checkpoint metadata now stores:
- `pre_ms`
- `post_ms`
- `n_timesteps`
- `target_rate_hz`

## Key results
See `/Users/shiyi/备份（mac_vs专用）/results/stage3_window_ablation_20260327/SUMMARY.md` for the compact table.

### still probe GT fixed CER
- baseline 100/200: 0.334444
- wider-1 125/250: 0.366111
- wider-2 150/300: 0.341667

### fair6 gt_keyframes fixed CER
- baseline 100/200: 0.269663
- wider-1 125/250: 0.269663
- wider-2 150/300: 0.292135

### fair6 stage1_bestpred overlap CER
- baseline 100/200: 0.269663
- wider-1 125/250: 0.382022
- wider-2 150/300: 0.404494

### mixed holdout adapted CER
- baseline 100/200: 0.258427
- wider-1 125/250: 0.269663
- wider-2 150/300: 0.292135

## Conclusion
Wider fixed windows are not the next Stage3 win.
- Wider-1 preserved fair6 GT fixed performance, but degraded automatic fair6 decoding and worsened still-probe GT fixed.
- Wider-2 degraded both still and fair6.
- Current evidence does not support wider fixed windows as the right fix for the remaining Stage3 resolution gap.

## Current interpretation
- Stage1 coarse merge already removed the largest catastrophic segment-selection failures.
- The remaining Stage3 errors are still dominated by local neighbor-key confusions.
- The next Stage3 step should not continue the wider-window direction.
