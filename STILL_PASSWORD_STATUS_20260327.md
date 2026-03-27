# Still-Password-Aftermath Status (2026-03-27)

This note freezes the current understanding of the `still -> password -> still` line.

## What is already established

### 1. Stage1 catastrophic wrong-fragment failures are largely a posthoc issue
A simple coarse-merge posthoc for nearby Stage1 fragments already fixes the worst fragment-selection failures.

Implemented in:
- `/Users/shiyi/备份（mac_vs专用）/demo_inference_api/inference/pipeline_inference.py`
- exposed in probe eval through:
  - `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/eval_still_password_probe.py`

### 2. Coarse merge helps both still-password-still and fair6 mixed evaluation
Still-password-still probe (5 local fair6 strings):
- baseline auto CTC: `0.472`
- coarse-merge-4s auto CTC: `0.347`

fair6 local reproduction:
- baseline CER: `0.3034`
- coarse-merge-4s CER: `0.2697`

Interpretation:
- coarse merge is an effective engineering repair for Stage1 fragment truncation / fragmentation.
- we are **not** treating it as a universal paper method.
- it is safe to keep using it in the main engineering path.

### 3. After Stage1 catastrophic errors are suppressed, the remaining gap is mostly Stage3 resolution
GT-assisted Stage3 on the 5 local still probe strings:
- fixed mean CER: `0.3344`
- overlap mean CER: `0.3989`

The remaining errors are mainly neighbor-key confusions such as:
- `x -> c`
- `p -> o`
- `w -> e`
- `d -> f`
- `t -> y`

## What we are NOT claiming
- We are not claiming Stage1 now produces zero false positives.
- We are not claiming coarse merge is a theoretically clean or universally robust method.
- We are not claiming still-password-still is already identical to the mainline fair6 setting.

## Current project decision
- Keep coarse merge in the engineering mainline.
- Do not present the heuristic itself as a core paper method.
- Focus the next iteration on Stage3 resolution rather than reopening Stage1 from scratch.

## Relevant files
- code:
  - `/Users/shiyi/备份（mac_vs专用）/demo_inference_api/inference/pipeline_inference.py`
  - `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/eval_still_password_probe.py`
  - `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/eval_still_password_probe_gt_stage3.py`
- documentation:
  - `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/README.md`
  - `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/STEP0_STEP1_PROTOCOL_PROBE.md`
  - `/Users/shiyi/备份（mac_vs专用）/STILL_PASSWORD_STATUS_20260327.md`
