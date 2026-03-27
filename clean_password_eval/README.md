# Clean Password Eval Tools

This folder is for the eval-only dataset you plan to record with the simplest pattern:
- still for 3 seconds
- type one password
- press Enter to finish
- still for 3 seconds

The goal is to measure how well the existing routes work in the easiest realistic setting.

## Files

- `collect_clean_password_eval.py`
  - dedicated collector for this pattern
  - writes `*_sensor.csv`, `*_events.csv`, `*_attempts.csv`, `*_protocol.json`, `*_meta.txt`
  - marks every session as eval-only and not for training
- `eval_clean_password_routes.py`
  - runs both current routes on the recorded dataset
  - route 1: Stage1 -> pipeline Stage2+3
  - route 2: Stage1 -> CTC

## Default dataset location

- `data/raw/clean_password_eval`

## Important rule

This dataset is for evaluation only.
Do not mix it into any existing training split unless you explicitly decide to later.

## Recommended workflow for tomorrow

1. Record a few trials with `collect_clean_password_eval.py`
2. Confirm the protocol json shows `eval_only: true`
3. Run `eval_clean_password_routes.py`
4. Compare:
   - pipeline exact match / CER / predicted length
   - CTC exact match / CER / predicted length
   - Stage1 IoU vs event-derived typing interval


## Current status (2026-03-27)

- `still -> password -> still` is now a first-class evaluation scenario.
- We have a probe evaluator for three views:
  - `auto_fullsample`
  - `event_window`
  - `tight_burst`
- We also have GT-assisted Stage3 evaluation for the same probe set.
- A `coarse merge` Stage1 posthoc has already been validated as a useful engineering fix for fragment-selection failures.
- The current project interpretation is:
  - use coarse merge in the engineering mainline
  - do not oversell the heuristic itself as a paper method
  - focus the next iteration on Stage3 resolution / neighbor-key discrimination

### Key scripts

- `collect_clean_password_eval.py`
- `collect_still_password_imu_only.py`
- `collect_still_password_probe_batch.py`
- `eval_clean_password_routes.py`
- `eval_still_password_probe.py`
- `eval_still_password_probe_gt_stage3.py`

### Current status note

- [`STILL_PASSWORD_STATUS_20260327.md`](/Users/shiyi/备份（mac_vs专用）/STILL_PASSWORD_STATUS_20260327.md)
