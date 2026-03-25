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
