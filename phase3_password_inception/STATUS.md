# Current Status

## Scope

This folder is a Phase 3 experiment track for:

- `InceptionTime` baseline
- password-style continuous-string inference
- password-like threat model
- top-k / top-N attack-facing evaluation

It exists to avoid changing the main workspace under
[备份（mac_vs专用）](/Users/shiyi/备份（mac_vs专用）).

## What is already confirmed

1. The strongest visible Phase 2 server baseline is `InceptionTime`
   - [results_phase2.json](/Users/shiyi/备份（mac_vs专用）/results/服务器results/results_phase2.json)
   - `dl_InceptionTime = 0.8592`
   - `dl_Transformer = 0.8095`

2. The old sentence-style free_type route is not the right first target for the
   current attack story
   - it keeps `space/enter` in the label space
   - it uses a weaker backbone than the current best visible baseline
   - it evaluates sentence reconstruction rather than continuous-string recovery

3. The new no-space route runs end-to-end
   - [run_password_closure_inception.py](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/run_password_closure_inception.py)
   - local smoke test completed successfully

4. The current main story is now more focused than before
   - `single_key + boost` remains the main baseline training source
   - `password` prompt profile is the main continuous-input target
   - sentence-style free_type is kept, but not the current headline route
   - `continuous` bridge prompts are optional rather than required

5. The password-route trainer is now closer to the original strong Phase 2
   Inception recipe
   - same InceptionTime family
   - AdamW + cosine schedule
   - Phase 2-style augmentation family
   - stronger defaults (`epochs=280`, `patience=60`, `batch_size=32`)

## What has not been claimed yet

1. We are not claiming the local smoke-test accuracy is meaningful
2. We are not claiming sentence-level natural-language recovery is solved
3. We are not claiming fast-overlap typing is solved
4. We are not claiming blind onset detection is solved yet
5. We are still not claiming this is a perfect byte-for-byte clone of the full
   Phase 2 benchmark pipeline (e.g. it is not the 5-fold benchmark script)

## Confirmed password v1 dataset status

The current password dataset is now fully collected:

- path: `data/raw/password/len_8`
- charset: `a-z0-9`
- length: `8`
- current total pool size: `200`
- grouping: `20 x 10`

Collection quality summary:

- the first `10` parts completed
- effective sampling rate stays near `200 Hz`
- successful prompts align with the generated password list
- retries are logged in `prompts.csv`, but the successful (`YES`) rows are
  correctly aligned
- the collected dataset is considered valid for downstream training/evaluation

## Current experimental conclusion

The latest server-side experiments establish the following:

1. The current InceptionTime password-route trainer is strong on held-out
   isolated-key data
   - single-key diagnostic:
     - `val_best_accuracy = 89.0%`
     - `test_top1 = 85.7%`
     - `test_top3 = 98.7%`
     - `test_top5 = 99.6%`

2. Direct zero-shot transfer from `single_key + boost` to continuous password
   input is weak
   - `char_top1 = 8.1%`
   - `char_top3 = 16.2%`
   - `char_top5 = 27.5%`
   - `sequence_top100 = 0.0%`
   - `CER = 91.9%`

3. Password-style adaptation is highly effective
   - split:
     - parts `1-8` (`80` strings) for adaptation
     - parts `9-10` (`20` strings) held out for test
   - adapted result:
     - `char_top1 = 62.5%`
     - `char_top3 = 87.5%`
     - `char_top5 = 96.2%`
     - `sequence_top100 = 35.0%`
     - `CER = 37.5%`

Interpretation:

- the password-route problem is **not** caused by a broken trainer
- it is **not** primarily caused by bad password collection
- the main issue is domain shift from isolated-key training to continuous
  password input
- limited password-style adaptation can substantially recover performance
- collection is now continuing from `part 11` onward in the same `len_8` pool

## Immediate next step

Use the `len_8` result as the first stable password-route reference point, then
expand along one of these axes:

1. replicate `len_8` to confirm stability
2. extend to `len_10` / `len_12`
3. add symbol classes after the alphanumeric route is stable
4. add onset detection for continuous stream segmentation
