# Code Map

This file is the current map of the main workspace under
[备份（mac_vs专用）](/Users/shiyi/备份（mac_vs专用）).

Its goal is simple:

- explain which files are part of the current mainline
- explain which files are legacy or auxiliary
- reduce confusion when running experiments locally or on the server

## 1. Current Mainline

These files define the active password-route story.

### Data collection

- [collector.py](/Users/shiyi/备份（mac_vs专用）/collector.py)
  - main data collection entrypoint
  - supports:
    - `single_key`
    - `free_type` with `sentence / continuous / password`

- [sensor_reader.py](/Users/shiyi/备份（mac_vs专用）/sensor_reader.py)
  - IMU reader abstraction
  - currently aligned to the non-root direct SPU route

- [spu_backend.py](/Users/shiyi/备份（mac_vs专用）/spu_backend.py)
  - direct Apple SPU / IOKit backend used by the current non-root path

- [keyboard_listener.py](/Users/shiyi/备份（mac_vs专用）/keyboard_listener.py)
  - keyboard label capture for `events.csv`

- [typing_prompt_profiles.py](/Users/shiyi/备份（mac_vs专用）/typing_prompt_profiles.py)
  - defines prompt profiles
  - current active password pool:
    - `a-z0-9`
    - `len=8`
    - `200` total strings
    - `20` groups

### Collection helpers

- [run_password_len8_part.sh](/Users/shiyi/备份（mac_vs专用）/run_password_len8_part.sh)
  - helper for `password/len_8`
  - current use: record `part 1..20`

- [run_text_part.sh](/Users/shiyi/备份（mac_vs专用）/run_text_part.sh)
  - generic helper for `sentence / continuous / password`

- [run_single_key_boost_finger.sh](/Users/shiyi/备份（mac_vs专用）/run_single_key_boost_finger.sh)
  - targeted single-key boost capture helper

### Preprocessing / training

- [preprocessor.py](/Users/shiyi/备份（mac_vs专用）/preprocessor.py)
  - event-aligned window extraction + resampling
  - current default target rate remains `190 Hz`

- [phase3_password_inception/run_password_closure_inception.py](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/run_password_closure_inception.py)
  - current zero-shot password-route script
  - trains baseline on `single_key + boost`
  - tests on password strings

- [adapt_password_len8_inception.py](/Users/shiyi/备份（mac_vs专用）/adapt_password_len8_inception.py)
  - current main adaptation script
  - baseline from `single_key + boost`
  - default split now:
    - adapt parts `1-16`
    - test parts `17-20`
  - automatically ignores older duplicate/incomplete sessions for the same part

- [multisplit_password_len8_inception.py](/Users/shiyi/备份（mac_vs专用）/multisplit_password_len8_inception.py)
  - repeated group-split evaluation for:
    - zero-shot
    - `single_key + password adaptation`
  - intended to estimate split stability over multiple random `16/4` group splits

- [password_only_len8_inception.py](/Users/shiyi/备份（mac_vs专用）/password_only_len8_inception.py)
  - password-only baseline
  - no `single_key + boost`
  - intended as the domain-native comparison

### Diagnostics / reporting

- [phase3_password_inception/diagnose_singlekey_inception.py](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/diagnose_singlekey_inception.py)
  - checks whether the current password-route Inception trainer is still strong
    on held-out isolated-key data

- [scan_sampling_rates.py](/Users/shiyi/备份（mac_vs专用）/scan_sampling_rates.py)
  - sampling-rate audit for raw sessions

## 2. Current Main Data Directories

- [data/raw/single_key](/Users/shiyi/备份（mac_vs专用）/data/raw/single_key)
  - main isolated-key baseline data

- [data/raw/boost](/Users/shiyi/备份（mac_vs专用）/data/raw/boost)
  - targeted single-key boost data

- [data/raw/password/len_8](/Users/shiyi/备份（mac_vs专用）/data/raw/password/len_8)
  - current password-route dataset
  - active main testbed

- [data/processed/merged_dataset.npz](/Users/shiyi/备份（mac_vs专用）/data/processed/merged_dataset.npz)
  - main baseline training set
  - expected source:
    - `single_key + boost`

## 3. Current Main Docs

- [README.md](/Users/shiyi/备份（mac_vs专用）/README.md)
  - top-level working plan

- [PAPER_OUTLINE.md](/Users/shiyi/备份（mac_vs专用）/PAPER_OUTLINE.md)
  - current paper structure

- [COLLECTION_PROFILES_AND_MODELS.md](/Users/shiyi/备份（mac_vs专用）/COLLECTION_PROFILES_AND_MODELS.md)
  - collection profile overview and model-route logic

- [phase3_password_inception/README.md](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/README.md)
  - password-route explanation

- [phase3_password_inception/STATUS.md](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/STATUS.md)
  - current experimental status

- [phase3_password_inception/ROADMAP.md](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/ROADMAP.md)
  - next-step plan

- [phase3_password_inception/RESULTS_LEN8.md](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/RESULTS_LEN8.md)
  - current `len=8` result summary

## 4. Legacy / Secondary Files

These files still matter, but are not the current password-mainline entrypoint.

- [run_real_freetype.py](/Users/shiyi/备份（mac_vs专用）/run_real_freetype.py)
  - older sentence/free-type end-to-end path

- [run_freetype_closure_eval.py](/Users/shiyi/备份（mac_vs专用）/run_freetype_closure_eval.py)
  - sentence/free-type closure evaluation

- [run_freetype_finetune_beam.py](/Users/shiyi/备份（mac_vs专用）/run_freetype_finetune_beam.py)
  - sentence/free-type fine-tune + beam search path

- [phase3_decoder.py](/Users/shiyi/备份（mac_vs专用）/phase3_decoder.py)
  - older sentence/word decoder logic
  - not the current main password route

- [train_phase2.py](/Users/shiyi/备份（mac_vs专用）/train_phase2.py)
  - broader Phase 2 model benchmark / training script

- [run_transformer_only.py](/Users/shiyi/备份（mac_vs专用）/run_transformer_only.py)
  - transformer-only training/eval route

- [train_baseline.py](/Users/shiyi/备份（mac_vs专用）/train_baseline.py)
  - classical feature-based baselines

## 5. Working Rules We Are Following

1. keep `InceptionTime` fixed while validating password-route protocol choices
2. do not mix model changes with data-protocol changes too early
3. prefer root-level scripts for server runs
4. treat `single_key + boost` as the baseline training source unless explicitly
   testing `password only`
5. treat `password/len_8` as the current main continuous-input benchmark

## 6. Near-Term Planned Experiments

1. repeated random `16/4` group splits for:
   - zero-shot
   - adaptation
2. repeated random `16/4` group splits for:
   - password-only
3. compare:
   - `single_key + password adaptation`
   - `password only`
4. only after protocol stability is established, revisit model comparison
