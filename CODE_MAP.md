# Code Map

This file is the current map of the main workspace under
[备份（mac_vs专用）](/Users/shiyi/备份（mac_vs专用）).

## 1. Current Reliable Mainline Pieces

### Data collection
- [collector.py](/Users/shiyi/备份（mac_vs专用）/collector.py)
- [sensor_reader.py](/Users/shiyi/备份（mac_vs专用）/sensor_reader.py)
- [spu_backend.py](/Users/shiyi/备份（mac_vs专用）/spu_backend.py)
- [keyboard_listener.py](/Users/shiyi/备份（mac_vs专用）/keyboard_listener.py)
- [typing_prompt_profiles.py](/Users/shiyi/备份（mac_vs专用）/typing_prompt_profiles.py)

### Password classifier route
- [phase3_password_inception/run_password_closure_inception.py](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/run_password_closure_inception.py)
- [adapt_password_len8_inception.py](/Users/shiyi/备份（mac_vs专用）/adapt_password_len8_inception.py)
- [multisplit_password_len8_inception.py](/Users/shiyi/备份（mac_vs专用）/multisplit_password_len8_inception.py)
- [password_only_len8_inception.py](/Users/shiyi/备份（mac_vs专用）/password_only_len8_inception.py)

Status:
- classifier route is working
- `single_key + password adaptation` is still the strongest classifier story
- this has only been clearly validated on the `InceptionTime` route so far
- `password-only` did not beat `baseline + password adaptation` in the current verified experiments
- multi-model comparison is still incomplete
- the password classifier is useful, but should not be treated as already “full-strength” or fully exhausted

### Onset Stage 1 / historical baseline
- [onset_detection/password_segment_preprocessor.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_preprocessor.py)
- [onset_detection/password_segment_detector.py](/Users/shiyi/备份（mac_vs专用）/onset_detection/password_segment_detector.py)

Status:
- Stage 1 coarse localization works on mixed2
- this file is still useful as a historical baseline and utility path
- it is no longer the only recommended Stage 2 direction

## 2. Current Onset Branches

### Claude branch
- [onset_detection/stage2_claude](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude)

Contents:
- `stage2_dp_segmentation.py`
- `stage2_ctc.py`
- `password_segment_detector.py`

Role:
- preserve Claude-style Stage 2 ideas
- keep runnable baselines / side branches
- current best use: comparison branch, not mainline

### GPT branch
- [onset_detection/stage2_gpt54](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54)

Contents:
- `password_stage2_model.py`
- `password_stage2_preprocessor.py`
- `password_stage2_dataset.py`
- `stage2_dense_structured.py`
- `stage2_decoder.py`
- `password_segment_detector.py`

Role:
- current main exploration branch for dense Stage 2
- already supports training + Path B inference
- current lesson: model/decoder improved, but train-test mismatch remains the main concern

## 3. Current Main Data Directories

- [data/raw/single_key](/Users/shiyi/备份（mac_vs专用）/data/raw/single_key)
- [data/raw/boost](/Users/shiyi/备份（mac_vs专用）/data/raw/boost)
- [data/raw/password/len_8](/Users/shiyi/备份（mac_vs专用）/data/raw/password/len_8)
- [data/raw/onset_negative](/Users/shiyi/备份（mac_vs专用）/data/raw/onset_negative)
- [data/raw/onset_mixed2](/Users/shiyi/备份（mac_vs专用）/data/raw/onset_mixed2)

Current interpretation:
- `single_key + boost` mainly support classifier baseline
- `password/len_8` supports classifier adaptation and Stage 2 pretraining ideas
- `mixed2` is the held-out continuous-stream evaluation target
- current suspicion: Stage 2 needs mixed-style training data, not just clean password segments

## 4. Key Docs To Read First

- [README.md](/Users/shiyi/备份（mac_vs专用）/README.md)
- [onset_detection/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md)
- [onset_detection/README_password_segment.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/README_password_segment.md)
- [onset_detection/stage2_claude/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_claude/README.md)
- [onset_detection/stage2_gpt54/README.md](/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_gpt54/README.md)
- [ONSET_CODEX_HANDOFF.md](/Users/shiyi/备份（mac_vs专用）/ONSET_CODEX_HANDOFF.md)

## 5. Current Recommendation

Do not assume Stage 2 is “almost solved”.
Current evidence says:
- Stage 1 is solved enough
- Stage 3 is solved enough
- Stage 2 needs either:
  - mixed-style training data / pseudo mixed training
  - or a cleaner rebuild into `Stage 2A group segmentation + Stage 2B onset detection`
