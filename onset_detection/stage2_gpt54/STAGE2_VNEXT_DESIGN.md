# Stage2-vNext for `onset_detection`

## Goal
Rebuild Stage 2 as **dense sequence modeling + structured decode** inside the already-detected coarse password region.

This bundle does **not** redefine:
- Stage 1 coarse localization
- Stage 3 password classifier
- the overall project outside `onset_detection`

## Core change
Replace:
- onset detector -> NMS -> rhythm grouping -> heuristic scoring
- energy valley split -> per-segment onset detection -> top-N keep

with:
- patch-level dense sequence encoder
- multi-head outputs: `key`, `boundary`, optional `inside`
- global decode with protocol prior: `5 passwords x len 8`

## Why this is the right pivot
The current failure mode is not just bad thresholding. It is that Stage 2 is framed as **local event proposal + heuristic stitching**, while the actual target is **structured segmentation of a continuous IMU stream**.

This design borrows the useful parts of current literature:
- patch-level labeling for continuous sensor streams
- explicit boundary modeling
- multi-stage temporal refinement to reduce over-segmentation
- global decode with hard structural constraints

## Files in this bundle
- `password_stage2_model.py`
  - lightweight MS-TCN++-style temporal model
  - heads for `key`, `boundary`, `inside`
- `password_stage2_dataset.py`
  - variable-length sequence dataset and collate function
- `password_stage2_preprocessor.py`
  - patchification, feature construction, dense target construction
- `stage2_decoder.py`
  - protocol prior, boundary DP, per-segment 8-slot key decode

## Recommended integration plan

### 1) Keep existing files as-is for baseline compatibility
Keep current:
- `password_segment_preprocessor.py`
- `onset_model.py`
- `train_onset.py`
- `password_segment_detector.py`

but add one new path:
- `stage2_method="dense_structured"`

### 2) Preprocessing
Create a new dataset builder that:
- takes Stage 1 coarse password regions or GT refined password regions for training
- patchifies the sequence with a fixed patch width + stride
- constructs dense targets:
  - `key_target[t]`
  - `boundary_target[t]`
  - `inside_target[t]`

Recommended default config:
- target rate: same as current repo default
- patch width: 160 ms
- patch stride: 20 ms
- key radius: 60 ms
- boundary radius: 120 ms

### 3) Training
Train a new Stage 2 model with:
- BCE or focal BCE per head
- temporal smoothing regularizer on adjacent logits/probabilities
- optional class balancing on positive heads

### 4) Decoding
Given dense outputs on a coarse region:
- decode exactly 4 inner boundaries to get 5 segments
- decode exactly 8 key slots inside each segment
- optionally rerank top-K segmentation hypotheses with the existing password classifier

## Decoder scoring intuition
Boundary DP should prefer:
- high boundary score at the cut
- enough inside mass inside each segment
- segment durations within a protocol-consistent range

Key-slot decode should prefer:
- high key score on chosen patches
- monotonic increasing positions
- plausible inter-key spacing
- not collapsing all 8 slots into one local neighborhood

## What should be changed later in main repo

### `onset_model.py`
Add a new model alias, or import `PasswordStage2TCN` from `password_stage2_model.py`.

### `train_onset.py`
Add a new task:
- `password_stage2`

Support:
- sequence dataset
- multi-head losses
- smoothing loss

### `password_segment_detector.py`
Add:
- `stage2_method="dense_structured"`
- loading of stage2 dense checkpoint
- call to `decode_stage2_dense(...)`
- optional classifier-aware reranking hook

## What this bundle intentionally does not claim
This is a **serious scaffold**, not a promise that end-to-end metrics will improve without data iteration. The main value is:
- the problem formulation is corrected
- the interfaces are explicit
- the decode logic is no longer heuristic-only

## Recommended first experiment
Do **not** connect classifier rerank on day one.
First validate:
- boundary count stability
- per-segment 8-slot recovery rate
- Stage 2 output quality before classifier feedback

If those move in the right direction, then add classifier-aware reranking.
