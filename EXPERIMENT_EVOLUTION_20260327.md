# Experiment Evolution Log (2026-03-27)

This document is the consolidated top-level experiment log for the project.
It exists because the project history had already been recorded in many separate
README / handoff / status files, but not yet in one chronological place.

The goal of this file is to help with:
- paper writing
- method section / experiment section reconstruction
- remembering why a route was adopted or abandoned
- quickly onboarding a fresh Codex / Claude / collaborator

---

## 1. Project goal

We are studying an Apple internal IMU keystroke side-channel attack.

High-level objective:
- prove that internal IMU motion can leak password information
- first in a controlled single-user / single-device setting
- then gradually relax the threat model

Paper target:
- top security venue / top journal quality
- meaning we need both:
  - strong end-to-end quantitative results
  - a clean, honest experimental narrative

---

## 2. Major phases so far

### Phase A: Data collection foundation
Key work:
- establish collector pipeline
- verify non-root IMU capture path
- stabilize sampling rate / quality gates
- define single-key / free-type / password collection profiles

Main files:
- `/Users/shiyi/备份（mac_vs专用）/collector.py`
- `/Users/shiyi/备份（mac_vs专用）/sensor_reader.py`
- `/Users/shiyi/备份（mac_vs专用）/spu_backend.py`
- `/Users/shiyi/备份（mac_vs专用）/COLLECTION_PROFILES_AND_MODELS.md`
- `/Users/shiyi/备份（mac_vs专用）/PERMISSION_MODEL.md`
- `/Users/shiyi/备份（mac_vs专用）/NONROOT_TRIAL.md`
- `/Users/shiyi/备份（mac_vs专用）/ROOT_VS_NONROOT_COMPARISON.md`

Problems encountered:
- unstable sample-rate sessions
- permission / non-root access uncertainty
- session quality inconsistency

How we addressed them:
- precheck frequency gate
- watchdog-based aborts
- session-rate scanning
- explicit audit files such as attempts / protocol / meta

---

### Phase B: Stage3 password classifier became real
Key work:
- move from generic single-key intuition to real password-window classifier
- establish InceptionTime-based Stage3 for password sequences
- build len8/9/10 standalone password route
- add mixed-scene adaptation and hard-negative oversampling

Main files:
- `/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/run_password_closure_inception.py`
- `/Users/shiyi/备份（mac_vs专用）/adapt_password_len8_inception.py`
- `/Users/shiyi/备份（mac_vs专用）/adapt_password_multilen_inception.py`
- `/Users/shiyi/备份（mac_vs专用）/adapt_password_mixed_inception.py`
- `/Users/shiyi/备份（mac_vs专用）/multisplit_password_len8_inception.py`
- `/Users/shiyi/备份（mac_vs专用）/password_only_len8_inception.py`
- `/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/README.md`
- `/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/STATUS.md`

Problems encountered:
- zero-shot `single_key -> password` transfer was weak
- domain shift from isolated keystrokes to password typing
- neighboring-key confusion persisted

How we addressed them:
- direct password data collection
- adaptation from standalone password data
- multi-length training
- mixed-scene adaptation
- hard-negative oversampling on confusion-heavy characters

Important conclusion:
- Stage3 itself is real and useful; it is not the main blocker anymore.

---

### Phase C: Stage1 / onset / segment route became the main bottleneck
Key work:
- build Stage1 dense labeling / onset route
- build Stage2 peak-keyness / overlap / segmental decoding chain
- move toward full-stream password localization

Main files:
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/README.md`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_ctc/README.md`
- `/Users/shiyi/备份（mac_vs专用）/ONSET_DETECTION_PLAN.md`
- `/Users/shiyi/备份（mac_vs专用）/ONSET_CODEX_HANDOFF.md`
- `/Users/shiyi/备份（mac_vs专用）/PROJECT_HANDOFF_20260322.md`
- `/Users/shiyi/备份（mac_vs专用）/STAGE1_IDEA_LOG_20260324.md`

Problems encountered:
- full-stream localization was harder than segment classification
- generic duration-aware / pointwise detectors were not robust enough
- the real challenge became: which full-stream segment is the password burst?

How we addressed them:
- moved toward peak keyness + proposer + ranking style pipeline
- built segmental Stage1 / Stage2 variants and multiple evaluations
- established fair6-style evaluation for automatic route comparison

Important conclusion:
- the primary difficulty shifted from “classify window” to “find the right password segment in the stream”.

---

### Phase D: Demo / still-password-still / protocol stress-testing
Key work:
- build clean still-password-still collectors and evaluation routes
- test both local and cross-device still-password-still behavior
- package demo inference API and portable collection tools
- isolate what breaks under still-password-still protocol

Main files:
- `/Users/shiyi/备份（mac_vs专用）/demo_inference_api/README.md`
- `/Users/shiyi/备份（mac_vs专用）/demo_inference_api/docs/VERIFIED_STATUS_20260326.md`
- `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/README.md`
- `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/STEP0_STEP1_PROTOCOL_PROBE.md`
- `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/eval_still_password_probe.py`
- `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/eval_still_password_probe_gt_stage3.py`
- `/Users/shiyi/备份（mac_vs专用）/STILL_PASSWORD_STATUS_20260327.md`

Problems encountered:
- still-password-still initially looked much worse than fair6 / continuous typing
- local and cross-device still tests both suffered from segment selection errors
- it was easy to misdiagnose the issue as “the model learned only the old mode”

How we addressed them:
- probe-style controlled recordings
- auto vs event-window vs GT-assisted comparisons
- local vs cross-device comparisons
- fair6 regression checks after still-scene fixes

Important conclusion:
- the initial still-password-still failure was not mainly “model forgot the essence”
- a big part was catastrophic Stage1 fragment selection / truncation

---

## 3. Important recent findings

### 3.1 Coarse merge repaired the worst Stage1 failure mode
We added an engineering posthoc coarse merge for nearby Stage1 fragments.

Implemented in:
- `/Users/shiyi/备份（mac_vs专用）/demo_inference_api/inference/pipeline_inference.py`

Observed effects:
- local still-password-still probe:
  - auto CTC `0.472 -> 0.347`
- fair6 local reproduction:
  - CER `0.3034 -> 0.2697`

Interpretation:
- this is a useful engineering enhancement
- it is **not** being treated as a paper-worthy general method
- it is acceptable to keep using it in the engineering mainline

### 3.2 After Stage1 catastrophic failures are reduced, the residual gap looks like Stage3 resolution
GT-assisted Stage3 on local still probe:
- fixed mean CER `0.3344`
- overlap mean CER `0.3989`

Per-character probe details show:
- Top-5 hit is effectively saturated on the small probe set
- remaining errors are mostly neighboring-key confusions such as:
  - `x -> c`
  - `p -> o`
  - `w -> e`
  - `d -> f`
  - `t -> y`

Interpretation:
- after Stage1 repair, still-password-still is much closer to mainline behavior
- the next likely gain is in Stage3 resolution / alignment / windowing, not in reopening the full Stage1 problem from scratch

---

## 4. Data / protocol realities we learned the hard way

### What we first thought
- maybe still-password-still fails because the model only learned one protocol
- maybe cross-device shift is the main cause
- maybe we would need still-specific adaptation data just to make it work

### What the evidence forced us to accept
- local still-password-still also failed before Stage1 repair, so it was not purely a cross-device problem
- cross-device failures were real, but they were mixed with Stage1 wrong-fragment failures
- once Stage1 catastrophic errors were suppressed, still-scene results became much closer to the fair6 mainline

### Current honest interpretation
- the original route does have genuine transfer ability
- but it was previously masked by Stage1 fragment-selection failures
- the remaining gap is smaller and more localized than we first feared

---

## 5. Current paper-safe narrative

What is safe to say:
- the core attack route works
- Stage3 is real
- Stage1/Stage2 can be strengthened with engineering posthoc that improves target-scene performance
- still-password-still is not a fundamentally different world once catastrophic segmentation errors are reduced

What we should avoid overselling:
- coarse merge as a universal or theoretically elegant method
- any hidden adaptation / training use that is not disclosed
- any claim that false positives are completely solved

Current preferred positioning:
- use improved results in the paper
- do **not** make coarse merge itself the star method
- treat it as implementation / engineering repair

---

## 6. Where key records live

### Global / onboarding
- `/Users/shiyi/备份（mac_vs专用）/README.md`
- `/Users/shiyi/备份（mac_vs专用）/CODE_MAP.md`
- `/Users/shiyi/备份（mac_vs专用）/PROJECT_HANDOFF_20260322.md`
- `/Users/shiyi/备份（mac_vs专用）/NEXT_CHAT_BRIEF_20260322.md`

### Dataset bookkeeping
- `/Users/shiyi/备份（mac_vs专用）/DATASET_REGISTRY_20260327.md`
- `/Users/shiyi/备份（mac_vs专用）/dataset_registry_20260327.json`
- `/Users/shiyi/备份（mac_vs专用）/COLLECTION_PROFILES_AND_MODELS.md`
- `/Users/shiyi/备份（mac_vs专用）/MIXED_SINGLE_RETRY_COLLECTION_20260324.md`

### Onset / Stage1 / Stage2
- `/Users/shiyi/备份（mac_vs专用）/ONSET_DETECTION_PLAN.md`
- `/Users/shiyi/备份（mac_vs专用）/ONSET_CODEX_HANDOFF.md`
- `/Users/shiyi/备份（mac_vs专用）/STAGE1_IDEA_LOG_20260324.md`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/README.md`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/README.md`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_ctc/README.md`

### Stage3 / password route
- `/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/README.md`
- `/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/STATUS.md`
- `/Users/shiyi/备份（mac_vs专用）/adapt_password_mixed_inception.py`
- `/Users/shiyi/备份（mac_vs专用）/multisplit_password_len8_inception.py`

### Still / probe / demo track
- `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/README.md`
- `/Users/shiyi/备份（mac_vs专用）/clean_password_eval/STEP0_STEP1_PROTOCOL_PROBE.md`
- `/Users/shiyi/备份（mac_vs专用）/STILL_PASSWORD_STATUS_20260327.md`
- `/Users/shiyi/备份（mac_vs专用）/demo_inference_api/README.md`
- `/Users/shiyi/备份（mac_vs专用）/demo_inference_api/docs/VERIFIED_STATUS_20260326.md`

---

## 7. Audit result (2026-03-27)

### What was already well recorded
- collection protocol and raw data conventions
- onset / Stage1 design history
- current mainline route and handoff notes
- still-password-still diagnosis and coarse-merge result
- dataset roots and major usage distinctions

### What was missing before this file
- one single chronological experiment-evolution record
- one place that explains how the story moved from:
  - collector / data quality
  - to Stage3 classifier
  - to Stage1 bottleneck
  - to still-password-still debugging
  - to current focus on Stage3 resolution

This document closes that gap.

---

## 8. Current next step

Current project decision after freezing the recent Stage1/posthoc work:
- keep coarse merge in the engineering mainline
- keep the heuristic itself low-profile in the paper
- focus the next iteration on Stage3 resolution improvement
- especially windowing / alignment / neighbor-key discrimination
