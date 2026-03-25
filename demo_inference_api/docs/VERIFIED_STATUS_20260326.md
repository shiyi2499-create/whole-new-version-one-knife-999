# Demo Inference API Verified Status (2026-03-26)

## What was actually verified

### Local Mac M4
- `inference.preprocess` import: passed
- `inference.sensor_capture` import: passed
- `inference.pipeline_inference` import: passed after fixing package path issues
- end-to-end smoke test on an existing sensor CSV: passed
  - sample: `data/raw/password/len9/p01_free_type_password_part1_20260321_195652_sensor.csv`
  - stage1 -> pipeline and stage1 -> ctc both produced outputs

### Server
- light imports: passed
- heavy import (`pipeline_inference`): passed
- end-to-end smoke test on the same sensor CSV: passed
  - stage1 -> pipeline and stage1 -> ctc both produced outputs

## Fixes made during verification

1. `demo_inference_api/inference/__init__.py`
- changed to lazy-load the heavy pipeline module
- this lets `preprocess` and `sensor_capture` import cleanly without pulling torch/scipy-heavy modules immediately

2. `demo_inference_api/inference/pipeline_inference.py`
- added `onset_detection/stage2_ctc` to `sys.path`
- fixes `from utils.vocab ...` imports inside the legacy CTC utilities

3. `demo_inference_api/inference/checkpoints/CHECKPOINT_MANIFEST.json`
- corrected Stage1 `base_filters` from `24` to `12`
- this now matches the actual Stage1 checkpoint tensor shapes

4. Local main workdir checkpoint completeness
- copied the missing local CTC checkpoint folder back into `runs/stage2_ctc_e2e_fairholdout_ctcdom_20260325/`
- this was required for the local Mac M4 smoke test to load CTC successfully

## Current honest status

The API is no longer just syntax-checked.
It has now passed:
- real import checks
- real model-loading checks
- one end-to-end smoke test on both server and local Mac M4

It is still not the same thing as full demo validation on a fresh live capture, but it is now in much better shape for Claude to build the demo on top of it.
