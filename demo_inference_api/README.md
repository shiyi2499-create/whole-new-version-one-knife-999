# Demo Inference API

This folder contains a thin inference-facing wrapper around the current Apple IMU attack pipeline.

It is intentionally scoped for demo integration:
- `sensor_capture.py`: non-root IMU capture wrapper
- `preprocess.py`: CSV parsing, sample-rate estimation, resampling
- `pipeline_inference.py`: Stage1, pipeline Stage2+3, and CTC inference entrypoints
- `checkpoints/CHECKPOINT_MANIFEST.json`: canonical checkpoint mapping and loading notes

## API surface

- `capture_imu(duration_sec, output_csv=None) -> str`
- `csv_to_array(csv_string) -> np.ndarray`
- `extract_timestamps(csv_string) -> np.ndarray`
- `estimate_sample_rate(imu_array, timestamp_col) -> float`
- `resample_to_190hz(imu_array, original_hz) -> np.ndarray`
- `load_all_models(checkpoint_dir) -> dict`
- `run_stage1(imu_array, models) -> list[dict]`
- `run_pipeline_stage23(imu_segment, models, beam_width=500) -> dict`
- `run_ctc(imu_segment, models) -> dict`

## Design notes

- Inputs are plain numpy arrays.
- Outputs are plain Python dict/list structures.
- The code does not depend on argparse.
- The checkpoint manifest allows two modes:
  1. put canonical filenames into a standalone `checkpoint_dir`
  2. rely on the recorded `source_path` entries inside the original repo tree

## Current best route baked into defaults

- Stage1: dense labeling + fixed posthoc parameters from current best dev3 bundle
- Pipeline stage2+3: keyness RF + mixed-adapt hard-neg Stage3 + new overlap + beam500
- CTC: best current CTC-dominant checkpoint + greedy / beam20 decode


## Verification helpers

- `smoke_test_api.py`: runs a real import + model-load + one-sample inference smoke test
- `docs/VERIFIED_STATUS_20260326.md`: records what was actually verified on server and local Mac M4
