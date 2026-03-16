# Onset Detection Module

Keystroke onset detection for the Apple Internal IMU side-channel attack.

This module fills the missing link in the attack chain: given a continuous
IMU data stream, automatically determine **when** keystrokes occur, estimate
when a keyboard-activity episode starts and ends, then hand the password-like
windows to the existing password classifier.

## Architecture

```
continuous IMU stream
  → onset_preprocessor.py  (build sliding-window dataset from existing data)
  → train_onset.py         (train 1D-CNN binary classifier)
  → eval_onset.py          (segment-level + event-level metrics)
  → eval_onset_e2e.py      (onset / episode boundary → classifier → password top-k)
```

For new data collection:
```
onset_collector.py --mode negative   (idle / trackpad / shake / etc.)
onset_collector.py --mode mixed      (interleaved evaluation streams)
```

## File Map

| File | Role |
|------|------|
| `onset_utils.py` | Peak detection, NMS, event matching, metrics |
| `onset_model.py` | 1D-CNN onset detector (+ energy baseline) |
| `onset_preprocessor.py` | Build sliding-window dataset from raw data |
| `onset_dataset.py` | PyTorch Dataset + session-level splitting |
| `train_onset.py` | Training script with balanced sampling |
| `onset_collector.py` | Negative sample + mixed-stream collector |
| `eval_onset.py` | Segment-level + event-level evaluation |
| `eval_onset_e2e.py` | Full attack chain: onset → classifier → top-k |

## Quick Start (MVP)

### Step 1: Build dataset from existing data (keyboard positives only)

Uses existing single_key + password sessions as a first sanity check:

```bash
python3 onset_detection/onset_preprocessor.py \
  --keyboard-dirs data/raw/single_key data/raw/boost \
  --password-dirs data/raw/password/len_8 \
  --output data/processed/onset_dataset_keyboard_only.npz
```

### Step 2: Collect nuisance-motion negative samples

**This is required for a credible result.** Without trackpad/idle negatives,
the detector only sees keystroke-vs-interkey-gap and cannot reject the
most important real-world confounders (trackpad clicks produce mechanical
impulses similar to keystrokes).

Minimum set for MVP — record at least 60s of each:

```bash
python3 onset_detection/onset_collector.py --mode negative --activity idle --duration 60
python3 onset_detection/onset_collector.py --mode negative --activity trackpad_move --duration 60
python3 onset_detection/onset_collector.py --mode negative --activity trackpad_click --duration 60
```

Strongly recommended additions (can follow after the initial three):

```bash
python3 onset_detection/onset_collector.py --mode negative --activity shake --duration 45
python3 onset_detection/onset_collector.py --mode negative --activity desk_bump --duration 45
```

### Step 3: Build full dataset (keyboard + negatives)

```bash
python3 onset_detection/onset_preprocessor.py \
  --keyboard-dirs data/raw/single_key data/raw/boost \
  --password-dirs data/raw/password/len_8 \
  --negative-dirs data/raw/onset_negative \
  --output data/processed/onset_dataset.npz
```

### Step 4: Train onset detector

```bash
python3 onset_detection/train_onset.py \
  --dataset data/processed/onset_dataset.npz \
  --checkpoint results/onset_detector.pt \
  --scaler results/onset_scaler.npz \
  --epochs 80
```

### Step 5: Evaluate (segment-level)

```bash
python3 onset_detection/eval_onset.py \
  --checkpoint results/onset_detector.pt \
  --scaler results/onset_scaler.npz \
  --dataset data/processed/onset_dataset.npz
```

### Step 6 (recommended): Collect mixed evaluation streams

```bash
python3 onset_detection/onset_collector.py --mode mixed --n-segments 15

# Event-level evaluation on mixed streams
python3 onset_detection/eval_onset.py \
  --checkpoint results/onset_detector.pt \
  --scaler results/onset_scaler.npz \
  --mixed-dirs data/raw/onset_mixed
```

### Step 7: End-to-end attack chain demo

```bash
python3 onset_detection/eval_onset_e2e.py \
  --onset-checkpoint results/onset_detector.pt \
  --onset-scaler results/onset_scaler.npz \
  --classifier-checkpoint results/inception_password_final.pt \
  --classifier-scaler results/inception_password_scaler.npz \
  --password-dirs data/raw/password/len_8 \
  --test-parts 17 18 19 20
```

## Paper-Oriented Demo Goal

The current paper-oriented target is **not** a 1-hour monitoring demo.
Instead, the preferred controlled demo is a ~2-minute mixed stream:

- several non-keyboard intervals:
  - idle
  - trackpad_move
  - trackpad_click
  - shake / desk_bump
- two keyboard episodes:
  - `typing_1`: free/random typing
  - `typing_2`: slow password-style typing that matches the current
    `a-z0-9`, `len=8` password protocol

The intended claim is:

1. detect where keyboard activity starts
2. detect where keyboard activity ends
3. separate `typing_1` from `typing_2`
4. only on `typing_2`, run the existing password classifier and recover the
   password-like content

This keeps the onset section aligned with the current paper scope:
feasibility of controlled continuous-stream attack, rather than long-term
deployment monitoring.

## Detection Protocol

- **Window**: 150ms sliding window (29 samples @ 190Hz)
- **Stride**: 25ms (5 samples) → 40 classifications per second
- **Label**: positive if window centre is within ±30ms of a keystroke
- **Post-processing**: smooth probability curve → peak detection → NMS (±100ms)
- **Model**: 3-layer 1D-CNN, ~25k parameters

## Metrics Reported

### Segment-level (on held-out windows)
- AUC-ROC, Precision, Recall, F1, Accuracy

### Event-level (on continuous streams)
- Event Precision / Recall / F1 at ±50ms tolerance
- Timing error (mean, median, std) in milliseconds
- False alarms per minute
- Tolerance sweep at ±25 / 50 / 75 / 100 ms
- Keyboard-activity episode start / end boundary error
- Whether two keyboard episodes in a mixed stream are correctly separated

### End-to-end (onset → classifier)
- char_top1 / top3 / top5
- sequence_top10 / top50 / top100 (beam-search candidate matching)
- CER
- Δ degradation compared to ground-truth onset baseline
- Missed characters (onset FN) and extra characters (onset FP)

## Integration Notes

- **Zero modification** to existing main-line code
- Reuses `sensor_reader.py`, `spu_backend.py`, `keyboard_listener.py` via import
- Loads password classifier checkpoint without modification
- All onset-specific code lives in the `onset_detection/` directory

## Design Decisions

1. **Sliding window binary classification** (not seq2seq / CTC):
   simpler, more robust with limited data, easy to tune threshold

2. **150ms detection window** (not 300ms classifier window):
   onset detection needs only the impact transient, not the full
   pre/post context that the key-identity classifier uses

3. **NMS radius 100ms**: conservative enough to avoid merging adjacent
   keystrokes in slow controlled typing (IKI typically 500-1500ms in
   our current protocol) while suppressing duplicate detections of
   the same keystroke.  The radius can be tightened later if the
   protocol moves to faster typing speeds.

4. **Session-level splitting**: prevents any data leakage between
   train/val/test splits, matching the main project's protocol

5. **Balanced sampling** (default on): handles the natural ~1:4 pos/neg
   ratio from sliding windows without manual under/oversampling
