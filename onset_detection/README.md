# Onset Detection & Keyboard Activity Segmentation Module

Keystroke onset detection **and keyboard activity boundary segmentation** for
the Apple Internal IMU side-channel attack.

This module handles the two critical "where" questions in the attack chain:

1. **Activity Segmentation**: In a continuous mixed stream, *where* does
   keyboard activity start and end? Which episode is free typing vs
   password-style typing?
2. **Onset Detection**: Within a keyboard episode, *when* does each
   individual keystroke occur?

The segmentation output feeds directly into the existing password classifier:
only the `typing_2` (password-style) episodes are passed downstream.

## Architecture Overview

```
continuous IMU stream (~2 min mixed)
  │
  ├─ Activity Segmenter (ActivitySegmentCNN)
  │     ├─ frame-level keyboard-active probability curve
  │     ├─ threshold + merge → Episode boundaries
  │     └─ typing_1/typing_2 classification (demo-protocol heuristic*)
  │
  └─ Onset Detector (OnsetCNN)
        ├─ within typing_2 episode(s) only
        ├─ peak detection + NMS → onset timestamps
        ├─ gap-based grouping → per-password onset groups
        └─ cut 300ms windows → password classifier → top-k recovery

* typing_1 vs typing_2 uses median IKI + keystroke rate thresholds,
  not a learned classifier.  See classify_episodes_by_density() docs.
```

## Two Attack Paths

### Path A: Password session (original)
```
password session → onset detect → classifier → top-k
```

### Path B: Mixed2 stream (new, paper demo target)
```
~2 min mixed stream
  → activity segment → find typing_2 episode
  → onset detect within typing_2
  → gap-based password grouping (no GT)
  → classifier → password recovery
```

## File Map

| File | Role |
|------|------|
| `onset_utils.py` | Peak detection, NMS, event matching, **episode metrics (IoU, boundary error, separation)** |
| `onset_model.py` | OnsetCNN + OnsetCNNLarge + **ActivitySegmentCNN** + energy baseline |
| `onset_preprocessor.py` | Sliding-window dataset builder for **both onset and activity tasks** |
| `onset_dataset.py` | PyTorch Dataset + session-level splitting (supports activity_labels) |
| `train_onset.py` | Training script with **`--task onset` / `--task activity`** modes |
| `onset_collector.py` | Negative + mixed + **mixed2** (structured 2-min protocol) collection |
| `eval_onset.py` | Segment-level + event-level + **episode boundary evaluation** |
| `eval_onset_e2e.py` | Full chain: **Path A** (onset→classify) + **Path B** (segment→typing_2→classify) |

## Quick Start

### Step 1: Collect structured 2-minute mixed streams

```bash
python3 onset_detection/onset_collector.py --mode mixed2 --n-trials 5
```

This records the protocol:
idle → trackpad_move → typing_1 (free) → trackpad_click → shake →
typing_2 (password) → desk_bump → idle

Each trial produces:
- `*_sensor.csv` – continuous IMU data
- `*_events.csv` – keyboard events with timestamps
- `*_activity_log.csv` – ground-truth segment boundaries with labels
- `*_protocol.json` – full protocol definition

### Step 2: Build activity segmentation dataset

```bash
python3 onset_detection/onset_preprocessor.py \
  --task activity \
  --mixed2-dirs data/raw/onset_mixed2 \
  --keyboard-dirs data/raw/single_key data/raw/boost \
  --negative-dirs data/raw/onset_negative \
  --output data/processed/activity_dataset.npz
```

**Data source roles for activity segmentation:**
- `--mixed2-dirs`: **primary source** — provides real activity transition
  boundaries from `activity_log.csv`.  The model learns where keyboard
  activity starts and ends relative to idle/trackpad/shake.
- `--keyboard-dirs`: **supplementary positives** — entire sessions are
  labelled keyboard-active.  Adds positive-class IMU diversity but does
  NOT contribute real start/end boundary supervision (boundaries are the
  session edges, not ecological activity transitions).
- `--negative-dirs`: inactive-class samples, no boundary information.

### Step 3: Train activity segmenter

```bash
python3 onset_detection/train_onset.py \
  --task activity \
  --dataset data/processed/activity_dataset.npz \
  --epochs 80
```

Automatically selects `ActivitySegmentCNN` and saves to
`results/activity_detector.pt` + `results/activity_scaler.npz`.

### Step 4: Build onset detection dataset (unchanged)

```bash
python3 onset_detection/onset_preprocessor.py \
  --keyboard-dirs data/raw/single_key data/raw/boost \
  --password-dirs data/raw/password/len_8 \
  --negative-dirs data/raw/onset_negative \
  --output data/processed/onset_dataset.npz
```

### Step 5: Train onset detector (unchanged)

```bash
python3 onset_detection/train_onset.py \
  --dataset data/processed/onset_dataset.npz \
  --checkpoint results/onset_detector.pt \
  --scaler results/onset_scaler.npz \
  --epochs 80
```

### Step 6: Evaluate activity segmentation

```bash
# Episode boundary metrics on mixed2 streams
python3 onset_detection/eval_onset.py \
  --task activity \
  --mixed2-dirs data/raw/onset_mixed2

# Joint evaluation (both onset + activity)
python3 onset_detection/eval_onset.py \
  --task both \
  --checkpoint results/onset_detector.pt \
  --scaler results/onset_scaler.npz \
  --activity-checkpoint results/activity_detector.pt \
  --activity-scaler results/activity_scaler.npz \
  --mixed2-dirs data/raw/onset_mixed2
```

### Step 7: Full E2E attack chain on mixed2 streams

```bash
python3 onset_detection/eval_onset_e2e.py \
  --onset-checkpoint results/onset_detector.pt \
  --onset-scaler results/onset_scaler.npz \
  --activity-checkpoint results/activity_detector.pt \
  --activity-scaler results/activity_scaler.npz \
  --classifier-checkpoint results/inception_password_final.pt \
  --classifier-scaler results/inception_password_scaler.npz \
  --mixed2-dirs data/raw/onset_mixed2
```

This runs **both** Path A (password sessions) and Path B (mixed2 streams)
and reports four comparison baselines for Path B:

1. **Full E2E**: activity segment → typing_2 → onset → gap-based grouping → classify
   (zero GT information)
2. **GT-segment**: GT typing_2 boundary → onset → gap-based grouping → classify
   (GT boundary only)
3. **GT-aligned**: GT boundary → onset → GT-onset-assisted per-password alignment → classify
   (GT boundary + GT onset timing — explicit oracle baseline)
4. **GT-onset baseline**: GT onset times → classify (full oracle)

This decomposition isolates degradation sources:
- Full E2E − GT baseline = total pipeline degradation
- GT-segment − GT baseline = onset detection + grouping error (segmentation removed)
- GT-aligned − GT baseline = onset detection error only (grouping also removed)

## Mixed2 Protocol Design

The `--mode mixed2` collector implements a fixed ~2-minute protocol:

| Segment | Activity | Duration | Label |
|---------|----------|----------|-------|
| 1 | idle | 10s | idle_1 |
| 2 | trackpad_move | 12s | trackpad_move_1 |
| 3 | **keyboard (free)** | 20s | **typing_1** |
| 4 | trackpad_click | 10s | trackpad_click_1 |
| 5 | shake | 8s | shake_1 |
| 6 | **keyboard (password)** | 25s | **typing_2** |
| 7 | desk_bump | 8s | desk_bump_1 |
| 8 | idle | 10s | idle_2 |

The two keyboard segments differ in:
- **typing_1**: free typing (faster, varied content)
- **typing_2**: controlled password typing (8-char a-z0-9, slower IKI)

This protocol supports the paper claim: the system can distinguish and
correctly segment the password-typing episode from a realistic mixed stream.

## Metrics Reported

### Segment-level (held-out windows)
- AUC-ROC, Precision, Recall, F1, Accuracy
- For both onset and activity tasks

### Event-level (continuous streams)
- Event Precision / Recall / F1 at ±50ms tolerance
- Timing error (mean, median, std) in milliseconds
- False alarms per minute
- Tolerance sweep at ±25 / 50 / 75 / 100 ms

### Episode boundary metrics (NEW)
1. **Start boundary error** (ms): |predicted_start − GT_start|
2. **End boundary error** (ms): |predicted_end − GT_end|
3. **Episode IoU**: temporal intersection-over-union
4. **2-episode separation**: whether typing_1 and typing_2 are
   correctly identified as distinct episodes
5. **Separation accuracy**: fraction of streams where separation succeeds

### End-to-end (onset → classifier)
- char_top1 / top3 / top5
- sequence_top10 / top50 / top100
- CER
- Δ degradation tables: Full E2E / GT-segment / GT-aligned / GT-onset
- Missed characters (onset FN) and extra characters (onset FP) (Path A)

## Design Decisions

1. **Two-stage pipeline** (activity segmenter + onset detector):
   More robust than a single model trying to do both. The activity
   segmenter sees wider context (400ms windows) while the onset
   detector uses precise 150ms windows.

2. **Episode typing style classification — demo-protocol heuristic**:
   `classify_episodes_by_density()` uses two hand-tuned features:
   median IKI > 0.6s and keystroke rate < 2.5 Hz.
   **This is NOT a learned classifier.** It is designed specifically for
   the structured 2-minute mixed2 protocol where typing_1 is continuous
   free text and typing_2 is slow 8-char password entry.  It will not
   generalise to arbitrary typing tasks without re-tuning or replacement
   with a learned episode-level classifier.

3. **ActivitySegmentCNN uses wider kernels** (k=9,7,5,3):
   Needs to capture sustained multi-keystroke energy patterns,
   not just single transients.

4. **Four-way comparison in Path B E2E**:
   Full E2E / GT-segment / GT-aligned / GT-onset baselines.
   Full E2E uses NO ground-truth information — per-password grouping
   is done via gap-based splitting of predicted onsets.
   GT-aligned is the explicit oracle baseline that uses GT onset times
   for per-password alignment, kept separate to avoid contaminating
   the full E2E numbers.

5. **Gap-based per-password grouping**:
   Within a predicted typing_2 episode, detected onsets are split into
   N groups (N = number of passwords from protocol) by finding the
   (N−1) largest inter-onset gaps.  This is a reasonable heuristic
   for the demo protocol where passwords are separated by deliberate
   pauses (Enter key + prompt reading).

6. **Activity dataset: mixed2 as primary boundary source**:
   mixed2 sessions provide real activity-transition boundaries.
   single_key/password sessions are supplementary positives that add
   intra-episode IMU diversity but do NOT provide ecologically valid
   start/end boundary supervision.

7. **Backward compatibility**: All original onset detection functionality
   is preserved. Path A works exactly as before.

## Integration Notes

- **Zero modification** to existing main-line code
- Reuses `sensor_reader.py`, `spu_backend.py`, `keyboard_listener.py` via import
- Loads password classifier checkpoint without modification
- All onset-specific code lives in the `onset_detection/` directory
