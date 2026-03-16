# Onset Detection Brief

## Goal

Design and implement a first controlled onset detector for continuous IMU monitoring on Apple Silicon Macs.

The detector should answer:

- when did a keyboard keystroke likely happen?
- how can we separate keystrokes from other laptop motion?

## Recommended framing

Treat onset detection as a separate binary / proposal task before key classification:

1. `keyboard onset`
2. `non-keyboard motion`

This is not yet full password recovery. It is the missing stage between:

- continuous non-root IMU collection
- per-keystroke classification / password inference

## Recommended data design

### Positive data

Can reuse existing keyboard-labeled windows from:

- `single_key`
- `password/len_8`

But this is not enough by itself, because onset detection sees continuous context.
Also collect continuous keyboard-only streams.

### Negative / nuisance data

Collect separate sessions for:

- `idle` (laptop untouched)
- `trackpad_move`
- `trackpad_tap`
- `shake`
- `lift_put_down`
- `desk_bump`

These should be recorded as dedicated sessions, not mixed into password labels.

### Mixed continuous streams

Also record long continuous streams with randomized segments, e.g.:

- 20s idle
- 20s trackpad move
- 20s trackpad tap
- 20s shake
- 20s keyboard typing

Important: randomize the order across runs so the model cannot exploit fixed segment position.

## Suggested modeling stages

### Stage A: window classifier / proposal scorer

Use sliding windows over continuous IMU and classify:

- keyboard onset
- not keyboard onset

### Stage B: event extraction

Convert dense scores into onset proposals using:

- thresholding
- non-max suppression / peak picking
- minimum spacing constraints

### Stage C: end-to-end chaining

Feed proposed onsets into the current key/password classifier and measure downstream drop.

## Metrics

- onset precision
- onset recall
- onset F1
- mean absolute timing error
- false alarms per minute or per hour
- downstream key/password top-k after onset proposals

## Recommended first experiment

1. Build nuisance-only sessions
2. Reuse existing keyboard windows as initial positives
3. Add a small amount of continuous keyboard-only streams
4. Train a first onset detector
5. Evaluate on mixed long streams
