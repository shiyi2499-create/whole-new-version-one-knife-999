# Stage 2 Episode: Migration Guide

## What Changed and Why

### The Problem with stage2_open (3-class)

The old approach used 3 frame classes:
- 0 = gap (silence within a password or context)
- 1 = keystroke (typing activity)
- 2 = separator (silence between passwords)

**Why it fails:** Classes 0 and 2 have *identical IMU signatures* — both are
"no typing activity." The only difference is duration, but the frame-level model
sees local context, not global timing. This forces the model to hallucinate
separators wherever it sees a slightly longer pause, causing over-segmentation.

### The Episode-Based Solution (2-class)

New approach:
- Frame model: 2 classes only — **silence (0)** and **typing (1)**
- Episode detection: post-hoc rule — merge typing runs separated by gaps
  shorter than `episode_gap_ms` (default 600ms)

**Why this works:** The frame model only needs to answer "is there keystroke
activity here?" — a much easier task with clear IMU signatures. The episode
boundary question ("is this gap between passwords or within one password?")
is answered by a simple duration threshold, which is where the actual
distinguishing information lives.

## File Mapping

```
stage2_open/                     →  stage2_episode/
  configs/config.py              →  configs/config.py
    3 classes, DecoderConfig         2 classes, EpisodeConfig ←NEW
  models/tcn.py (OpenTCN)        →  models/tcn.py (EpisodeTCN)
    num_classes=3                    num_classes=2
  models/losses.py               →  models/losses.py
    3 class weights                  2 class weights
  utils/decoder.py               →  utils/decoder.py ←REWRITTEN
    separator-based splitting        gap-based episode merging
  utils/metrics.py               →  utils/metrics.py
    groups terminology               episodes terminology, 2-class acc
  data/synthesis.py              →  data/synthesis.py
    3-class labels                   2-class labels, no separator
  data/datasets.py               →  data/datasets.py
    3-class                          2-class, backward-compatible with old data
  data/loaders.py                →  data/loaders.py (unchanged)
  trainers/trainer.py            →  trainers/trainer.py
    3-class loss                     2-class loss + episode eval
  scripts/                       →  scripts/ (all adapted)
```

## Backward Compatibility

### Dataset compatibility
`EpisodeFrameDataset` auto-converts old 3-class data:
- Labels with value 2 (separator) → 0 (silence)
- Labels with value 1 (keystroke) → 1 (typing)

So you can train on existing mixed_training data without re-building.

### API compatibility
`episodes_to_groups()` in decoder.py converts episode format to the old
group format expected by Stage 3 and eval code.

## Pipeline Integration

### Stage 1 → Stage 2 (unchanged)
Stage 1 produces a coarse password region (start/end timestamps).
Stage 2 episode receives this region's IMU and finds episodes within it.
No interface change needed.

### Stage 2 → Stage 3 (minimal change)
Old: Stage 2 produces exactly N groups, each with fixed-length onsets.
New: Stage 2 produces variable number of episodes, each with variable onsets.

Stage 3 needs to:
1. Accept a list of episodes (not a fixed count)
2. Process each episode independently
3. This is likely already how it works if it takes groups as input.

Use `episodes_to_groups()` for the adapter.

## Key Hyperparameter

`episode_gap_ms` (default: 600ms) is THE critical threshold.

To tune it, use the sweep mode:
```bash
python scripts/run_e2e_episode.py \
    --mixed2_dir data/raw/onset_mixed2 \
    --checkpoint runs/stage2_episode/best.pt \
    --sweep_gap 300,400,500,600,700,800,1000
```

Expected range: 400-800ms. Lower values → more episodes (over-split),
higher values → fewer episodes (under-split/merge).

## Training Recipe

```bash
# 1. Build real dataset from existing mixed_training recordings
python scripts/build_real_episode_dataset.py \
    --input_dir data/raw/mixed_training \
    --output_dir data/stage2_episode_real

# 2. Optionally generate synthetic data
python scripts/synthesize_episode.py \
    --password_dir data/raw/password \
    --neg_dir data/raw/onset_negative \
    --output_dir data/stage2_episode_synth

# 3. Train on combined data
python scripts/train_episode.py \
    --data_dir data/stage2_episode_synth,data/stage2_episode_real \
    --output_dir runs/stage2_episode \
    --episode_gap_ms 600

# 4. Evaluate
python scripts/run_e2e_episode.py \
    --mixed2_dir data/raw/onset_mixed2 \
    --checkpoint runs/stage2_episode/best.pt \
    --sweep_gap 300,400,500,600,700,800,1000
```

## Collector Changes (Optional)

The existing `onset_collector.py --mode mixed_training` already produces
data that works with this pipeline. Optional enhancements in
`onset_collector_patch.py`:
- Variable password lengths (4-12 chars instead of fixed 8)
- Variable password count per trial (3-7 instead of fixed 5)
