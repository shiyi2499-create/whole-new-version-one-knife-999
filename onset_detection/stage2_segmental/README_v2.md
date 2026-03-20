# Stage 2 Segmental v2: Learned Overlapping Windows

## What changed from v1

v1 (`model.py`) partitions the episode into **non-overlapping segments** with shared
boundaries.  This is structurally incompatible with how the classifier was trained:

```
v1 (partition):
  key_1 ---- key_2 ---- key_3
  [seg_1    ][seg_2    ][seg_3    ]   ← no overlap, tiny segments for fast keys

v2 (overlapping windows):
  key_1 ---- key_2 ---- key_3
  [----win_1----]                     ← each key gets ~300ms
         [----win_2----]              ← windows CAN overlap
                [----win_3----]       ← matches classifier's training distribution
```

The classifier was trained on ~300ms overlapping windows.  When two keys are 50ms
apart (common — 28-46% of keys in our data), v1 gives each key only 5-10 frames
and resamples to 57 (11x upsampling → signal destroyed).  v2 gives each key its
own ~60-frame window just like training time.

## Why v1 failed (17.5% vs 58.75% baseline)

| | Fixed-window baseline | v1 partition | v2 overlap (expected) |
|---|---|---|---|
| Per-key window size | ~60 frames (fixed) | 5-60 frames (variable, often tiny) | ~60 frames (learned, near prior) |
| Resample ratio | 60→57 (0.95x) | 5→57 (11.4x!) | ~60→57 (0.95x) |
| Windows overlap | Yes (by design) | No (partition) | Yes (by design) |
| Signal integrity | Preserved | Destroyed for fast keys | Preserved |

## How v2 works

For each key at frame `t_i`, the model predicts:
- **offset** ∈ [-12, +12] frames: small temporal shift (init ≈ 0)
- **width_scale** ∈ [0.5, 2.0]: multiplicative scale of default 300ms window (init ≈ 1.0)

Window boundaries:
```
center = t_i + offset_i
start  = center - width * trigger_ratio        (trigger_ratio = 1/3)
end    = center + width * (1 - trigger_ratio)
```

Key design properties:
1. **At initialization, v2 = fixed-window baseline** (offset=0, scale=1.0)
2. Windows can freely overlap → no signal destruction
3. Gradients flow through differentiable grid_sample → learned refinement
4. Regularization keeps offsets small and widths near prior

## Usage

### Quick smoke test
```bash
python onset_detection/stage2_segmental/scripts/smoke_test_overlap.py
```

### Full training with strong classifier
```bash
python onset_detection/stage2_segmental/scripts/train_gt_overlap.py \
  --input_dir data/raw/mixed_training \
  --output_dir runs/stage2_overlap_gt \
  --classifier_checkpoint results/inception_password_final.pt \
  --classifier_scaler results/inception_password_scaler.npz \
  --device mps
```

### Without external classifier (trains a local one)
```bash
python onset_detection/stage2_segmental/scripts/train_gt_overlap.py \
  --input_dir data/raw/mixed_training \
  --output_dir runs/stage2_overlap_gt \
  --device mps
```

## Output files

Same structure as v1 for easy comparison:
- `best_overlap.pt` — best model checkpoint
- `training_report.json` — final comparison (baseline vs overlap, deltas)
- `overlap_debug.json` — per-episode learned offsets/widths
- `overlap_results.json` — per-episode predictions
- `baseline_results.json` — fixed-window baseline predictions

## What to look for in results

1. **Pre-train metrics should match baseline** — this confirms the initialization
   equivalence.  If they don't match, the encoder is adding too much noise.
2. **Training should not degrade below baseline** — since initialization = baseline,
   any degradation means the learning is going wrong (LR too high, or regularization
   too weak).
3. **Positive delta_top1 / delta_top5** — this is the win condition: learned
   offsets/widths improve over fixed windows.
4. **Learned offsets** — look at `overlap_debug.json`.  If the model finds that
   pre-shift of a few ms helps, you'll see consistent negative offsets (classifier
   benefits from earlier window start).

## Key hyperparameters to tune

| Param | Default | Effect |
|---|---|---|
| `--overlap_lr` | 2e-4 | Learning rate for encoder + heads |
| `--max_offset_ms` | 60.0 | Max learnable shift (± frames) |
| `--max_width_scale` | 2.0 | Max width multiplier |
| `--loss_offset` | 0.10 | Regularization on offset magnitude |
| `--loss_width` | 0.08 | Regularization on width deviation |
| `--loss_consistency` | 0.05 | Smoothness of neighboring offsets |
| `--unfreeze_classifier` | False | Co-train classifier (risky with small data) |

## Files

```
stage2_segmental/
├── model.py                      # v1 (partition, kept for reference)
├── model_v2.py                   # v2 (overlapping windows) ← NEW
├── data.py                       # unchanged
├── metrics.py                    # unchanged
├── __init__.py                   # unchanged
└── scripts/
    ├── train_gt_segmental.py     # v1 training script
    ├── train_gt_overlap.py       # v2 training script ← NEW
    ├── smoke_test.py             # v1 smoke test
    └── smoke_test_overlap.py     # v2 smoke test ← NEW
```
