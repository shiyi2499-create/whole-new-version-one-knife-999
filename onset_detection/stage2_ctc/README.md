# Stage 2 CTC: Frame-level Character Decoding

Replaces the onset-detection pipeline with direct frame-level character
prediction + CTC decoding.

## Why

The onset detection → window cutting → per-window classification pipeline
has a structural information bottleneck: even with perfect onset positions
(GT baseline), cutting fixed windows and resampling loses 33 percentage
points of accuracy compared to oracle windows (73% → 40% char_top1).

This module eliminates the window-cutting step entirely. The model
predicts `P(char|frame)` at every timestep, and CTC decoding recovers
the character sequence directly.

## Prerequisites

```bash
# Python 3.10+ on macOS (tested with Apple Silicon / MPS)
pip install torch numpy scipy
```

## Quick Start

```bash
cd onset_detection/stage2_ctc

# 1. Verify everything works
python scripts/sanity_test.py --device mps

# 2. Build dataset from existing mixed_training + password data
python scripts/build_dataset.py \
    --mixed_training_dir ../../data/raw/mixed_training \
    --password_dir ../../data/raw/password \
    --neg_dir ../../data/raw/onset_negative \
    --output_dir ../../data/stage2_ctc \
    --num_synth 600

# 3. Train (with optional onset backbone init)
python scripts/train.py \
    --data_dir ../../data/stage2_ctc \
    --output_dir ../../runs/stage2_ctc \
    --device mps \
    --onset_checkpoint ../../runs/stage2_episode/best.pt

# 4. Evaluate on mixed_training sessions
python scripts/eval_e2e.py \
    --mixed_dir ../../data/raw/mixed_training \
    --checkpoint ../../runs/stage2_ctc/best.pt \
    --output_dir ../../results/stage2_ctc_eval \
    --device mps
```

## Architecture

```
episode IMU [T, 6]
    │
    ▼  preprocess (add magnitude → 8ch, normalize)
[T, 8]
    │
    ▼  Conv1d input projection
[T, 128]
    │
    ▼  12× DilatedResBlock (dilation 1,2,4,...,2048,1,2,4,...,2048)
[T, 128]
    │
    ▼  Character head (Conv1d → ReLU → Dropout → Conv1d)
[T, 38]  ← logits over {blank, a-z, 0-9, <unk>}
    │
    ├──▶ Frame-level CE loss (supervised by per-key timestamps)
    └──▶ CTC loss (supervised by character sequence)
```

## Loss Design

**Frame-level CE** (primary, weight=1.0):
- Each keystroke center frame is labeled with its character
- All other frames are labeled blank
- Gaussian soft weights focus the loss around keystrokes
- Blank frames are downweighted (0.15×) to handle class imbalance

**CTC loss** (auxiliary, weight=0.3):
- Only sees the character sequence (no timestamps)
- Provides global sequence consistency
- Prevents independent frame predictions from producing incoherent sequences

The key insight: we have per-key timestamps, which is much stronger
supervision than standard CTC (e.g. ASR). So frame CE is the primary
signal, and CTC just adds sequence-level consistency.

## File Structure

```
stage2_ctc/
├── configs/
│   └── config.py          # ModelConfig, TrainConfig, SignalConfig, DataConfig
├── models/
│   ├── frame_ctc.py       # FrameCTCModel (TCN backbone + char head)
│   └── losses.py          # FrameCTCLoss (frame CE + CTC)
├── data/
│   ├── datasets.py        # CTCEpisodeDataset + build_frame_targets()
│   ├── loaders.py         # SessionLoader (reused from stage2_episode)
│   └── synthesis.py       # CTCSynthesizer (adapted from stage2_episode)
├── trainers/
│   └── trainer.py         # CTCTrainer
├── utils/
│   ├── vocab.py           # VOCAB, CHAR_TO_IDX, char_index()
│   ├── signal_processing.py
│   ├── decode.py          # greedy_decode, prefix_beam_search
│   └── metrics.py         # CER, char_topk, aggregate
├── scripts/
│   ├── sanity_test.py     # Run first: verifies imports + shapes + device
│   ├── build_dataset.py   # Build .npz dataset from raw sessions
│   ├── train.py           # Train the model
│   └── eval_e2e.py        # End-to-end evaluation
└── README.md
```

## Relation to Existing Pipeline

| Component | stage2_episode (old) | stage2_ctc (new) |
|-----------|---------------------|------------------|
| Frame model | 2-class (typing/silence) | 38-class (blank + chars) |
| Onset head | Gaussian impulse detector | (not needed) |
| Decoder | peak-pick → cut window → classify | CTC greedy / beam search |
| Per-key info used | onset position only | onset position + char label |
| Window cutting | Required (300ms window) | Eliminated |
| Classifier | Separate InceptionTime model | Integrated in frame model |

The onset TCN backbone weights transfer directly — the model already knows
"where are keystrokes" and just needs to learn "what character is there".

## Expected Metrics

The success criterion for this approach:

1. **char_top1 at GT positions > 40%**: This would beat the GT baseline
   of the old pipeline (which loses accuracy in the window-cutting step).
2. **CER < 50%**: This would beat the GT baseline CER of 59.8%.
3. If both are met, proceed to integrate with Stage 1 episode detection.
