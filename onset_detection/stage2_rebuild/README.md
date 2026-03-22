# Stage 2 Rebuild: Password Group Segmentation + Onset Detection

This scaffold is adapted to the current main workspace:

- password data: `data/raw/password/len_8`
- negative data: `data/raw/onset_negative`
- held-out continuous eval: `data/raw/onset_mixed2`
- default signal rate: `190 Hz`

It is intentionally isolated under `onset_detection/stage2_rebuild/` so that
the existing `stage2_claude` and `stage2_gpt54` branches remain intact as
baselines / exploratory branches.

## Architecture

```
Stage 1 (existing) → coarse password region
    ↓
Stage 2A: Group Segmentor (TCN) → 5 password groups
    ↓
Stage 2B: Onset Detector (TCN + Gaussian peak) → 8 onsets per group
    ↓
Stage 3 (existing) → character classification
```

## Quick Start

```bash
pip install torch numpy scipy pandas tqdm tensorboard --break-system-packages

# 1. Generate synthetic mixed sessions from existing data
python onset_detection/stage2_rebuild/scripts/synthesize_mixed.py \
  --password_dir data/raw/password/len_8 \
  --negative_dir data/raw/onset_negative \
  --output_dir data/processed/stage2_synthetic_mixed \
  --num_sessions 100

# 2. Train Stage 2A (Group Segmentor)
python onset_detection/stage2_rebuild/scripts/train_stage2a.py \
  --data_dir data/processed/stage2_synthetic_mixed \
  --output_dir results/stage2_rebuild/stage2a

# 3. Train Stage 2B (Onset Detector)  
python onset_detection/stage2_rebuild/scripts/train_stage2b.py \
  --data_dir data/processed/stage2_synthetic_mixed \
  --output_dir results/stage2_rebuild/stage2b

# 4. Run full E2E pipeline on mixed2
python onset_detection/stage2_rebuild/scripts/run_e2e.py \
  --mixed2_dir data/raw/onset_mixed2 \
  --stage2a_ckpt results/stage2_rebuild/stage2a/best.pt \
  --stage2b_ckpt results/stage2_rebuild/stage2b/best.pt
```

## Directory Structure

```
stage2_rebuild/
├── configs/          # Hyperparameter configs
├── data/             # Data loading & synthesis for the current repo layout
├── models/           # Stage 2A & 2B model architectures
├── trainers/         # Training loops
├── utils/            # Signal processing, metrics, helpers
└── scripts/          # Entry-point scripts
```
