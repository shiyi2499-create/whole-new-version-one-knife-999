# Stage 2 Open: Variable-Length Password Stream Recovery

**Parallel to `stage2_rebuild` (fixed 5×8 baseline). Does NOT overwrite it.**

## Core Difference

| | stage2_rebuild | stage2_open |
|---|---|---|
| Password count | Fixed 5 | Unknown (model discovers) |
| Password length | Fixed 8 | Unknown (model discovers) |
| Stage 2 output | 5 groups × 8 onsets | Variable groups × variable onsets |
| Decoder | Constrained peak picking | Segment extraction from frame labels |
| Training data | Synthetic 5×8 sessions | Synthetic N×L sessions (N∈[2,8], L∈[4,12]) |

## Architecture

```
Continuous IMU → [Stage 1: coarse region] → Stage 2-Open → [Stage 3: char classify]
                                                │
                                    Frame-wise 3-class TCN:
                                      0 = gap (no keystroke)
                                      1 = keystroke (key being pressed)
                                      2 = separator (inter-password pause)
                                                │
                                    Rule-based decoder:
                                      consecutive '1' runs → onsets
                                      '2' runs → password boundaries
                                      → variable #groups, variable #keys/group
```

## Quick Start

```bash
# 1. Generate variable-length synthetic training data
python scripts/synthesize_open.py \
    --password_dir /path/to/password/len_8 \
    --negative_dir /path/to/onset_negative \
    --output_dir data/synthetic_open \
    --num_sessions 300

# 2. Train the frame-wise classifier
python scripts/train_open.py \
    --data_dir data/synthetic_open \
    --output_dir runs/stage2_open

# 3. Evaluate on mixed2
python scripts/run_e2e_open.py \
    --mixed2_dir /path/to/onset_mixed2 \
    --checkpoint runs/stage2_open/best.pt \
    --output_dir results/open
```
