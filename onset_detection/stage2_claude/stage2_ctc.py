"""
Stage 2 — Dense CTC: Frame-Level Sequence Model for Password Recovery
======================================================================

Main-line approach for Stage 2.  Instead of detect-then-group, we treat
the coarse password region as a single input sequence and directly predict
the character sequence using CTC (Connectionist Temporal Classification).

Architecture
------------
    coarse IMU region  (T frames × 6 channels, resampled to 190 Hz)
      → 1-D CNN encoder  (strided convolutions, reduces T by ~4×)
        → per-frame 37-class logits  (36 chars + CTC blank)
          → CTC decode
            → character string  (e.g. "a8k3m2p9xr5t7n1q...")
              → split by password_len  → 5 passwords

Training data
-------------
    - password/len_8 sessions: sensor.csv + events.csv
    - Each session has multiple 8-char password attempts
    - For each attempt: cut IMU from first_key - margin to last_key + margin
    - Frame-level target: CTC only needs the ordered character sequence,
      NOT per-frame alignment (that's the whole point of CTC)
    - We also include mixed2 password segments with GT labels

Key references
--------------
    - Graves et al., "Connectionist Temporal Classification" (ICML 2006)
    - Wav2Vec 2.0 / DeepSpeech: CTC for speech-to-text on raw waveforms
    - Myo Armband (IMWUT 2022): end-to-end IMU keystroke inference
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_ONSET_DIR = os.path.dirname(HERE)
if PARENT_ONSET_DIR not in sys.path:
    sys.path.insert(0, PARENT_ONSET_DIR)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise ImportError("PyTorch is required")

try:
    from scipy.signal import resample as scipy_resample
except ImportError:
    raise ImportError("scipy is required")


# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

TARGET_RATE_HZ = 190
N_CHANNELS = 6
SUPPORTED_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789")
# Class layout: 0=CTC_blank, 1-36=chars, 37=separator (inter-password boundary)
BLANK_IDX = 0
SEP_IDX = len(SUPPORTED_CHARS) + 1   # 37
N_CLASSES = len(SUPPORTED_CHARS) + 2  # 36 chars + blank + separator
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(SUPPORTED_CHARS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(SUPPORTED_CHARS)}
IDX_TO_CHAR[SEP_IDX] = '|'  # visual representation of separator

# Margins around password segment for context
PRE_MARGIN_MS = 300
POST_MARGIN_MS = 500

# For single-password training windows
SINGLE_PW_PRE_MS = 200
SINGLE_PW_POST_MS = 400


# ══════════════════════════════════════════════════════════════
# Data loading helpers
# ══════════════════════════════════════════════════════════════

def _resample_to_rate(values: np.ndarray, src_timestamps_ns: np.ndarray,
                      target_rate_hz: int) -> np.ndarray:
    """Resample multi-channel sensor values to a fixed rate."""
    if len(values) < 2:
        return values
    duration_s = (src_timestamps_ns[-1] - src_timestamps_ns[0]) / 1e9
    target_len = max(2, int(duration_s * target_rate_hz))
    out = scipy_resample(values, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def _supported_key(key: str) -> bool:
    return bool(re.match(r'^[a-z0-9]$', (key or '').lower()))


def load_password_events(events_path: str) -> list[list[dict]]:
    """
    Load events.csv, split into per-password sequences by Enter delimiter.
    Returns list of sequences, each sequence is list of {key, timestamp_ns}.
    """
    rows = []
    with open(events_path, newline='') as f:
        for row in csv.DictReader(f):
            if row.get('event_type') != 'press':
                continue
            key = (row.get('key') or '').lower()
            try:
                ts = int(row['timestamp_ns'])
            except (ValueError, KeyError):
                continue
            rows.append({'key': key, 'timestamp_ns': ts})

    sequences = []
    cur = []
    for row in rows:
        key = row['key']
        if key in {'shift', 'capslock', 'ctrl', 'alt', 'cmd', 'tab', 'esc',
                   'left', 'right', 'up', 'down', 'delete'}:
            continue
        if key in {'enter', 'return'}:
            if cur:
                sequences.append(cur)
                cur = []
            continue
        if key in {'space', 'backspace'}:
            continue
        if _supported_key(key):
            cur.append(row)
    if cur:
        sequences.append(cur)
    return sequences


def load_sensor(path: str) -> np.ndarray:
    """Load sensor CSV → (N, 7) array [timestamp_ns, ax, ay, az, gx, gy, gz]."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append([
                int(row['timestamp_ns']),
                float(row['accel_x']), float(row['accel_y']), float(row['accel_z']),
                float(row['gyro_x']), float(row['gyro_y']), float(row['gyro_z']),
            ])
    return np.asarray(rows, dtype=np.float64)


# ══════════════════════════════════════════════════════════════
# Dataset: per-password CTC training examples
# ══════════════════════════════════════════════════════════════

def build_ctc_examples_from_session(
    sensor: np.ndarray,
    sequences: list[list[dict]],
    pre_ms: int = SINGLE_PW_PRE_MS,
    post_ms: int = SINGLE_PW_POST_MS,
    target_rate_hz: int = TARGET_RATE_HZ,
) -> list[dict]:
    """
    Build CTC training examples from a single password session.

    Produces two kinds of examples:
      1. Single-password: one password attempt → target is just the char sequence
      2. Multi-password: consecutive password pairs/triples → target includes
         separator tokens between passwords, so the model learns boundaries

    The separator token (SEP_IDX=37) is a real CTC output class that the model
    must learn to emit between passwords.  This is what makes the segmentation
    explicit rather than a post-hoc length-based split.
    """
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]
    examples = []

    # ── Single-password examples ──
    valid_seqs = []
    for seq in sequences:
        if not seq:
            continue
        chars = [evt['key'] for evt in seq]
        times_ns = [evt['timestamp_ns'] for evt in seq]
        if not all(_supported_key(ch) for ch in chars):
            continue
        valid_seqs.append((chars, times_ns))

        seg_start_ns = min(times_ns) - pre_ms * 1_000_000
        seg_end_ns = max(times_ns) + post_ms * 1_000_000
        i0 = np.searchsorted(ts_ns, seg_start_ns, side='left')
        i1 = np.searchsorted(ts_ns, seg_end_ns, side='right')
        if i1 - i0 < 10:
            continue

        resampled = _resample_to_rate(vals[i0:i1], ts_ns[i0:i1], target_rate_hz)
        target = [CHAR_TO_IDX[ch] for ch in chars]
        examples.append({
            'input': resampled,
            'target': np.array(target, dtype=np.int32),
            'reference': ''.join(chars),
            'n_frames': len(resampled),
            'n_chars': len(chars),
        })

    # ── Multi-password examples (consecutive pairs/triples) with separators ──
    # These teach the model to emit SEP between passwords
    for window_size in (2, 3):
        for start in range(len(valid_seqs) - window_size + 1):
            group = valid_seqs[start:start + window_size]
            all_times = [t for _, times in group for t in times]

            seg_start_ns = min(all_times) - pre_ms * 1_000_000
            seg_end_ns = max(all_times) + post_ms * 1_000_000
            i0 = np.searchsorted(ts_ns, seg_start_ns, side='left')
            i1 = np.searchsorted(ts_ns, seg_end_ns, side='right')
            if i1 - i0 < 10:
                continue

            resampled = _resample_to_rate(vals[i0:i1], ts_ns[i0:i1], target_rate_hz)

            # Build target: chars_pw1 + SEP + chars_pw2 [+ SEP + chars_pw3]
            target = []
            ref_parts = []
            for g_idx, (chars, _) in enumerate(group):
                if g_idx > 0:
                    target.append(SEP_IDX)
                    ref_parts.append('|')
                target.extend(CHAR_TO_IDX[ch] for ch in chars)
                ref_parts.append(''.join(chars))

            examples.append({
                'input': resampled,
                'target': np.array(target, dtype=np.int32),
                'reference': '|'.join(ref_parts).replace('||', '|'),
                'n_frames': len(resampled),
                'n_chars': len(target),
            })

    return examples


def build_ctc_examples_from_mixed2(
    sensor: np.ndarray,
    gt_refined_segs: list[dict],
    events_path: str,
    pre_ms: int = PRE_MARGIN_MS,
    post_ms: int = POST_MARGIN_MS,
    target_rate_hz: int = TARGET_RATE_HZ,
) -> list[dict]:
    """
    Build CTC examples from mixed2 sessions.

    Option A: one example per password (shorter, more samples)
    Option B: one example per entire password segment (longer, multi-password)

    We do Option A for training (consistent with single-password sessions),
    and Option B for inference.
    """
    sequences = load_password_events(events_path)
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]
    examples = []

    # Map sequences to GT segments
    for seg in gt_refined_segs:
        seg_start = int(seg['start_time_ns'])
        seg_end = int(seg['end_time_ns'])
        prompts = seg.get('prompts', [])

        # Find sequences that fall within this segment
        seg_seqs = []
        for seq in sequences:
            seq_times = [evt['timestamp_ns'] for evt in seq]
            if seq_times and seg_start <= min(seq_times) and max(seq_times) <= seg_end:
                seg_seqs.append(seq)

        # Build per-password examples
        for seq_idx, seq in enumerate(seg_seqs):
            chars = [evt['key'] for evt in seq]
            times_ns = [evt['timestamp_ns'] for evt in seq]
            if not all(_supported_key(ch) for ch in chars):
                continue

            cut_start = min(times_ns) - pre_ms * 1_000_000
            cut_end = max(times_ns) + post_ms * 1_000_000
            i0 = np.searchsorted(ts_ns, cut_start, side='left')
            i1 = np.searchsorted(ts_ns, cut_end, side='right')
            if i1 - i0 < 10:
                continue

            resampled = _resample_to_rate(vals[i0:i1], ts_ns[i0:i1], target_rate_hz)
            target = [CHAR_TO_IDX[ch] for ch in chars]

            examples.append({
                'input': resampled,
                'target': np.array(target, dtype=np.int32),
                'reference': ''.join(chars),
                'n_frames': len(resampled),
                'n_chars': len(chars),
            })

    return examples


def build_multi_password_example(
    sensor: np.ndarray,
    gt_refined_segs: list[dict],
    events_path: str,
    pre_ms: int = PRE_MARGIN_MS,
    post_ms: int = POST_MARGIN_MS,
    target_rate_hz: int = TARGET_RATE_HZ,
) -> Optional[dict]:
    """
    Build a single long CTC example covering ALL passwords in a mixed2 session.
    This is used for inference-time evaluation (not training).

    Target is the concatenated character sequence of all passwords.
    """
    sequences = load_password_events(events_path)
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]

    all_chars = []
    all_times = []
    for seg in gt_refined_segs:
        seg_start = int(seg['start_time_ns'])
        seg_end = int(seg['end_time_ns'])
        for seq in sequences:
            seq_times = [evt['timestamp_ns'] for evt in seq]
            if seq_times and seg_start <= min(seq_times) and max(seq_times) <= seg_end:
                for evt in seq:
                    if _supported_key(evt['key']):
                        all_chars.append(evt['key'])
                        all_times.append(evt['timestamp_ns'])

    if not all_chars:
        return None

    cut_start = min(all_times) - pre_ms * 1_000_000
    cut_end = max(all_times) + post_ms * 1_000_000
    i0 = np.searchsorted(ts_ns, cut_start, side='left')
    i1 = np.searchsorted(ts_ns, cut_end, side='right')
    if i1 - i0 < 10:
        return None

    resampled = _resample_to_rate(vals[i0:i1], ts_ns[i0:i1], target_rate_hz)
    target = [CHAR_TO_IDX[ch] for ch in all_chars]

    return {
        'input': resampled,
        'target': np.array(target, dtype=np.int32),
        'reference': ''.join(all_chars),
        'n_frames': len(resampled),
        'n_chars': len(all_chars),
    }


class CTCDataset(Dataset):
    """PyTorch dataset for CTC training. Handles variable-length sequences."""

    def __init__(self, examples: list[dict], means: np.ndarray, stds: np.ndarray,
                 augment: bool = False):
        self.examples = examples
        self.means = means.astype(np.float32)
        self.stds = np.maximum(stds.astype(np.float32), 1e-10)
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x = ex['input'].copy().astype(np.float32)

        # Normalize
        x = (x - self.means) / self.stds

        # Augment
        if self.augment and np.random.random() < 0.5:
            aug = np.random.choice(['noise', 'scale', 'shift', 'ch_drop'])
            if aug == 'noise':
                x += np.random.randn(*x.shape).astype(np.float32) * 0.01
            elif aug == 'scale':
                x *= np.random.uniform(0.85, 1.15)
            elif aug == 'shift':
                shift = np.random.randint(-x.shape[0] // 20, x.shape[0] // 20 + 1)
                x = np.roll(x, shift, axis=0)
            elif aug == 'ch_drop':
                ch = np.random.randint(0, x.shape[1])
                x[:, ch] = 0.0

        target = ex['target'].copy()
        return torch.from_numpy(x), torch.from_numpy(target), x.shape[0], len(target)


def ctc_collate_fn(batch):
    """Collate variable-length sequences for CTC training."""
    inputs, targets, input_lens, target_lens = zip(*batch)
    # Pad inputs to max length in batch
    max_t = max(inp.shape[0] for inp in inputs)
    batch_size = len(inputs)
    n_ch = inputs[0].shape[1]

    padded = torch.zeros(batch_size, max_t, n_ch)
    for i, inp in enumerate(inputs):
        padded[i, :inp.shape[0]] = inp

    targets_cat = torch.cat(targets)
    input_lens = torch.tensor(input_lens, dtype=torch.long)
    target_lens = torch.tensor(target_lens, dtype=torch.long)

    return padded, targets_cat, input_lens, target_lens


# ══════════════════════════════════════════════════════════════
# Model: 1D-CNN encoder → per-frame CTC logits
# ══════════════════════════════════════════════════════════════

class CTCEncoder(nn.Module):
    """
    1D-CNN encoder that preserves temporal resolution (no global pooling).

    Output: (batch, T', n_classes) where T' ≈ T / stride_product.
    Strided convolutions reduce the sequence length, which is essential for
    CTC to work well (CTC needs T' > target_length).
    """

    def __init__(self, n_channels: int = 6, n_classes: int = N_CLASSES,
                 dropout: float = 0.2):
        super().__init__()

        # Encoder: reduce temporal resolution by ~4x total
        # For a ~2s password at 190Hz = ~380 frames → ~95 output frames
        # CTC needs output_len > target_len (8 chars), so 95 >> 8: fine
        self.encoder = nn.Sequential(
            # Block 1: stride 2
            nn.Conv1d(n_channels, 48, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 2: stride 1
            nn.Conv1d(48, 96, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 3: stride 2
            nn.Conv1d(96, 96, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Block 4: stride 1
            nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Per-frame classifier head
        self.head = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, n_classes, kernel_size=1),
        )

        self.total_stride = 4   # stride 2 × stride 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time, channels) — note: channels-last input
        returns: (time', batch, n_classes) — CTC convention
        """
        x = x.permute(0, 2, 1)       # → (B, C, T)
        x = self.encoder(x)           # → (B, 128, T')
        x = self.head(x)              # → (B, n_classes, T')
        x = x.permute(2, 0, 1)       # → (T', B, n_classes) for CTC
        return x

    def compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Compute output sequence lengths after strided convolutions."""
        # Each stride-2 conv: out_len = floor((in_len + 2*pad - kernel) / stride) + 1
        # With our padding, it's approximately in_len // stride
        lengths = input_lengths.clone()
        # Block 1: stride 2, kernel 7, pad 3
        lengths = (lengths + 2 * 3 - 7) // 2 + 1
        # Block 2: stride 1 — no change
        # Block 3: stride 2, kernel 5, pad 2
        lengths = (lengths + 2 * 2 - 5) // 2 + 1
        # Block 4: stride 1 — no change
        return lengths.clamp(min=1)


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

def compute_scaler(examples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean/std across all training examples."""
    all_vals = np.concatenate([ex['input'] for ex in examples], axis=0)
    means = all_vals.mean(axis=0)
    stds = all_vals.std(axis=0)
    return means.astype(np.float32), np.maximum(stds.astype(np.float32), 1e-10)


def train_ctc_model(
    train_examples: list[dict],
    val_examples: list[dict],
    checkpoint_path: str,
    scaler_path: str,
    device: torch.device,
    epochs: int = 120,
    batch_size: int = 16,
    lr: float = 3e-4,
    patience: int = 25,
):
    """Train the CTC model."""
    means, stds = compute_scaler(train_examples)
    np.savez(scaler_path, means=means, stds=stds)
    print(f"  Scaler saved → {scaler_path}")

    train_ds = CTCDataset(train_examples, means, stds, augment=True)
    val_ds = CTCDataset(val_examples, means, stds, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          collate_fn=ctc_collate_fn, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=ctc_collate_fn, num_workers=0)

    model = CTCEncoder(n_channels=N_CHANNELS, n_classes=N_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ctc_loss_fn = nn.CTCLoss(blank=BLANK_IDX, reduction='mean', zero_infinity=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CTC model: {n_params:,} parameters")
    print(f"  Train: {len(train_examples)} examples, Val: {len(val_examples)} examples")
    print(f"  Training for up to {epochs} epochs, patience={patience}")

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for padded, targets_cat, input_lens, target_lens in train_dl:
            padded = padded.to(device)
            targets_cat = targets_cat.to(device)

            log_probs = model(padded)  # (T', B, C)
            output_lens = model.compute_output_lengths(input_lens).to(device)
            log_probs = F.log_softmax(log_probs, dim=2)

            loss = ctc_loss_fn(log_probs, targets_cat, output_lens, target_lens)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss_sum += loss.item() * padded.size(0)
            train_n += padded.size(0)

        scheduler.step()
        train_loss = train_loss_sum / max(train_n, 1)

        # ── Val ──
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_correct_chars = 0
        val_total_chars = 0
        with torch.no_grad():
            for padded, targets_cat, input_lens, target_lens in val_dl:
                padded = padded.to(device)
                targets_cat = targets_cat.to(device)

                log_probs = model(padded)
                output_lens = model.compute_output_lengths(input_lens).to(device)
                log_probs_sm = F.log_softmax(log_probs, dim=2)

                loss = ctc_loss_fn(log_probs_sm, targets_cat, output_lens, target_lens)
                val_loss_sum += loss.item() * padded.size(0)
                val_n += padded.size(0)

                # Greedy decode for monitoring
                preds = log_probs_sm.argmax(dim=2).permute(1, 0)  # (B, T')
                offset = 0
                for b in range(padded.size(0)):
                    tlen = int(target_lens[b])
                    ref_indices = targets_cat[offset:offset + tlen].cpu().tolist()
                    offset += tlen
                    pred_seq = preds[b, :int(output_lens[b])].cpu().tolist()
                    decoded = _greedy_decode(pred_seq)
                    for i, ref_idx in enumerate(ref_indices):
                        if i < len(decoded) and decoded[i] == ref_idx:
                            val_correct_chars += 1
                    val_total_chars += tlen

        val_loss = val_loss_sum / max(val_n, 1)
        val_char_acc = val_correct_chars / max(val_total_chars, 1)

        if epoch <= 5 or epoch % 5 == 0 or val_loss < best_val_loss:
            print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_char_acc={val_char_acc:.1%}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                'model_state': model.state_dict(),
                'n_classes': N_CLASSES,
                'n_channels': N_CHANNELS,
                'chars': SUPPORTED_CHARS,
                'blank_idx': BLANK_IDX,
                'total_stride': model.total_stride,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
            }, checkpoint_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Checkpoint → {checkpoint_path}")
    return model


# ══════════════════════════════════════════════════════════════
# Decoding
# ══════════════════════════════════════════════════════════════

def _greedy_decode(pred_indices: list[int]) -> list[int]:
    """CTC greedy decode: collapse repeats and remove blanks."""
    result = []
    prev = BLANK_IDX
    for idx in pred_indices:
        if idx != prev and idx != BLANK_IDX:
            result.append(idx)
        prev = idx
    return result


def greedy_decode_to_string(pred_indices: list[int]) -> str:
    """CTC greedy decode to character string. SEP tokens become '|'."""
    decoded = _greedy_decode(pred_indices)
    return ''.join(IDX_TO_CHAR.get(idx, '?') for idx in decoded)


def beam_decode_to_string(log_probs: np.ndarray, beam_width: int = 20) -> str:
    """
    CTC beam search decode.
    log_probs: (T, C) log probabilities per frame.
    SEP tokens are decoded as '|'.
    """
    T, C = log_probs.shape
    beam = [((), 0.0)]

    for t in range(T):
        new_beam = {}
        for prefix, score in beam:
            for c in range(C):
                lp = float(log_probs[t, c])
                if c == BLANK_IDX:
                    key = prefix
                else:
                    if prefix and prefix[-1] == c:
                        key = prefix
                    else:
                        key = prefix + (c,)
                new_score = score + lp
                if key not in new_beam or new_beam[key] < new_score:
                    new_beam[key] = new_score

        beam = sorted(new_beam.items(), key=lambda x: -x[1])[:beam_width]

    if not beam:
        return ''
    best_prefix = beam[0][0]
    return ''.join(IDX_TO_CHAR.get(idx, '?') for idx in best_prefix)


# ══════════════════════════════════════════════════════════════
# Segmentation: split decoded string into passwords
# ══════════════════════════════════════════════════════════════

def split_by_separator(
    decoded: str,
    n_passwords: int = 5,
    password_len: int = 8,
) -> list[str]:
    """
    Split the CTC-decoded string into individual passwords.

    Primary strategy: split on '|' (separator tokens the model learned to emit).
    Fallback: if no separators found, split by password_len.

    Each resulting password is trimmed/padded to password_len.
    """
    # Strategy 1: split on separator
    if '|' in decoded:
        parts = [p for p in decoded.split('|') if p]  # drop empty
    else:
        # Strategy 2: no separators found → equal-length split (fallback)
        total = n_passwords * password_len
        s = decoded[:total]
        parts = []
        for i in range(n_passwords):
            start = i * password_len
            end = start + password_len
            if start < len(s):
                parts.append(s[start:end])

    # Pad / trim to exactly n_passwords of password_len
    passwords = []
    for i in range(n_passwords):
        if i < len(parts):
            pw = parts[i][:password_len].ljust(password_len, '?')
        else:
            pw = '?' * password_len
        passwords.append(pw)

    return passwords


# ══════════════════════════════════════════════════════════════
# Inference on coarse region
# ══════════════════════════════════════════════════════════════

def load_ctc_model(checkpoint_path: str, scaler_path: str, device: torch.device):
    """Load trained CTC model and scaler."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = CTCEncoder(
        n_channels=ckpt.get('n_channels', N_CHANNELS),
        n_classes=ckpt.get('n_classes', N_CLASSES),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    scaler = np.load(scaler_path)
    means = scaler['means'].astype(np.float32)
    stds = np.maximum(scaler['stds'].astype(np.float32), 1e-10)

    return model, means, stds, ckpt


def infer_on_region(
    model: nn.Module,
    sensor: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    device: torch.device,
    region_start_s: float,
    region_end_s: float,
    target_rate_hz: int = TARGET_RATE_HZ,
    beam_width: int = 20,
) -> tuple[str, np.ndarray]:
    """
    Run CTC inference on a coarse password region.

    Returns:
        decoded_string: full decoded character sequence
        log_probs: (T', n_classes) frame-level log probabilities
    """
    ts_ns = sensor[:, 0]
    vals = sensor[:, 1:]
    mask = (ts_ns >= region_start_s * 1e9) & (ts_ns <= region_end_s * 1e9)
    if mask.sum() < 10:
        return '', np.array([])

    region_vals = vals[mask]
    region_ts = ts_ns[mask]
    resampled = _resample_to_rate(region_vals, region_ts, target_rate_hz)

    # Normalize
    x = (resampled.astype(np.float32) - means) / stds
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, T, 6)

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)  # (T', 1, C)
        log_probs = F.log_softmax(logits, dim=2)
        log_probs_np = log_probs.squeeze(1).cpu().numpy()  # (T', C)

    # Decode
    if beam_width > 1:
        decoded = beam_decode_to_string(log_probs_np, beam_width=beam_width)
    else:
        pred_indices = log_probs_np.argmax(axis=1).tolist()
        decoded = greedy_decode_to_string(pred_indices)

    return decoded, log_probs_np


# ══════════════════════════════════════════════════════════════
# Pipeline integration: run_stage2_ctc
# ══════════════════════════════════════════════════════════════

def run_stage2_ctc(
    sensor: np.ndarray,
    coarse_regions: list,
    ctc_model: nn.Module,
    ctc_means: np.ndarray,
    ctc_stds: np.ndarray,
    device: torch.device,
    n_passwords: int = 5,
    password_len: int = 8,
    beam_width: int = 20,
) -> tuple[list[str], dict]:
    """
    Stage 2 main-line: Dense CTC inference on coarse password regions.

    Returns:
        passwords: list of decoded password strings
        debug: debug info
    """
    all_decoded = ''
    all_log_probs = []
    debug = {
        'method': 'dense_ctc',
        'n_coarse_regions': len(coarse_regions),
        'regions': [],
    }

    for region in coarse_regions:
        decoded, log_probs = infer_on_region(
            ctc_model, sensor, ctc_means, ctc_stds, device,
            region.start_s, region.end_s,
            beam_width=beam_width,
        )
        all_decoded += decoded
        if len(log_probs) > 0:
            all_log_probs.append(log_probs)
        debug['regions'].append({
            'start_s': region.start_s,
            'end_s': region.end_s,
            'decoded_len': len(decoded),
            'decoded': decoded[:80],
            'n_output_frames': len(log_probs) if len(log_probs) > 0 else 0,
        })

    debug['full_decoded'] = all_decoded
    debug['full_decoded_len'] = len(all_decoded)

    passwords = split_by_separator(all_decoded, n_passwords, password_len)
    debug['passwords'] = passwords
    debug['n_separators_found'] = all_decoded.count('|')
    debug['used_separator_split'] = '|' in all_decoded

    return passwords, debug


# ══════════════════════════════════════════════════════════════
# Evaluation helpers
# ══════════════════════════════════════════════════════════════

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def score_ctc_passwords(
    predicted_passwords: list[str],
    gt_passwords: list[str],
) -> dict:
    """Score CTC-decoded passwords against ground truth."""
    n_seqs = len(gt_passwords)
    n_chars = sum(len(pw) for pw in gt_passwords)

    total_correct = 0
    total_edits = 0
    exact = 0

    per_pw = []
    for pred, ref in zip(predicted_passwords, gt_passwords):
        # Character accuracy (positional)
        correct = sum(1 for a, b in zip(pred, ref) if a == b)
        total_correct += correct
        ed = levenshtein(pred, ref)
        total_edits += ed
        if pred == ref:
            exact += 1
        per_pw.append({
            'ref': ref,
            'hyp': pred,
            'char_acc': correct / max(len(ref), 1),
            'cer': ed / max(len(ref), 1),
        })

    return {
        'n_passwords': n_seqs,
        'n_chars': n_chars,
        'char_accuracy': total_correct / max(n_chars, 1),
        'cer': total_edits / max(n_chars, 1),
        'exact_match': exact,
        'exact_match_rate': exact / max(n_seqs, 1),
        'per_password': per_pw,
    }


# ══════════════════════════════════════════════════════════════
# CLI: Train + Evaluate
# ══════════════════════════════════════════════════════════════

PART_RE = re.compile(r'_part(\d+)_')


def _parse_part(path: str) -> int:
    m = PART_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else -1


def discover_password_sessions(dirs: list[str]) -> list[str]:
    sessions = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in sorted(files):
                if f.startswith('.') or f.startswith('._'):
                    continue
                if f.endswith('_sensor.csv') and '_free_type_' in f:
                    prefix = os.path.join(root, f.replace('_sensor.csv', ''))
                    if os.path.exists(prefix + '_events.csv'):
                        sessions.append(prefix)
    return sorted(sessions)


def main():
    p = argparse.ArgumentParser(description='Dense CTC Stage 2: train + evaluate')
    p.add_argument('--project-root', default='')
    p.add_argument('--password-dirs', nargs='+', default=['data/raw/password/len_8'])
    p.add_argument('--mixed2-dirs', nargs='*', default=['data/raw/onset_mixed2'])
    p.add_argument('--test-parts', nargs='+', type=int, default=[17, 18, 19, 20])
    p.add_argument('--checkpoint', default='results/ctc_stage2.pt')
    p.add_argument('--scaler', default='results/ctc_stage2_scaler.npz')
    p.add_argument('--report', default='results/ctc_stage2_report.json')
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--no-train', action='store_true')
    p.add_argument('--beam-width', type=int, default=20)
    args = p.parse_args()

    if args.project_root:
        root = os.path.abspath(args.project_root)
        for attr in ['checkpoint', 'scaler', 'report']:
            v = getattr(args, attr)
            if not os.path.isabs(v):
                setattr(args, attr, os.path.join(root, v))
        args.password_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                              for d in args.password_dirs]
        args.mixed2_dirs = [os.path.join(root, d) if not os.path.isabs(d) else d
                            for d in (args.mixed2_dirs or [])]
        # Add project root to path for imports
        if root not in sys.path:
            sys.path.insert(0, root)
        phase3 = os.path.join(root, 'phase3_password_inception')
        if phase3 not in sys.path:
            sys.path.insert(0, phase3)

    req = (args.device or 'auto').lower()
    if req == 'auto':
        if torch.cuda.is_available():
            req = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            req = 'mps'
        else:
            req = 'cpu'
    device = torch.device(req)
    print(f'Device: {device}')
    print(f'CTC classes: {N_CLASSES} (36 chars + blank)')

    # ── Discover sessions ──
    all_sessions = discover_password_sessions(args.password_dirs)
    test_parts = set(args.test_parts)
    train_sessions = [s for s in all_sessions if _parse_part(s) not in test_parts]
    val_sessions = [s for s in all_sessions if _parse_part(s) in test_parts]
    print(f'Password sessions: {len(train_sessions)} train, {len(val_sessions)} val/test')

    # ── Build training data ──
    if not args.no_train:
        print('\nBuilding CTC training data...')
        train_examples = []
        for sess in train_sessions:
            sensor = load_sensor(sess + '_sensor.csv')
            sequences = load_password_events(sess + '_events.csv')
            exs = build_ctc_examples_from_session(sensor, sequences)
            train_examples.extend(exs)
        print(f'  Train examples: {len(train_examples)}')

        val_examples = []
        for sess in val_sessions:
            sensor = load_sensor(sess + '_sensor.csv')
            sequences = load_password_events(sess + '_events.csv')
            exs = build_ctc_examples_from_session(sensor, sequences)
            val_examples.extend(exs)
        print(f'  Val examples: {len(val_examples)}')

        if not train_examples:
            print('ERROR: No training examples found.')
            return

        # NOTE: mixed2 is intentionally excluded from training.
        # It is our held-out evaluation set for the full Path B pipeline.
        # CTC training uses only password/len_8 sessions.

        print(f'  Total train: {len(train_examples)}, val: {len(val_examples)}')
        print(f'  (mixed2 excluded from training — eval only)')
        frame_lens = [ex['n_frames'] for ex in train_examples]
        char_lens = [ex['n_chars'] for ex in train_examples]
        print(f'  Frame lengths: {np.min(frame_lens)}-{np.max(frame_lens)} '
              f'(mean {np.mean(frame_lens):.0f})')
        print(f'  Char lengths: {np.min(char_lens)}-{np.max(char_lens)}')

        # ── Train ──
        print('\nTraining CTC model...')
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        train_ctc_model(
            train_examples, val_examples,
            args.checkpoint, args.scaler, device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
        )

    # ── Evaluate on password sessions ──
    if os.path.exists(args.checkpoint) and os.path.exists(args.scaler):
        print('\nEvaluating CTC model on val sessions...')
        model, means, stds, ckpt = load_ctc_model(args.checkpoint, args.scaler, device)
        print(f'  Loaded checkpoint (epoch {ckpt.get("epoch", "?")}, '
              f'val_loss={ckpt.get("best_val_loss", "?"):.4f})')

        all_refs = []
        all_hyps = []
        for sess in val_sessions:
            sensor = load_sensor(sess + '_sensor.csv')
            sequences = load_password_events(sess + '_events.csv')
            for seq in sequences:
                chars = [evt['key'] for evt in seq if _supported_key(evt['key'])]
                times = [evt['timestamp_ns'] for evt in seq if _supported_key(evt['key'])]
                if not chars:
                    continue
                ref = ''.join(chars)
                # Infer on the password region
                start_s = (min(times) - SINGLE_PW_PRE_MS * 1e6) / 1e9
                end_s = (max(times) + SINGLE_PW_POST_MS * 1e6) / 1e9
                decoded, _ = infer_on_region(
                    model, sensor, means, stds, device,
                    start_s, end_s, beam_width=args.beam_width)
                all_refs.append(ref)
                all_hyps.append(decoded)

        if all_refs:
            total_chars = sum(len(r) for r in all_refs)
            correct = sum(sum(1 for a, b in zip(h, r) if a == b)
                          for h, r in zip(all_hyps, all_refs))
            edits = sum(levenshtein(h, r) for h, r in zip(all_hyps, all_refs))
            exact = sum(1 for h, r in zip(all_hyps, all_refs) if h == r)

            print(f'\n  === CTC Val Results ({len(all_refs)} passwords) ===')
            print(f'  char_accuracy:  {correct / max(total_chars, 1):.1%}')
            print(f'  CER:            {edits / max(total_chars, 1):.1%}')
            print(f'  exact_match:    {exact}/{len(all_refs)} ({exact/max(len(all_refs),1):.1%})')
            print(f'  Examples:')
            for ref, hyp in zip(all_refs[:8], all_hyps[:8]):
                match = '✓' if ref == hyp else '✗'
                print(f'    {match} ref={ref}  hyp={hyp}')

        report = {
            'n_val_passwords': len(all_refs),
            'char_accuracy': correct / max(total_chars, 1) if all_refs else 0,
            'cer': edits / max(total_chars, 1) if all_refs else 1,
            'exact_match_rate': exact / max(len(all_refs), 1) if all_refs else 0,
            'examples': [{'ref': r, 'hyp': h} for r, h in zip(all_refs[:20], all_hyps[:20])],
        }
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'\n  Report → {args.report}')


if __name__ == '__main__':
    main()
