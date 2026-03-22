#!/usr/bin/env python3
"""
Sanity test: verify the CTC pipeline end-to-end with fake data.

Creates a tiny synthetic dataset, trains for a few epochs, and decodes.
Designed to catch import errors, shape mismatches, and device issues
before running on real data.

Usage:
    python scripts/sanity_test.py
    python scripts/sanity_test.py --device mps
"""
import sys
import os
import tempfile
import json

import numpy as np
import torch

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from utils.vocab import VOCAB, CHAR_TO_IDX, NUM_CLASSES, char_index
from utils.decode import greedy_decode, prefix_beam_search
from utils.signal_processing import preprocess
from data.datasets import build_frame_targets, CTCEpisodeDataset
from models.frame_ctc import FrameCTCModel
from models.losses import FrameCTCLoss


def make_fake_episode(T=200, num_keys=8, sr=100):
    """Create one fake episode with random IMU and known char sequence."""
    imu = np.random.randn(T, 6).astype(np.float32) * 0.1

    # Place keystrokes evenly
    spacing = T // (num_keys + 2)
    chars = [VOCAB[1 + i % 36] for i in range(num_keys)]  # cycle through a-z, 0-9

    key_events = []
    for i, ch in enumerate(chars):
        t = spacing * (i + 1)
        # Add a bump at keystroke position so the model has something to learn
        imu[max(0, t-2):min(T, t+3)] += np.random.randn(min(5, T - max(0, t-2)), 6) * 0.5
        key_events.append({'ts_frame': t, 'char': ch})

    sigma = max(1.0, 20.0 / 1000.0 * sr)
    hard_targets, soft_weights, ctc_target = build_frame_targets(T, key_events, sigma)

    return {
        'imu': imu,
        'hard_targets': hard_targets,
        'soft_weights': soft_weights,
        'ctc_target': np.array(ctc_target, dtype=np.int64),
        'password': ''.join(chars),
    }


def test_vocab():
    print("Test vocab...")
    assert char_index('a') == 1
    assert char_index('z') == 26
    assert char_index('0') == 27
    assert char_index('9') == 36
    assert char_index('enter') == 37  # UNK
    assert NUM_CLASSES == 38
    print("  OK")


def test_frame_targets():
    print("Test frame targets...")
    T = 100
    events = [
        {'ts_frame': 20, 'char': 'a'},
        {'ts_frame': 50, 'char': '5'},
        {'ts_frame': 80, 'char': 'z'},
    ]
    ht, sw, ct = build_frame_targets(T, events, sigma_frames=2.0)
    assert ht.shape == (T,)
    assert ht[20] == CHAR_TO_IDX['a']
    assert ht[50] == CHAR_TO_IDX['5']
    assert ht[80] == CHAR_TO_IDX['z']
    assert ht[0] == 0  # blank
    assert sw[20] > sw[0]  # higher weight at keystroke
    assert len(ct) == 3
    print(f"  hard_targets sum non-blank: {(ht > 0).sum()}")
    print(f"  soft_weights range: [{sw.min():.3f}, {sw.max():.3f}]")
    print("  OK")


def test_model_forward(device):
    print(f"Test model forward ({device})...")
    model = FrameCTCModel(in_ch=8, hidden=32, num_layers=4, kernel=3,
                          dropout=0.1, num_classes=NUM_CLASSES).to(device)
    x = torch.randn(2, 8, 150).to(device)
    logits = model(x)
    assert logits.shape == (2, NUM_CLASSES, 150), f"Got {logits.shape}"
    print(f"  output shape: {logits.shape}")
    print("  OK")


def test_loss(device):
    print(f"Test loss ({device})...")
    criterion = FrameCTCLoss(num_classes=NUM_CLASSES).to(device)

    B, C, T = 2, NUM_CLASSES, 100
    logits = torch.randn(B, C, T).to(device)
    hard_targets = torch.zeros(B, T, dtype=torch.long).to(device)
    hard_targets[0, 20] = 1  # 'a'
    hard_targets[0, 50] = 2  # 'b'
    soft_weights = torch.ones(B, T).to(device)
    mask = torch.ones(B, T).to(device)
    ctc_targets = torch.tensor([1, 2, 1, 3], dtype=torch.long).to(device)
    ctc_target_lengths = torch.tensor([2, 2], dtype=torch.long).to(device)
    input_lengths = torch.tensor([T, T], dtype=torch.long).to(device)

    result = criterion(logits, hard_targets, soft_weights, mask,
                       ctc_targets, ctc_target_lengths, input_lengths)
    print(f"  loss={result['loss'].item():.4f} "
          f"ce={result['frame_ce'].item():.4f} "
          f"ctc={result['ctc'].item():.4f}")
    assert not torch.isnan(result['loss'])
    print("  OK")


def test_decode():
    print("Test decoding...")
    # Fake log probs: [C, T]
    C, T = NUM_CLASSES, 50
    lp = np.full((C, T), -10.0, dtype=np.float64)
    lp[0, :] = -0.1  # blank dominates

    # Place 'a' at frames 10-12, 'b' at frames 30-32
    lp[1, 10:13] = -0.01  # 'a'
    lp[0, 10:13] = -5.0
    lp[2, 30:33] = -0.01  # 'b'
    lp[0, 30:33] = -5.0

    greedy = greedy_decode(lp)
    print(f"  greedy: '{greedy}'")
    assert 'a' in greedy
    assert 'b' in greedy

    beam = prefix_beam_search(lp, beam_width=5)
    print(f"  beam top: '{beam[0]['candidate']}' (score={beam[0]['score']:.2f})")
    print("  OK")


def test_train_loop(device):
    print(f"Test mini train loop ({device})...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake dataset
        for split in ('train', 'val'):
            d = os.path.join(tmpdir, 'data', split)
            os.makedirs(d)
            for j in range(4 if split == 'train' else 2):
                ep = make_fake_episode(T=150, num_keys=6)
                np.savez_compressed(
                    os.path.join(d, f"episode_{j:05d}.npz"),
                    **ep,
                )

        data_dir = os.path.join(tmpdir, 'data')
        out_dir = os.path.join(tmpdir, 'run')

        from configs.config import ModelConfig, TrainConfig, SignalConfig
        from trainers.trainer import CTCTrainer

        scfg = SignalConfig(sample_rate=100)
        mcfg = ModelConfig(input_channels=8, hidden_channels=32,
                           num_layers=4, kernel_size=3, dropout=0.1)
        tcfg = TrainConfig(lr=1e-3, batch_size=2, num_epochs=3, patience=5)

        trainer = CTCTrainer(mcfg, tcfg, scfg, data_dir, out_dir,
                             device=str(device))
        ckpt = trainer.train()
        assert os.path.exists(ckpt)
        print(f"  checkpoint: {ckpt}")
    print("  OK")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}\n")

    test_vocab()
    test_frame_targets()
    test_model_forward(device)
    test_loss(device)
    test_decode()
    test_train_loop(device)

    print(f"\n{'='*40}")
    print("  ALL SANITY TESTS PASSED")
    print(f"{'='*40}")


if __name__ == '__main__':
    main()
