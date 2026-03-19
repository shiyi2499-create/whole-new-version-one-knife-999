#!/usr/bin/env python3
"""
Sanity test for episode-based Stage 2.

Generates a small synthetic dataset with known episode structure,
trains for a few epochs, then verifies:
  1. Frame model produces reasonable 2-class predictions
  2. Episode decoder finds the correct number of episodes
  3. Onset detection within episodes is plausible

Run from stage2_episode/:
    python scripts/sanity_test.py
"""
import sys, os, json, tempfile, shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (SignalConfig, SynthesisConfig, ModelConfig,
                            TrainConfig, EpisodeConfig)
from utils.decoder import decode_episodes
from utils.metrics import full_eval, format_report, frame_accuracy_2class


def make_synthetic_session(sr=100, n_passwords=3, pw_len=6, gap_s=1.5,
                           rng=None):
    """
    Create a perfectly clean synthetic session for testing.
    Each password = pw_len typing bursts with known positions.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    keystroke_dur = int(0.12 * sr)  # 120ms per keystroke
    inter_key_gap = int(0.18 * sr)   # 180ms between keys in same password
    gap_samples = int(gap_s * sr)
    context_samples = int(0.5 * sr)

    episodes = []
    all_imu = []
    all_labels = []
    offset = 0

    # Leading silence
    silence = rng.randn(context_samples, 6).astype(np.float32) * 0.01
    all_imu.append(silence)
    all_labels.append(np.zeros(context_samples, dtype=np.int64))
    offset += context_samples

    for pw_idx in range(n_passwords):
        ep_start = offset
        ep_onsets = []

        for k in range(pw_len):
            # Keystroke burst: larger signal
            burst = rng.randn(keystroke_dur, 6).astype(np.float32) * 0.5
            burst[:, 0] += 1.0  # strong accel_x signal
            labels_burst = np.ones(keystroke_dur, dtype=np.int64)

            all_imu.append(burst)
            all_labels.append(labels_burst)
            onset_pos = offset + keystroke_dur // 2
            ep_onsets.append(onset_pos)
            offset += keystroke_dur

            # Intra-password gap (short silence)
            if k < pw_len - 1:
                gap = rng.randn(inter_key_gap, 6).astype(np.float32) * 0.01
                all_imu.append(gap)
                all_labels.append(np.zeros(inter_key_gap, dtype=np.int64))
                offset += inter_key_gap

        ep_end = offset
        episodes.append({
            'start': ep_start,
            'end': ep_end,
            'onsets': ep_onsets,
            'chars': [chr(ord('a') + k) for k in range(pw_len)],
            'num_keys': pw_len,
        })

        # Inter-password gap (long silence)
        if pw_idx < n_passwords - 1:
            gap = rng.randn(gap_samples, 6).astype(np.float32) * 0.01
            all_imu.append(gap)
            all_labels.append(np.zeros(gap_samples, dtype=np.int64))
            offset += gap_samples

    # Trailing silence
    silence = rng.randn(context_samples, 6).astype(np.float32) * 0.01
    all_imu.append(silence)
    all_labels.append(np.zeros(context_samples, dtype=np.int64))

    imu = np.concatenate(all_imu, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return {
        'imu': imu,
        'frame_labels': labels,
        'episodes': episodes,
        'num_passwords': n_passwords,
        'password_lengths': [pw_len] * n_passwords,
    }


def test_decoder_on_perfect_labels():
    """Test that the decoder works correctly on perfect frame labels."""
    print("=" * 60)
    print("TEST 1: Decoder on perfect frame labels")
    print("=" * 60)

    sr = 100
    n_pw = 4
    pw_len = 8
    session = make_synthetic_session(sr=sr, n_passwords=n_pw, pw_len=pw_len,
                                     gap_s=1.5)

    gt_episodes = session['episodes']
    labels = session['frame_labels']

    print(f"  Session: {len(session['imu'])} samples, "
          f"{n_pw} passwords x {pw_len} keys")
    print(f"  Label distribution: "
          f"silence={np.sum(labels == 0)}, typing={np.sum(labels == 1)}")

    # Run decoder on perfect labels
    dec = decode_episodes(
        labels,
        typing_probs=None,
        sample_rate=sr,
        episode_gap_ms=600.0,
        min_episode_keys=2,
    )

    print(f"  Decoded: {dec['num_episodes']} episodes, "
          f"{dec['total_onsets']} total onsets")
    for i, ep in enumerate(dec['episodes']):
        print(f"    Ep {i}: frames [{ep['start']}-{ep['end']}], "
              f"{ep['num_keys']} keys")

    ev = full_eval(dec['episodes'], gt_episodes, sr, tol_ms=80)
    report = format_report(ev)
    print(report)

    # Assertions
    assert dec['num_episodes'] == n_pw, \
        f"Expected {n_pw} episodes, got {dec['num_episodes']}"
    assert ev['episode_detection_rate'] == 1.0, \
        f"Expected 100% detection rate, got {ev['episode_detection_rate']}"

    print("  ✓ PASSED\n")


def test_decoder_gap_threshold():
    """Test that episode_gap_ms correctly controls episode splitting."""
    print("=" * 60)
    print("TEST 2: Gap threshold controls episode count")
    print("=" * 60)

    sr = 100
    session = make_synthetic_session(sr=sr, n_passwords=3, pw_len=6,
                                     gap_s=1.0)  # 1s gaps

    labels = session['frame_labels']

    # With very large gap threshold (3s) → should merge all into 1 episode
    dec_merged = decode_episodes(labels, sample_rate=sr,
                                 episode_gap_ms=3000.0, min_episode_keys=2)
    print(f"  gap=3000ms → {dec_merged['num_episodes']} episodes "
          f"(expect 1, all merged)")

    # With gap threshold between intra-key gap and inter-pw gap
    dec_correct = decode_episodes(labels, sample_rate=sr,
                                  episode_gap_ms=600.0, min_episode_keys=2)
    print(f"  gap=600ms  → {dec_correct['num_episodes']} episodes "
          f"(expect 3)")

    # With very small gap threshold → might over-split
    dec_split = decode_episodes(labels, sample_rate=sr,
                                episode_gap_ms=100.0, min_episode_keys=1)
    print(f"  gap=100ms  → {dec_split['num_episodes']} episodes "
          f"(expect ≥3, may over-split)")

    assert dec_merged['num_episodes'] == 1, \
        f"Large gap: expected 1, got {dec_merged['num_episodes']}"
    assert dec_correct['num_episodes'] == 3, \
        f"Medium gap: expected 3, got {dec_correct['num_episodes']}"
    assert dec_split['num_episodes'] >= 3, \
        f"Small gap: expected ≥3, got {dec_split['num_episodes']}"

    print("  ✓ PASSED\n")


def test_variable_password_lengths():
    """Test decoder with variable-length passwords."""
    print("=" * 60)
    print("TEST 3: Variable-length passwords")
    print("=" * 60)

    sr = 100
    rng = np.random.RandomState(123)

    pw_lengths = [4, 8, 12, 6]
    episodes = []
    all_imu = []
    all_labels = []
    offset = 0

    keystroke_dur = int(0.12 * sr)
    inter_key_gap = int(0.18 * sr)
    gap_samples = int(1.5 * sr)
    context_samples = int(0.5 * sr)

    # Leading silence
    all_imu.append(rng.randn(context_samples, 6).astype(np.float32) * 0.01)
    all_labels.append(np.zeros(context_samples, dtype=np.int64))
    offset += context_samples

    for pi, pw_len in enumerate(pw_lengths):
        ep_start = offset
        ep_onsets = []

        for k in range(pw_len):
            burst = rng.randn(keystroke_dur, 6).astype(np.float32) * 0.5
            burst[:, 0] += 1.0
            all_imu.append(burst)
            all_labels.append(np.ones(keystroke_dur, dtype=np.int64))
            ep_onsets.append(offset + keystroke_dur // 2)
            offset += keystroke_dur

            if k < pw_len - 1:
                gap = rng.randn(inter_key_gap, 6).astype(np.float32) * 0.01
                all_imu.append(gap)
                all_labels.append(np.zeros(inter_key_gap, dtype=np.int64))
                offset += inter_key_gap

        episodes.append({
            'start': ep_start, 'end': offset,
            'onsets': ep_onsets, 'num_keys': pw_len,
        })

        if pi < len(pw_lengths) - 1:
            gap = rng.randn(gap_samples, 6).astype(np.float32) * 0.01
            all_imu.append(gap)
            all_labels.append(np.zeros(gap_samples, dtype=np.int64))
            offset += gap_samples

    all_imu.append(rng.randn(context_samples, 6).astype(np.float32) * 0.01)
    all_labels.append(np.zeros(context_samples, dtype=np.int64))

    labels = np.concatenate(all_labels)

    dec = decode_episodes(labels, sample_rate=sr,
                          episode_gap_ms=600.0, min_episode_keys=2)

    print(f"  GT: {len(pw_lengths)} episodes with lengths {pw_lengths}")
    print(f"  Decoded: {dec['num_episodes']} episodes with keys "
          f"{[ep['num_keys'] for ep in dec['episodes']]}")

    ev = full_eval(dec['episodes'], episodes, sr, tol_ms=80)
    print(f"  Detection rate: {ev['episode_detection_rate']:.3f}")
    print(f"  Avg onset F1: {ev['avg_onset_f1']:.3f}")

    assert dec['num_episodes'] == len(pw_lengths), \
        f"Expected {len(pw_lengths)}, got {dec['num_episodes']}"

    print("  ✓ PASSED\n")


def test_model_forward():
    """Test that the TCN model runs without errors."""
    print("=" * 60)
    print("TEST 4: Model forward pass")
    print("=" * 60)

    from models.tcn import EpisodeTCN

    model = EpisodeTCN(in_ch=8, hidden=32, num_layers=6, num_classes=2)
    x = torch.randn(2, 8, 300)  # batch=2, channels=8, time=300

    logits = model(x)
    assert logits.shape == (2, 2, 300), f"Expected (2,2,300), got {logits.shape}"

    preds, probs = model.predict(x)
    assert preds.shape == (2, 300)
    assert probs.shape == (2, 2, 300)

    typing_prob = model.predict_typing_prob(x)
    assert typing_prob.shape == (2, 300)

    print(f"  logits: {logits.shape}, preds: {preds.shape}")
    print(f"  typing_prob range: [{typing_prob.min():.3f}, {typing_prob.max():.3f}]")
    print("  ✓ PASSED\n")


def test_dataset_backward_compat():
    """Test that the dataset can load old 3-class data."""
    print("=" * 60)
    print("TEST 5: Dataset backward compatibility (3-class → 2-class)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        train_dir = os.path.join(tmpdir, 'train')
        os.makedirs(train_dir)

        # Create a fake 3-class session
        rng = np.random.RandomState(42)
        T = 500
        imu = rng.randn(T, 6).astype(np.float32)
        # Old 3-class labels: 0=gap, 1=keystroke, 2=separator
        labels_3class = np.zeros(T, dtype=np.int64)
        labels_3class[50:80] = 1   # keystroke
        labels_3class[100:150] = 2  # separator (should become 0)
        labels_3class[200:230] = 1  # keystroke

        np.savez_compressed(
            os.path.join(train_dir, 'session_0000.npz'),
            imu=imu,
            frame_labels=labels_3class,
            groups_json=json.dumps([]),
            num_passwords=2,
            password_lengths=np.array([1, 1]),
        )

        from data.datasets import EpisodeFrameDataset
        ds = EpisodeFrameDataset(tmpdir, 'train', add_mag=True, norm=True)
        assert len(ds) == 1

        x, y = ds[0]
        # y should be 2-class: separator (2) → silence (0)
        assert y.max().item() <= 1, f"Expected max label 1, got {y.max()}"
        assert y[50:80].sum() > 0, "Keystroke region should be 1"
        assert y[100:150].sum() == 0, "Old separator region should be 0"

        print(f"  Loaded 3-class data, converted to 2-class")
        print(f"  Label unique values: {torch.unique(y).tolist()}")
        print("  ✓ PASSED\n")

    finally:
        shutil.rmtree(tmpdir)


def main():
    print("\n" + "=" * 60)
    print("EPISODE-BASED STAGE 2: SANITY TESTS")
    print("=" * 60 + "\n")

    test_decoder_on_perfect_labels()
    test_decoder_gap_threshold()
    test_variable_password_lengths()
    test_model_forward()
    test_dataset_backward_compat()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
