#!/usr/bin/env python3
"""
Sanity test (no PyTorch required).
Tests signal processing, metrics, data synthesis, and peak picking logic.
"""
import sys
import os
import tempfile
import shutil
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.signal_processing import (
    preprocess_imu, compute_magnitude, splice_smooth,
    time_stretch, compute_energy_envelope
)
from utils.metrics import compute_group_iou, compute_onset_metrics, compute_e2e_metrics
from configs.config import SynthesisConfig


def test_signal_processing():
    print("1. Testing signal processing...")
    data = np.random.randn(500, 6).astype(np.float32)

    mag = compute_magnitude(data)
    assert mag.shape == (500, 8), f"Expected (500,8), got {mag.shape}"

    processed, stats = preprocess_imu(data, sample_rate=100)
    assert processed.shape == (500, 8)
    assert 'mean' in stats and 'std' in stats

    a = np.random.randn(100, 6).astype(np.float32)
    b = np.random.randn(100, 6).astype(np.float32)
    spliced = splice_smooth(a, b, overlap_samples=10)
    assert spliced.shape == (190, 6), f"Got {spliced.shape}"

    stretched = time_stretch(data, rate=1.2)
    assert stretched.shape[1] == 6
    assert stretched.shape[0] < 500

    energy = compute_energy_envelope(data, window_size=50)
    assert energy.shape == (500,)

    print("   ✓ All signal processing tests passed")


def test_metrics():
    print("2. Testing evaluation metrics...")

    # Group IoU
    pred_groups = [(10, 50), (60, 100), (110, 150), (160, 200), (210, 250)]
    gt_groups = [(12, 48), (62, 98), (112, 148), (162, 198), (212, 248)]
    m = compute_group_iou(pred_groups, gt_groups)
    assert m['mean_iou'] > 0.8, f"IoU too low: {m['mean_iou']}"
    assert m['group_count_correct'] == True
    print(f"   Group IoU: {m['mean_iou']:.4f}, boundary err: {m['mean_boundary_error_samples']:.1f} samples")

    # Onset metrics
    pred_onsets = np.array([10, 30, 50, 70, 90, 110, 130, 150])
    gt_onsets = np.array([12, 31, 49, 71, 88, 112, 129, 151])
    m = compute_onset_metrics(pred_onsets, gt_onsets, tolerance_samples=5, sample_rate=100)
    assert m['f1'] > 0.5, f"F1 too low: {m['f1']}"
    print(f"   Onset F1: {m['f1']:.4f}, Recall: {m['recall']:.4f}, MAE: {m['mean_abs_error_ms']:.1f}ms")

    # E2E
    pred_chars = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
    gt_chars = [['a', 'b', 'x', 'd', 'e', 'f', 'g', 'h']]
    m = compute_e2e_metrics(pred_chars, gt_chars)
    assert m['char_top1'] == 7 / 8
    print(f"   E2E top1: {m['char_top1']:.4f}, CER: {m['CER']:.4f}")

    print("   ✓ All metric tests passed")


def test_peak_picking():
    print("3. Testing peak picking logic...")
    from utils.postprocess import pick_onset_peaks

    # Create synthetic onset probability signal
    T = 300
    probs = np.zeros(T)
    gt_positions = [20, 50, 80, 115, 145, 180, 215, 255]
    sigma = 3.0

    for p in gt_positions:
        t = np.arange(T)
        probs += 0.85 * np.exp(-((t - p) ** 2) / (2 * sigma ** 2))

    # Add noise
    probs += np.random.rand(T) * 0.05
    probs = np.clip(probs, 0, 1)

    peaks = pick_onset_peaks(probs, expected_onsets=8, min_iki_samples=10)
    print(f"   GT:   {gt_positions}")
    print(f"   Pred: {peaks.tolist()}")
    assert len(peaks) == 8, f"Expected 8 peaks, got {len(peaks)}"

    # Check that peaks are close to GT
    errors = []
    for gt in gt_positions:
        dists = np.abs(peaks - gt)
        errors.append(np.min(dists))
    mean_error = np.mean(errors)
    print(f"   Mean peak error: {mean_error:.1f} samples")
    assert mean_error < 5, f"Mean error too high: {mean_error}"

    # Test with fewer clear peaks (robustness)
    probs2 = np.zeros(T)
    for p in gt_positions[:4]:  # only 4 clear peaks
        t = np.arange(T)
        probs2 += 0.8 * np.exp(-((t - p) ** 2) / (2 * sigma ** 2))
    probs2 = np.clip(probs2, 0, 1)

    peaks2 = pick_onset_peaks(probs2, expected_onsets=8, min_iki_samples=10)
    print(f"   Degraded input (4 clear peaks): found {len(peaks2)} peaks")
    assert len(peaks2) == 8, f"Expected 8 peaks even with degraded input"

    print("   ✓ Peak picking tests passed")


def test_group_postprocess():
    print("4. Testing group segmentor post-processing...")
    from utils.postprocess import extract_groups_from_probs

    T = 1000
    probs = np.random.rand(T) * 0.1  # low baseline

    # Create 5 clear typing regions
    regions = [(50, 130), (200, 280), (350, 430), (520, 600), (680, 760)]
    for start, end in regions:
        probs[start:end] = 0.7 + np.random.rand(end - start) * 0.3

    groups = extract_groups_from_probs(
        probs, sample_rate=100, expected_groups=5,
        min_group_duration_s=0.1, median_kernel=5
    )

    print(f"   Found {len(groups)} groups (expected 5)")
    for i, (s, e) in enumerate(groups):
        print(f"   Group {i}: [{s}, {e}] duration={e - s} samples")

    assert len(groups) == 5, f"Expected 5 groups, got {len(groups)}"

    # Check IoU with GT
    m = compute_group_iou(groups, regions)
    print(f"   IoU with GT: {m['mean_iou']:.4f}")
    assert m['mean_iou'] > 0.7

    print("   ✓ Group post-processing tests passed")


def test_synthetic_generation():
    print("5. Testing synthetic data generation...")
    from data.synthesis import SyntheticMixedGenerator, _pad_onset_lists

    # Create fake password segments
    segments = []
    for i in range(30):
        T = np.random.randint(150, 400)
        imu = np.random.randn(T, 6).astype(np.float32)
        onsets = sorted(np.random.choice(range(10, T - 10), size=8, replace=False).tolist())
        chars = [chr(ord('a') + np.random.randint(0, 26)) for _ in range(8)]
        segments.append({
            'imu': imu,
            'timestamps': np.arange(T, dtype=np.int64) * 10_000_000,
            'key_onsets': onsets,
            'key_chars': chars,
            'prompt': f'test{i:04d}',
            'duration_s': T / 100.0,
        })

    class FakeNegLoader:
        @property
        def clips(self):
            return [np.random.randn(500, 6).astype(np.float32)]
        def sample_clip(self, duration_samples, rng=None):
            rng = rng or np.random.RandomState()
            return rng.randn(duration_samples, 6).astype(np.float32) * 0.05

    config = SynthesisConfig(
        num_sessions=10,
        passwords_per_session=5,
        keys_per_password=8,
        gap_duration_min=0.3,
        gap_duration_max=1.5,
    )

    gen = SyntheticMixedGenerator(
        password_segments=segments,
        negative_loader=FakeNegLoader(),
        config=config,
        sample_rate=100,
        seed=42,
    )

    session = gen.generate_one()
    print(f"   IMU shape: {session['imu'].shape}")
    print(f"   Group labels shape: {session['group_labels'].shape}")
    print(f"   Groups: {len(session['group_boundaries'])}")
    print(f"   Onsets per group: {[len(o) for o in session['onset_positions']]}")

    assert session['imu'].shape[1] == 6
    assert len(session['group_boundaries']) == 5
    assert session['group_labels'].sum() > 0  # some frames should be labeled as typing

    # Check that onset positions fall within group boundaries
    for g, (start, end) in enumerate(session['group_boundaries']):
        for onset in session['onset_positions'][g]:
            assert start <= onset < end, \
                f"Onset {onset} outside group [{start}, {end}]"

    # Test full dataset generation
    tmp_dir = tempfile.mkdtemp()
    try:
        gen.generate_dataset(num_sessions=15, output_dir=tmp_dir)

        # Verify files exist
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(tmp_dir, split)
            assert os.path.isdir(split_dir), f"Missing split dir: {split}"
            files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
            print(f"   {split}: {len(files)} files")
            assert len(files) > 0

        # Load and verify a file
        train_files = sorted(os.listdir(os.path.join(tmp_dir, 'train')))
        sample = np.load(os.path.join(tmp_dir, 'train', train_files[0]), allow_pickle=True)
        print(f"   Sample keys: {list(sample.keys())}")
        assert 'imu' in sample
        assert 'group_labels' in sample
        assert 'group_boundaries' in sample
        assert 'onset_positions' in sample

        # Check metadata
        with open(os.path.join(tmp_dir, 'metadata.json')) as f:
            meta = json.load(f)
        print(f"   Metadata: {json.dumps(meta, indent=2)}")

    finally:
        shutil.rmtree(tmp_dir)

    print("   ✓ Synthetic data generation tests passed")


def test_config():
    print("6. Testing configuration...")
    from configs.config import PipelineConfig

    cfg = PipelineConfig()
    assert cfg.signal.sample_rate == 190
    assert cfg.stage2a.expected_groups == 5
    assert cfg.stage2b.expected_onsets == 8
    assert cfg.synthesis.passwords_per_session == 5

    print(f"   Signal: {cfg.signal.num_channels}ch @ {cfg.signal.sample_rate}Hz")
    print(f"   Stage2A: {cfg.stage2a.hidden_channels}h × {cfg.stage2a.num_layers}L")
    print(f"   Stage2B: {cfg.stage2b.hidden_channels}h × {cfg.stage2b.num_layers}L")

    print("   ✓ Configuration tests passed")


def test_full_report():
    print("7. Testing full report generation...")
    from utils.metrics import compute_full_report

    stage2a = {
        'pred_count': 5, 'gt_count': 5, 'group_count_correct': True,
        'mean_iou': 0.85, 'per_group_iou': [0.9, 0.8, 0.85, 0.82, 0.88],
        'mean_boundary_error_samples': 8.5,
    }
    stage2b = [
        {'precision': 0.875, 'recall': 0.875, 'f1': 0.875, 'mean_abs_error_ms': 12.3,
         'n_pred': 8, 'n_gt': 8, 'n_matched': 7},
    ] * 5
    e2e = {
        'char_top1': 0.425, 'char_top3': 0.7, 'char_top5': 0.825,
        'CER': 0.575, 'exact_match_count': 1, 'total_groups': 5, 'total_chars': 40,
    }

    report = compute_full_report(stage2a, stage2b, e2e, sample_rate=100)
    print(report)

    print("   ✓ Report generation tests passed")


def main():
    print("=" * 60)
    print("SANITY TEST: Stage 2 Rebuild (no PyTorch)")
    print("=" * 60)
    print()

    test_signal_processing()
    print()
    test_metrics()
    print()
    test_peak_picking()
    print()
    test_group_postprocess()
    print()
    test_synthetic_generation()
    print()
    test_config()
    print()
    test_full_report()

    print()
    print("=" * 60)
    print("ALL 7 TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("To run with PyTorch (models + training), install torch and run:")
    print("  python scripts/sanity_test.py")


if __name__ == '__main__':
    main()
