#!/usr/bin/env python3
"""
Script: Quick sanity test with dummy data.
Verifies the entire pipeline compiles, runs, and produces expected output shapes.

Usage:
    python scripts/sanity_test.py
"""
import sys
import os
import tempfile
import shutil
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    PipelineConfig, Stage2AConfig, Stage2BConfig,
    SignalConfig, SynthesisConfig
)
from models.stage2a import GroupSegmentor
from models.stage2b import OnsetDetector
from models.losses import Stage2ALoss, Stage2BLoss
from models.tcn import SingleStageTCN, DilatedResidualLayer
from utils.signal_processing import (
    preprocess_imu, compute_magnitude, splice_smooth, time_stretch
)
from utils.metrics import compute_group_iou, compute_onset_metrics, compute_e2e_metrics
from data.datasets import Stage2ADataset, Stage2BDataset


def test_signal_processing():
    print("Testing signal processing...")
    data = np.random.randn(500, 6).astype(np.float32)

    # Magnitude
    mag = compute_magnitude(data)
    assert mag.shape == (500, 8), f"Expected (500,8), got {mag.shape}"

    # Preprocess
    processed, stats = preprocess_imu(data, sample_rate=100)
    assert processed.shape[0] == 500
    assert processed.shape[1] == 8  # 6 raw + 2 magnitude

    # Splice
    a = np.random.randn(100, 6).astype(np.float32)
    b = np.random.randn(100, 6).astype(np.float32)
    spliced = splice_smooth(a, b, overlap_samples=10)
    assert spliced.shape == (190, 6), f"Expected (190,6), got {spliced.shape}"

    # Time stretch
    stretched = time_stretch(data, rate=1.2)
    assert stretched.shape[1] == 6
    assert stretched.shape[0] < 500  # faster = shorter

    print("  ✓ Signal processing OK")


def test_tcn_model():
    print("Testing TCN models...")

    # Single stage TCN
    model = SingleStageTCN(
        input_channels=8, hidden_channels=32,
        output_channels=1, num_layers=6
    )
    x = torch.randn(2, 8, 300)  # [B, C, T]
    y = model(x)
    assert y.shape == (2, 1, 300), f"Expected (2,1,300), got {y.shape}"

    print("  ✓ TCN model OK")


def test_stage2a():
    print("Testing Stage 2A (Group Segmentor)...")

    config = Stage2AConfig(input_channels=8, hidden_channels=32, num_layers=6)
    model = GroupSegmentor(config)

    x = torch.randn(2, 8, 500)
    logits = model(x)
    assert logits.shape == (2, 1, 500)

    # Test post-processing
    probs = np.random.rand(500)
    # Make 5 clear peaks
    for i in range(5):
        center = 50 + i * 90
        probs[center:center + 30] = 0.8 + np.random.rand(30) * 0.2
    probs = np.clip(probs, 0, 1)

    groups = GroupSegmentor.post_process(
        probs, sample_rate=100, expected_groups=5,
        min_group_duration_s=0.1
    )
    print(f"  Found {len(groups)} groups (expected ~5)")

    # Test loss
    loss_fn = Stage2ALoss()
    targets = torch.rand(2, 500)
    mask = torch.ones(2, 500)
    loss = loss_fn(logits, targets, mask)
    assert loss.item() > 0

    print("  ✓ Stage 2A OK")


def test_stage2b():
    print("Testing Stage 2B (Onset Detector)...")

    config = Stage2BConfig(input_channels=8, hidden_channels=32, num_layers=4)
    model = OnsetDetector(config)

    x = torch.randn(4, 8, 200)
    logits = model(x)
    assert logits.shape == (4, 1, 200)

    # Test peak picking
    probs = np.zeros(200)
    # Place 8 clear peaks
    gt_positions = [15, 35, 55, 75, 95, 120, 145, 170]
    for p in gt_positions:
        t = np.arange(200)
        probs += 0.9 * np.exp(-((t - p) ** 2) / (2 * 3 ** 2))

    peaks = OnsetDetector.pick_peaks(probs, expected_onsets=8, min_iki_samples=5)
    print(f"  Found {len(peaks)} peaks (expected 8)")
    print(f"  GT:   {gt_positions}")
    print(f"  Pred: {peaks.tolist()}")

    # Test loss
    loss_fn = Stage2BLoss(use_focal=True)
    targets = torch.rand(4, 200)
    mask = torch.ones(4, 200)
    loss = loss_fn(logits, targets, mask)
    assert loss.item() > 0

    print("  ✓ Stage 2B OK")


def test_metrics():
    print("Testing metrics...")

    # Group IoU
    pred_groups = [(10, 50), (60, 100), (110, 150), (160, 200), (210, 250)]
    gt_groups = [(12, 48), (62, 98), (112, 148), (162, 198), (212, 248)]
    metrics = compute_group_iou(pred_groups, gt_groups)
    print(f"  Group IoU: {metrics['mean_iou']:.4f}")
    assert metrics['mean_iou'] > 0.8

    # Onset metrics
    pred_onsets = np.array([10, 30, 50, 70, 90, 110, 130, 150])
    gt_onsets = np.array([12, 31, 49, 71, 88, 112, 129, 151])
    metrics = compute_onset_metrics(pred_onsets, gt_onsets, tolerance_samples=5)
    print(f"  Onset F1: {metrics['f1']:.4f}, MAE: {metrics['mean_abs_error_ms']:.1f}ms")
    assert metrics['f1'] > 0.5

    # E2E metrics
    pred_chars = [['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
    gt_chars = [['a', 'b', 'x', 'd', 'e', 'f', 'g', 'h']]
    metrics = compute_e2e_metrics(pred_chars, gt_chars)
    print(f"  char_top1: {metrics['char_top1']:.4f}, CER: {metrics['CER']:.4f}")
    assert metrics['char_top1'] == 7 / 8

    print("  ✓ Metrics OK")


def test_synthetic_data():
    print("Testing synthetic data generation (in-memory)...")

    # Create fake password segments
    fake_segments = []
    for i in range(20):
        T = np.random.randint(150, 400)
        imu = np.random.randn(T, 6).astype(np.float32)
        onsets = sorted(np.random.choice(range(10, T - 10), size=8, replace=False).tolist())
        chars = [chr(ord('a') + np.random.randint(0, 26)) for _ in range(8)]
        fake_segments.append({
            'imu': imu,
            'timestamps': np.arange(T) * 10_000_000,  # 100Hz in ns
            'key_onsets': onsets,
            'key_chars': chars,
            'prompt': 'test1234',
            'duration_s': T / 100.0,
        })

    # Create a fake negative loader
    from data.loaders import NegativeDataLoader

    class FakeNegLoader:
        @property
        def clips(self):
            return [np.random.randn(500, 6).astype(np.float32)]

        def sample_clip(self, duration_samples, rng=None):
            if rng is None:
                rng = np.random.RandomState()
            return rng.randn(duration_samples, 6).astype(np.float32) * 0.1

    from data.synthesis import SyntheticMixedGenerator

    config = SynthesisConfig(
        num_sessions=5,
        passwords_per_session=5,
        keys_per_password=8,
    )

    generator = SyntheticMixedGenerator(
        password_segments=fake_segments,
        negative_loader=FakeNegLoader(),
        config=config,
        sample_rate=100,
        seed=42,
    )

    session = generator.generate_one()
    print(f"  Session IMU shape: {session['imu'].shape}")
    print(f"  Group labels shape: {session['group_labels'].shape}")
    print(f"  Groups: {len(session['group_boundaries'])}")
    print(f"  Onsets per group: {[len(o) for o in session['onset_positions']]}")
    assert session['imu'].shape[1] == 6
    assert len(session['group_boundaries']) == 5

    # Save and load as dataset
    tmp_dir = tempfile.mkdtemp()
    try:
        generator.generate_dataset(num_sessions=10, output_dir=tmp_dir)

        # Test Stage2A dataset
        ds_2a = Stage2ADataset(tmp_dir, split='train', sample_rate=100)
        if len(ds_2a) > 0:
            x, y = ds_2a[0]
            print(f"  Stage2A sample: x={x.shape}, y={y.shape}")

        # Test Stage2B dataset
        ds_2b = Stage2BDataset(tmp_dir, split='train', sample_rate=100)
        if len(ds_2b) > 0:
            x, y, onsets = ds_2b[0]
            print(f"  Stage2B sample: x={x.shape}, y={y.shape}, onsets={onsets}")
    finally:
        shutil.rmtree(tmp_dir)

    print("  ✓ Synthetic data OK")


def test_full_pipeline_forward():
    print("Testing full pipeline forward pass...")

    config = PipelineConfig()
    config.signal.sample_rate = 100
    config.stage2a.input_channels = 8
    config.stage2a.hidden_channels = 32
    config.stage2a.num_layers = 4
    config.stage2b.input_channels = 8
    config.stage2b.hidden_channels = 32
    config.stage2b.num_layers = 4

    stage2a = GroupSegmentor(config.stage2a)
    stage2b = OnsetDetector(config.stage2b)

    from models.pipeline import Stage2Pipeline
    pipeline = Stage2Pipeline(stage2a, stage2b, config)

    # Fake coarse region: ~10s at 100Hz
    fake_imu = np.random.randn(1000, 6).astype(np.float32)
    results = pipeline.run(fake_imu, sample_rate=100)

    print(f"  Groups found: {results['num_groups']}")
    print(f"  Group boundaries: {results['group_boundaries']}")
    for g, onsets in enumerate(results['onset_positions']):
        print(f"  Group {g}: {len(onsets)} onsets")

    print("  ✓ Full pipeline OK")


def main():
    print("=" * 60)
    print("SANITY TEST: Stage 2 Rebuild")
    print("=" * 60)
    print()

    test_signal_processing()
    test_tcn_model()
    test_stage2a()
    test_stage2b()
    test_metrics()
    test_synthetic_data()
    test_full_pipeline_forward()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
