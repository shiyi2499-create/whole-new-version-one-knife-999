#!/usr/bin/env python3
"""Sanity test for stage2_open (no torch required)."""
import sys, os, tempfile, shutil, json
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.signal_processing import preprocess, compute_magnitude, splice_smooth, time_stretch
from utils.decoder import decode_frame_labels
from utils.metrics import match_groups, onset_metrics, frame_accuracy, full_eval, format_report
from configs.config import SynthesisConfig


def test_signal():
    print("1. Signal processing...")
    d = np.random.randn(400, 6).astype(np.float32)
    m = compute_magnitude(d)
    assert m.shape == (400, 8)
    p, s = preprocess(d, 100)
    assert p.shape == (400, 8)
    a, b = np.random.randn(100, 6).astype(np.float32), np.random.randn(80, 6).astype(np.float32)
    sp = splice_smooth(a, b, 10)
    assert sp.shape == (170, 6)
    ts = time_stretch(d, 1.5)
    assert ts.shape[1] == 6 and ts.shape[0] < 400
    print("   ✓")


def test_decoder():
    print("2. Decoder (frame labels → groups + onsets)...")

    T = 800
    labels = np.zeros(T, dtype=np.int64)

    # Build 3 password groups of varying length
    # Group 1: 5 keystrokes
    g1_start, g1_end = 50, 200
    g1_onsets = [60, 85, 110, 140, 175]
    for o in g1_onsets:
        labels[max(0, o - 3):min(T, o + 4)] = 1

    # Separator
    labels[200:260] = 2

    # Group 2: 8 keystrokes
    g2_start, g2_end = 260, 520
    g2_onsets = [280, 300, 325, 350, 380, 410, 440, 480]
    for o in g2_onsets:
        labels[max(0, o - 3):min(T, o + 4)] = 1

    # Separator
    labels[520:580] = 2

    # Group 3: 3 keystrokes
    g3_start, g3_end = 580, 700
    g3_onsets = [600, 640, 680]
    for o in g3_onsets:
        labels[max(0, o - 3):min(T, o + 4)] = 1

    result = decode_frame_labels(labels, sample_rate=100, median_kernel=3,
                                  min_keystroke_run=2, min_separator_run_ms=100)

    print(f"   Found {result['num_passwords']} groups (expected 3)")
    for g in result['groups']:
        print(f"   Group [{g['start']},{g['end']}]: {g['num_keys']} keys")
    print(f"   Total onsets: {result['total_onsets']} (expected 16)")

    assert result['num_passwords'] == 3, f"Expected 3 groups, got {result['num_passwords']}"
    assert result['total_onsets'] == 16, f"Expected 16 onsets, got {result['total_onsets']}"

    # Verify onset positions are close to GT
    all_pred_onsets = []
    for g in result['groups']:
        all_pred_onsets.extend(g['onsets'])
    all_gt_onsets = g1_onsets + g2_onsets + g3_onsets

    errors = []
    for gt in all_gt_onsets:
        dists = [abs(p - gt) for p in all_pred_onsets]
        errors.append(min(dists))
    print(f"   Mean onset error: {np.mean(errors):.1f} samples")
    assert np.mean(errors) < 5

    print("   ✓")


def test_decoder_edge_cases():
    print("3. Decoder edge cases...")

    # No separators → single group
    labels = np.zeros(200, dtype=np.int64)
    labels[20:25] = 1; labels[50:55] = 1; labels[80:85] = 1
    r = decode_frame_labels(labels, 100, median_kernel=3, min_keystroke_run=2)
    assert r['num_passwords'] == 1
    assert r['total_onsets'] == 3
    print(f"   No separators: {r['num_passwords']} group, {r['total_onsets']} onsets ✓")

    # All gap → no groups
    labels = np.zeros(200, dtype=np.int64)
    r = decode_frame_labels(labels, 100, median_kernel=3)
    assert r['num_passwords'] == 0
    print(f"   All gap: {r['num_passwords']} groups ✓")

    # Single keystroke → 1 group, 1 onset
    labels = np.zeros(100, dtype=np.int64)
    labels[40:46] = 1
    r = decode_frame_labels(labels, 100, median_kernel=3, min_keystroke_run=2)
    assert r['num_passwords'] == 1
    assert r['total_onsets'] == 1
    print(f"   Single key: {r['num_passwords']} group, {r['total_onsets']} onset ✓")

    print("   ✓")


def test_metrics():
    print("4. Metrics (variable-length)...")

    pred_groups = [
        {'start': 10, 'end': 100, 'onsets': [20, 40, 60, 80], 'num_keys': 4},
        {'start': 150, 'end': 350, 'onsets': [160, 180, 200, 220, 240, 260, 280, 300],
         'num_keys': 8},
    ]
    gt_groups = [
        {'start': 12, 'end': 98, 'onsets': [22, 42, 62, 82], 'num_keys': 4},
        {'start': 148, 'end': 348, 'onsets': [162, 182, 202, 222, 242, 262, 282, 302],
         'num_keys': 8},
        {'start': 400, 'end': 500, 'onsets': [420, 450, 480], 'num_keys': 3},  # unmatched GT
    ]

    ev = full_eval(pred_groups, gt_groups, sr=100, tol_ms=50)
    print(f"   Groups matched: {len(ev['group_match']['matches'])}/2 pred, "
          f"unmatched GT: {ev['group_match']['unmatched_gt']}")
    print(f"   Avg onset F1: {ev['avg_onset_f1']:.3f}")
    print(f"   Avg group IoU: {ev['avg_group_iou']:.3f}")

    assert len(ev['group_match']['matches']) == 2
    assert ev['avg_onset_f1'] > 0.8
    assert len(ev['group_match']['unmatched_gt']) == 1  # 3rd GT group unmatched

    report = format_report(ev)
    print(report)
    print("   ✓")


def test_synthesis():
    print("5. Variable-length synthesis...")
    from data.synthesis import OpenSynthesizer
    from data.loaders import NegativeLoader

    # Fake data
    segs = []
    for nk in [4, 5, 6, 7, 8, 10, 12]:
        for _ in range(5):
            T = nk * 25 + np.random.randint(20, 60)
            imu = np.random.randn(T, 6).astype(np.float32)
            onsets = sorted(np.random.choice(range(10, T - 10), size=nk, replace=False).tolist())
            chars = [chr(ord('a') + np.random.randint(0, 26)) for _ in range(nk)]
            segs.append({'imu': imu, 'onsets': onsets, 'chars': chars, 'num_keys': nk,
                          'prompt': 'x' * nk})

    class FakeNeg:
        clips = [np.random.randn(300, 6).astype(np.float32)]
        def sample(self, n, rng):
            return rng.randn(n, 6).astype(np.float32) * 0.05

    cfg = SynthesisConfig(num_sessions=20, min_passwords=2, max_passwords=6,
                           min_password_len=4, max_password_len=10)
    gen = OpenSynthesizer(segs, FakeNeg(), cfg, 100, 42)

    s = gen.generate_one()
    print(f"   IMU: {s['imu'].shape}, labels: {s['frame_labels'].shape}")
    print(f"   N passwords: {s['num_passwords']}, lengths: {s['password_lengths']}")
    print(f"   Groups: {len(s['groups'])}")
    for g in s['groups']:
        print(f"     [{g['start']},{g['end']}] {len(g['onsets'])} keys")

    assert s['num_passwords'] >= 2
    assert len(s['groups']) == s['num_passwords']

    # Check label distribution
    unique, counts = np.unique(s['frame_labels'], return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"   Label dist: gap={dist.get(0, 0)} ks={dist.get(1, 0)} sep={dist.get(2, 0)}")
    assert 1 in dist  # should have keystroke frames
    if s['num_passwords'] > 1:
        assert 2 in dist  # should have separator frames

    # Test full dataset generation
    tmp = tempfile.mkdtemp()
    try:
        gen.generate_dataset(20, tmp)
        for split in ['train', 'val', 'test']:
            files = list((Path(tmp) / split).glob('*.npz'))
            print(f"   {split}: {len(files)} files")
            assert len(files) > 0

        # Load one and verify
        f0 = sorted((Path(tmp) / 'train').glob('*.npz'))[0]
        d = np.load(f0, allow_pickle=True)
        assert 'imu' in d and 'frame_labels' in d and 'groups_json' in d
        groups = json.loads(str(d['groups_json']))
        print(f"   Sample: {d['imu'].shape}, {len(groups)} groups, "
              f"pw_lengths={d['password_lengths'].tolist()}")
    finally:
        shutil.rmtree(tmp)

    print("   ✓")


def main():
    print("=" * 60)
    print("SANITY TEST: stage2_open (no torch)")
    print("=" * 60)
    print()
    test_signal()
    print()
    test_decoder()
    print()
    test_decoder_edge_cases()
    print()
    test_metrics()
    print()
    test_synthesis()
    print()
    print("=" * 60)
    print("ALL 5 TESTS PASSED ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
