#!/usr/bin/env python3
"""
Script: Generate synthetic mixed sessions for Stage 2 training.

Usage:
    python scripts/synthesize_mixed.py \
        --password_dir data/raw/password/len_8 \
        --negative_dir data/raw/onset_negative \
        --output_dir data/processed/stage2_synthetic_mixed \
        --num_sessions 200 \
        --sample_rate 190 \
        --seed 42
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import NegativeDataLoader
from data.synthesis import (
    BlockTemplateGenerator,
    SyntheticMixedGenerator,
    load_all_password_blocks,
    load_all_password_segments,
)
from configs.config import SynthesisConfig


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic mixed sessions")
    parser.add_argument('--password_dir', type=str, required=True,
                        help='Path to password/len_8 data directory')
    parser.add_argument('--negative_dir', type=str, required=True,
                        help='Path to onset_negative data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for synthetic data')
    parser.add_argument('--num_sessions', type=int, default=200,
                        help='Number of synthetic sessions to generate')
    parser.add_argument('--sample_rate', type=int, default=190,
                        help='IMU sample rate in Hz')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--passwords_per_session', type=int, default=5)
    parser.add_argument('--keys_per_password', type=int, default=8)
    parser.add_argument('--template_mode', choices=['blocks', 'segments'], default='blocks',
                        help='blocks = real consecutive 5-password templates (recommended); '
                             'segments = old random attempt splicing')
    parser.add_argument('--gap_min', type=float, default=0.5,
                        help='Min gap between passwords (seconds)')
    parser.add_argument('--gap_max', type=float, default=3.0,
                        help='Max gap between passwords (seconds)')
    args = parser.parse_args()

    print("=" * 60)
    print("Synthetic Mixed Session Generator")
    print("=" * 60)

    valid_segments = None
    valid_blocks = None
    if args.template_mode == 'blocks':
        print(f"\nLoading real password block templates from: {args.password_dir}")
        valid_blocks = load_all_password_blocks(
            args.password_dir,
            target_rate_hz=args.sample_rate,
        )
        print(f"  Block templates: {len(valid_blocks)}")
        if len(valid_blocks) == 0:
            print("ERROR: No password block templates found!")
            sys.exit(1)
    else:
        print(f"\nLoading password segments from: {args.password_dir}")
        password_segments = load_all_password_segments(
            args.password_dir,
            target_rate_hz=args.sample_rate,
        )

        if len(password_segments) == 0:
            print("ERROR: No password segments found!")
            print("Make sure the directory contains password raw files like *_sensor.csv")
            sys.exit(1)

        valid_segments = [s for s in password_segments
                          if len(s['key_onsets']) >= args.keys_per_password]
        print(f"  Valid segments (>= {args.keys_per_password} onsets): {len(valid_segments)}")

        if len(valid_segments) == 0:
            print("WARNING: No segments with enough onsets. Using all segments.")
            valid_segments = password_segments

    # Load negative data
    print(f"\nLoading negative data from: {args.negative_dir}")
    neg_loader = NegativeDataLoader(args.negative_dir, target_rate_hz=args.sample_rate)
    print(f"  Negative clips loaded: {len(neg_loader.clips)}")

    # Configure synthesis
    config = SynthesisConfig(
        num_sessions=args.num_sessions,
        passwords_per_session=args.passwords_per_session,
        keys_per_password=args.keys_per_password,
        gap_duration_min=args.gap_min,
        gap_duration_max=args.gap_max,
    )

    # Generate
    print(f"\nGenerating {args.num_sessions} synthetic sessions...")
    if args.template_mode == 'blocks':
        generator = BlockTemplateGenerator(
            password_blocks=valid_blocks,
            negative_loader=neg_loader,
            config=config,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )
    else:
        generator = SyntheticMixedGenerator(
            password_segments=valid_segments,
            negative_loader=neg_loader,
            config=config,
            sample_rate=args.sample_rate,
            seed=args.seed,
        )

    metadata = generator.generate_dataset(
        num_sessions=args.num_sessions,
        output_dir=args.output_dir,
    )

    print("\nDone!")
    print(f"Output: {args.output_dir}")


if __name__ == '__main__':
    main()
