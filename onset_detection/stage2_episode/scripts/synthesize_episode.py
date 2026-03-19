#!/usr/bin/env python3
"""
Generate synthetic 2-class training data for episode-based Stage 2.

Usage:
    python scripts/synthesize_episode.py \
        --password_dir data/raw/password \
        --neg_dir data/raw/onset_negative \
        --output_dir data/stage2_episode_synth \
        --num_sessions 400
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import NegativeLoader
from data.synthesis import EpisodeSynthesizer, load_all_segments
from configs.config import SynthesisConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--password_dir', required=True,
                    help='Directory with password recording sessions')
    ap.add_argument('--neg_dir', required=True,
                    help='Directory with negative clips')
    ap.add_argument('--output_dir', required=True,
                    help='Output directory for synthesized dataset')
    ap.add_argument('--num_sessions', type=int, default=400)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    segs = load_all_segments(args.password_dir)
    if not segs:
        print("ERROR: no password segments found")
        raise SystemExit(1)

    neg = NegativeLoader(args.neg_dir)
    cfg = SynthesisConfig(num_sessions=args.num_sessions)

    synth = EpisodeSynthesizer(segs, neg, cfg, seed=args.seed)
    synth.generate_dataset(args.num_sessions, args.output_dir)


if __name__ == '__main__':
    main()
