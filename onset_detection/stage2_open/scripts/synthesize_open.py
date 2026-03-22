#!/usr/bin/env python3
"""
Generate variable-length synthetic sessions for Open Stage 2.

python scripts/synthesize_open.py \
    --password_dir data/password/len_8 \
    --negative_dir data/onset_negative \
    --output_dir data/synthetic_open \
    --num_sessions 300
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import NegativeLoader
from data.synthesis import OpenSynthesizer, load_all_segments
from configs.config import SynthesisConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--password_dir', required=True)
    ap.add_argument('--negative_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--num_sessions', type=int, default=300)
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_passwords', type=int, default=2)
    ap.add_argument('--max_passwords', type=int, default=8)
    ap.add_argument('--min_pw_len', type=int, default=4)
    ap.add_argument('--max_pw_len', type=int, default=12)
    args = ap.parse_args()

    print("=" * 60)
    print("Open Synthesizer: Variable-Length Sessions")
    print("=" * 60)

    segs = load_all_segments(args.password_dir)
    if not segs:
        print("ERROR: no password segments found"); sys.exit(1)

    neg = NegativeLoader(args.negative_dir)
    print(f"Negative clips: {len(neg.clips)}")

    cfg = SynthesisConfig(
        num_sessions=args.num_sessions,
        min_passwords=args.min_passwords,
        max_passwords=args.max_passwords,
        min_password_len=args.min_pw_len,
        max_password_len=args.max_pw_len,
    )

    gen = OpenSynthesizer(segs, neg, cfg, args.sample_rate, args.seed)
    gen.generate_dataset(args.num_sessions, args.output_dir)
    print("Done!")


if __name__ == '__main__':
    main()
