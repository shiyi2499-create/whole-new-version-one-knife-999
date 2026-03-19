#!/usr/bin/env python3
"""
Train the episode-based Stage 2 model.

Usage:
    python scripts/train_episode.py \
        --data_dir data/stage2_episode_synth \
        --output_dir runs/stage2_episode \
        --epochs 150

Can also train on mixed real+synthetic data by comma-separating dirs:
    python scripts/train_episode.py \
        --data_dir data/stage2_episode_synth,data/stage2_episode_real \
        --output_dir runs/stage2_episode
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import ModelConfig, TrainConfig, SignalConfig, EpisodeConfig
from trainers.trainer import EpisodeTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True,
                    help='Comma-separated data directories')
    ap.add_argument('--output_dir', default='runs/stage2_episode')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=10)
    ap.add_argument('--episode_gap_ms', type=float, default=600.0,
                    help='Episode boundary gap threshold in ms')
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    scfg = SignalConfig()
    mcfg = ModelConfig(
        input_channels=scfg.input_channels,
        hidden_channels=args.hidden,
        num_layers=args.layers,
    )
    tcfg = TrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    ecfg = EpisodeConfig(
        episode_gap_ms=args.episode_gap_ms,
    )

    trainer = EpisodeTrainer(
        mcfg, tcfg, scfg, ecfg,
        args.data_dir, args.output_dir, args.device,
    )
    trainer.train()


if __name__ == '__main__':
    main()
