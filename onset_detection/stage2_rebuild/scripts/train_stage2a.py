#!/usr/bin/env python3
"""
Script: Train Stage 2A (Group Segmentor).

Usage:
    python scripts/train_stage2a.py \
        --data_dir data/processed/stage2_synthetic_mixed \
        --output_dir runs/stage2a \
        --sample_rate 190 \
        --epochs 100 \
        --batch_size 8 \
        --lr 5e-4
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Stage2AConfig, SignalConfig
from trainers.train_stage2a import Stage2ATrainer


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2A Group Segmentor")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to synthetic_mixed (or mixed_training) data')
    parser.add_argument('--output_dir', type=str, default='runs/stage2a')
    parser.add_argument('--sample_rate', type=int, default=190)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--smoothing_weight', type=float, default=0.15)
    parser.add_argument('--expected_groups', type=int, default=5)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--init_ckpt', type=str, default='',
                        help='Optional checkpoint to initialize model weights from')
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 2A: Group Segmentor Training")
    print("=" * 60)

    signal_config = SignalConfig(sample_rate=args.sample_rate)

    # Determine input channels
    input_ch = 8 if signal_config.use_magnitude else 6

    stage2a_config = Stage2AConfig(
        input_channels=input_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        smoothing_weight=args.smoothing_weight,
        expected_groups=args.expected_groups,
    )

    trainer = Stage2ATrainer(
        config=stage2a_config,
        signal_config=signal_config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.init_ckpt:
        import torch
        ckpt = torch.load(args.init_ckpt, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Initialized model weights from: {args.init_ckpt}")

    best_ckpt = trainer.train()
    print(f"\nTraining complete. Best checkpoint: {best_ckpt}")


if __name__ == '__main__':
    main()
