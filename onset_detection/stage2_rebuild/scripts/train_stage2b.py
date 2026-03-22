#!/usr/bin/env python3
"""
Script: Train Stage 2B (Onset Detector).

Usage:
    python scripts/train_stage2b.py \
        --data_dir data/processed/stage2_synthetic_mixed \
        --output_dir runs/stage2b \
        --sample_rate 190 \
        --epochs 150 \
        --batch_size 32 \
        --lr 5e-4
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Stage2BConfig, SignalConfig
from trainers.train_stage2b import Stage2BTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Stage 2B Onset Detector")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to synthetic_mixed (or mixed_training) data')
    parser.add_argument('--output_dir', type=str, default='runs/stage2b')
    parser.add_argument('--sample_rate', type=int, default=190)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gaussian_sigma_ms', type=float, default=15.0)
    parser.add_argument('--expected_onsets', type=int, default=8)
    parser.add_argument('--min_iki_ms', type=float, default=50.0)
    parser.add_argument('--use_focal_loss', action='store_true', default=True)
    parser.add_argument('--no_focal_loss', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--init_ckpt', type=str, default='',
                        help='Optional checkpoint to initialize model weights from')
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 2B: Onset Detector Training")
    print("=" * 60)

    signal_config = SignalConfig(sample_rate=args.sample_rate)
    input_ch = 8 if signal_config.use_magnitude else 6

    use_focal = args.use_focal_loss and not args.no_focal_loss

    stage2b_config = Stage2BConfig(
        input_channels=input_ch,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gaussian_sigma_ms=args.gaussian_sigma_ms,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        use_focal_loss=use_focal,
        min_iki_ms=args.min_iki_ms,
        expected_onsets=args.expected_onsets,
    )

    trainer = Stage2BTrainer(
        config=stage2b_config,
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
