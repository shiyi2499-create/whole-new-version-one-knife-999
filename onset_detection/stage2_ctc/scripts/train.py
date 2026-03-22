#!/usr/bin/env python3
"""
Train the Frame-CTC character decoder.

Usage:
    # Basic training on CTC dataset
    python scripts/train.py \
        --data_dir data/stage2_ctc \
        --output_dir runs/stage2_ctc

    # With backbone init from onset checkpoint
    python scripts/train.py \
        --data_dir data/stage2_ctc \
        --output_dir runs/stage2_ctc \
        --onset_checkpoint runs/stage2_episode/best.pt

    # On Mac with MPS
    python scripts/train.py \
        --data_dir data/stage2_ctc \
        --output_dir runs/stage2_ctc \
        --device mps
"""
import sys
import os
import argparse

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from configs.config import ModelConfig, TrainConfig, SignalConfig
from trainers.trainer import CTCTrainer


def main():
    ap = argparse.ArgumentParser(description="Train Frame-CTC model")
    ap.add_argument('--data_dir', required=True,
                    help='CTC episode dataset (can be comma-separated for multiple)')
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    ap.add_argument('--sample_rate', type=int, default=100)

    # Model
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--num_layers', type=int, default=12)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--dropout', type=float, default=0.25)

    # Training
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--frame_ce_weight', type=float, default=1.0)
    ap.add_argument('--ctc_weight', type=float, default=0.05)
    ap.add_argument('--ctc_warmup_epochs', type=int, default=3)
    ap.add_argument('--ctc_ramp_epochs', type=int, default=6)
    ap.add_argument('--sigma_ms', type=float, default=20.0)
    ap.add_argument('--onset_checkpoint', default='',
                    help='Optional: onset TCN checkpoint for backbone init')
    ap.add_argument('--resume_checkpoint', default='',
                    help='Optional: resume / finetune from existing frame-CTC checkpoint')
    ap.add_argument('--freeze_backbone', action='store_true',
                    help='Freeze input_conv + TCN layers and only finetune char head')

    args = ap.parse_args()

    scfg = SignalConfig(sample_rate=args.sample_rate)
    mcfg = ModelConfig(
        input_channels=scfg.input_channels,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        kernel_size=args.kernel,
        dropout=args.dropout,
    )
    tcfg = TrainConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        patience=args.patience,
        frame_ce_weight=args.frame_ce_weight,
        ctc_weight=args.ctc_weight,
        ctc_warmup_epochs=args.ctc_warmup_epochs,
        ctc_ramp_epochs=args.ctc_ramp_epochs,
        keystroke_sigma_ms=args.sigma_ms,
        onset_checkpoint=args.onset_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        freeze_backbone=args.freeze_backbone,
    )

    trainer = CTCTrainer(mcfg, tcfg, scfg, args.data_dir, args.output_dir,
                         device=args.device)
    best_path = trainer.train()
    print(f"\nBest checkpoint: {best_path}")


if __name__ == '__main__':
    main()
