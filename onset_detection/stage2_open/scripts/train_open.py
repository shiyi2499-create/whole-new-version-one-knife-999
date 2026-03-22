#!/usr/bin/env python3
"""
Train the Open Stage 2 frame-wise 3-class TCN.

python scripts/train_open.py \
    --data_dir data/synthetic_open \
    --output_dir runs/stage2_open
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import ModelConfig, TrainConfig, SignalConfig, DecoderConfig
from trainers.trainer import OpenTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--output_dir', default='runs/stage2_open')
    ap.add_argument('--sample_rate', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=10)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    scfg = SignalConfig(sample_rate=args.sample_rate)
    mcfg = ModelConfig(
        input_channels=scfg.input_channels,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    )
    tcfg = TrainConfig(
        lr=args.lr, batch_size=args.batch_size,
        num_epochs=args.epochs,
    )
    dcfg = DecoderConfig()

    trainer = OpenTrainer(mcfg, tcfg, scfg, dcfg,
                          args.data_dir, args.output_dir, args.device)
    trainer.train()


if __name__ == '__main__':
    main()
