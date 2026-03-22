"""
Training loop for Stage 2B: Onset Detector.
"""
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from models.stage2b import OnsetDetector
from models.losses import Stage2BLoss
from data.datasets import Stage2BDataset
from utils.metrics import compute_onset_metrics
from configs.config import Stage2BConfig, SignalConfig


class Stage2BTrainer:
    """Train and evaluate the Onset Detector."""

    def __init__(self,
                 config: Stage2BConfig,
                 signal_config: SignalConfig,
                 data_dir: str,
                 output_dir: str,
                 device: str = 'auto'):
        self.config = config
        self.signal_config = signal_config
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Stage2B Trainer: device={self.device}")

        # Build model
        self.model = OnsetDetector(config).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {param_count:,}")

        # Loss
        self.criterion = Stage2BLoss(
            use_focal=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8
        )

        # Datasets
        self.train_dataset = Stage2BDataset(
            data_dir, split='train',
            sample_rate=signal_config.sample_rate,
            gaussian_sigma_ms=config.gaussian_sigma_ms,
            add_magnitude=signal_config.use_magnitude,
            expected_onsets=config.expected_onsets,
        )
        self.val_dataset = Stage2BDataset(
            data_dir, split='val',
            sample_rate=signal_config.sample_rate,
            gaussian_sigma_ms=config.gaussian_sigma_ms,
            add_magnitude=signal_config.use_magnitude,
            expected_onsets=config.expected_onsets,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=Stage2BDataset.collate_fn,
            num_workers=0,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=Stage2BDataset.collate_fn,
            num_workers=0,
        )

        print(f"  Train: {len(self.train_dataset)} segments, "
              f"Val: {len(self.val_dataset)} segments")

        # Compute min IKI in samples
        self.min_iki_samples = int(config.min_iki_ms / 1000.0 * signal_config.sample_rate)
        # Tolerance for onset matching (50ms)
        self.tolerance_samples = int(0.05 * signal_config.sample_rate)

    def train_epoch(self) -> Dict:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y, mask, onsets_list in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {'loss': total_loss / max(num_batches, 1)}

    @torch.no_grad()
    def validate(self) -> Dict:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_onset_metrics = []

        for x, y, mask, onsets_list in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y, mask)
            total_loss += loss.item()
            num_batches += 1

            # Compute onset-level metrics
            probs = torch.sigmoid(logits.squeeze(1))  # [B, T]

            for b in range(probs.shape[0]):
                p = probs[b].cpu().numpy()
                m = mask[b].cpu().numpy()
                valid_len = int(m.sum())
                p = p[:valid_len]

                gt_onsets = onsets_list[b]
                gt_onsets = gt_onsets[gt_onsets >= 0]  # remove padding (-1)

                # Peak picking
                pred_onsets = OnsetDetector.pick_peaks(
                    p,
                    expected_onsets=self.config.expected_onsets,
                    min_iki_samples=self.min_iki_samples,
                    base_threshold=self.config.peak_height_threshold,
                    fallback_thresholds=self.config.fallback_thresholds,
                )

                if len(gt_onsets) > 0:
                    metrics = compute_onset_metrics(
                        pred_onsets, gt_onsets,
                        tolerance_samples=self.tolerance_samples,
                        sample_rate=self.signal_config.sample_rate,
                    )
                    all_onset_metrics.append(metrics)

        avg_loss = total_loss / max(num_batches, 1)

        if all_onset_metrics:
            avg_f1 = float(np.mean([m['f1'] for m in all_onset_metrics]))
            avg_recall = float(np.mean([m['recall'] for m in all_onset_metrics]))
            avg_precision = float(np.mean([m['precision'] for m in all_onset_metrics]))
            avg_mae = float(np.mean([m['mean_abs_error_ms'] for m in all_onset_metrics
                                      if m['mean_abs_error_ms'] < 1000]))
        else:
            avg_f1 = avg_recall = avg_precision = 0.0
            avg_mae = float('inf')

        return {
            'loss': avg_loss,
            'onset_f1': avg_f1,
            'onset_recall': avg_recall,
            'onset_precision': avg_precision,
            'onset_mae_ms': avg_mae,
        }

    def train(self) -> str:
        """Full training loop. Returns path to best checkpoint."""
        best_val_loss = float('inf')
        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        best_ckpt_path = str(self.output_dir / "best.pt")
        log_path = self.output_dir / "training_log.txt"

        print(f"\nStarting Stage 2B training for {self.config.num_epochs} epochs...")
        print(f"  Output: {self.output_dir}")
        print(f"  Min IKI: {self.min_iki_samples} samples "
              f"({self.config.min_iki_ms}ms)")
        print(f"  Tolerance: {self.tolerance_samples} samples "
              f"({self.tolerance_samples / self.signal_config.sample_rate * 1000:.0f}ms)")

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            elapsed = time.time() - t0
            self.scheduler.step(val_metrics['loss'])

            log_line = (
                f"Epoch {epoch:3d}/{self.config.num_epochs} | "
                f"Train loss: {train_metrics['loss']:.4f} | "
                f"Val loss: {val_metrics['loss']:.4f} | "
                f"F1: {val_metrics['onset_f1']:.3f} "
                f"R: {val_metrics['onset_recall']:.3f} "
                f"P: {val_metrics['onset_precision']:.3f} "
                f"MAE: {val_metrics['onset_mae_ms']:.1f}ms | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            # Save best by loss (or by F1 - uncomment to switch)
            improved = val_metrics['loss'] < best_val_loss
            # improved = val_metrics['onset_f1'] > best_val_f1

            if improved:
                best_val_loss = val_metrics['loss']
                best_val_f1 = val_metrics['onset_f1']
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_f1': best_val_f1,
                    'val_metrics': val_metrics,
                    'config': self.config,
                }, best_ckpt_path)
                log_line += " ★"
            else:
                patience_counter += 1

            print(log_line)
            with open(log_path, 'a') as f:
                f.write(log_line + "\n")

            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch} (best: {best_epoch})")
                break

        # Save final
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, str(self.output_dir / "last.pt"))

        print(f"\nBest model: epoch {best_epoch}, "
              f"val_loss={best_val_loss:.4f}, val_f1={best_val_f1:.3f}")
        print(f"Saved to: {best_ckpt_path}")

        return best_ckpt_path
