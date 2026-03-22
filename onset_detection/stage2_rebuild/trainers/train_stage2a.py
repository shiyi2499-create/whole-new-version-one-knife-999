"""
Training loop for Stage 2A: Group Segmentor.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict

from models.stage2a import GroupSegmentor
from models.losses import Stage2ALoss
from data.datasets import Stage2ADataset
from utils.metrics import compute_group_iou
from configs.config import Stage2AConfig, SignalConfig


class Stage2ATrainer:
    """Train and evaluate the Group Segmentor."""

    def __init__(self,
                 config: Stage2AConfig,
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

        print(f"Stage2A Trainer: device={self.device}")

        # Build model
        self.model = GroupSegmentor(config, use_multistage=False).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Model parameters: {param_count:,}")

        # Loss
        self.criterion = Stage2ALoss(
            bce_weight=config.bce_weight,
            smoothing_weight=config.smoothing_weight,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )

        # Datasets
        self.train_dataset = Stage2ADataset(
            data_dir, split='train',
            sample_rate=signal_config.sample_rate,
            add_magnitude=signal_config.use_magnitude,
        )
        self.val_dataset = Stage2ADataset(
            data_dir, split='val',
            sample_rate=signal_config.sample_rate,
            add_magnitude=signal_config.use_magnitude,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=Stage2ADataset.collate_fn,
            num_workers=0,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=Stage2ADataset.collate_fn,
            num_workers=0,
        )

        print(f"  Train: {len(self.train_dataset)} sessions, "
              f"Val: {len(self.val_dataset)} sessions")

    def train_epoch(self) -> Dict:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y, mask in self.train_loader:
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
        all_ious = []
        count_correct = []

        for x, y, mask in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y, mask)
            total_loss += loss.item()
            num_batches += 1

            # Compute group-level metrics
            if isinstance(logits, list):
                logits = logits[-1]
            probs = torch.sigmoid(logits.squeeze(1))  # [B, T]

            for b in range(probs.shape[0]):
                p = probs[b].cpu().numpy()
                gt = y[b].cpu().numpy()
                m = mask[b].cpu().numpy()

                # Mask out padding
                valid_len = int(m.sum())
                p = p[:valid_len]
                gt = gt[:valid_len]

                # Post-process predictions
                pred_groups = GroupSegmentor.post_process(
                    p,
                    sample_rate=self.signal_config.sample_rate,
                    median_kernel=self.config.median_filter_kernel,
                    threshold=self.config.threshold,
                    min_group_duration_s=self.config.min_group_duration_s,
                    expected_groups=self.config.expected_groups,
                )

                # Extract GT groups from binary labels
                gt_groups = _extract_gt_groups(gt)

                # Compute IoU
                metrics = compute_group_iou(pred_groups, gt_groups)
                all_ious.append(metrics['mean_iou'])
                count_correct.append(1.0 if metrics['group_count_correct'] else 0.0)

        avg_loss = total_loss / max(num_batches, 1)
        avg_iou = float(np.mean(all_ious)) if all_ious else 0.0

        return {
            'loss': avg_loss,
            'mean_iou': avg_iou,
            'group_count_accuracy': float(np.mean(count_correct)) if count_correct else 0.0,
        }

    def train(self) -> str:
        """Full training loop. Returns path to best checkpoint."""
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_ckpt_path = str(self.output_dir / "best.pt")

        log_path = self.output_dir / "training_log.txt"

        print(f"\nStarting Stage 2A training for {self.config.num_epochs} epochs...")
        print(f"  Output: {self.output_dir}")

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
                f"Val IoU: {val_metrics['mean_iou']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_iou': val_metrics['mean_iou'],
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

        # Save final model too
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, str(self.output_dir / "last.pt"))

        print(f"\nBest model: epoch {best_epoch}, val_loss={best_val_loss:.4f}")
        print(f"Saved to: {best_ckpt_path}")

        return best_ckpt_path


def _extract_gt_groups(labels: np.ndarray,
                       min_duration: int = 10) -> list:
    """Extract group boundaries from binary label array."""
    groups = []
    in_group = False
    start = 0

    for i in range(len(labels)):
        if labels[i] > 0.5 and not in_group:
            start = i
            in_group = True
        elif labels[i] <= 0.5 and in_group:
            if i - start >= min_duration:
                groups.append((start, i))
            in_group = False

    if in_group and len(labels) - start >= min_duration:
        groups.append((start, len(labels)))

    return groups
