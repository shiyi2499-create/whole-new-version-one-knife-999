"""
Training loop for Open Stage 2: 3-class frame-level TCN.
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from models.tcn import OpenTCN
from models.losses import OpenFrameLoss
from data.datasets import OpenFrameDataset
from utils.decoder import decode_frame_labels
from utils.metrics import frame_accuracy, full_eval
from configs.config import ModelConfig, TrainConfig, SignalConfig, DecoderConfig


class OpenTrainer:

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 signal_cfg: SignalConfig, decoder_cfg: DecoderConfig,
                 data_dir: str, output_dir: str, device='auto'):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.scfg = signal_cfg
        self.dcfg = decoder_cfg
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available()
            else device if device != 'auto' else 'cpu'
        )

        self.model = OpenTCN(
            in_ch=model_cfg.input_channels,
            hidden=model_cfg.hidden_channels,
            num_layers=model_cfg.num_layers,
            kernel=model_cfg.kernel_size,
            dropout=model_cfg.dropout,
            num_classes=model_cfg.num_classes,
        ).to(self.device)

        nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"OpenTCN: {nparams:,} params on {self.device}")

        self.criterion = OpenFrameLoss(
            class_weights=train_cfg.class_weights,
            smooth_weight=train_cfg.smoothing_weight,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )

        self.train_ds = OpenFrameDataset(
            data_dir, 'train', signal_cfg.sample_rate,
            signal_cfg.use_magnitude, signal_cfg.normalize,
        )
        self.val_ds = OpenFrameDataset(
            data_dir, 'val', signal_cfg.sample_rate,
            signal_cfg.use_magnitude, signal_cfg.normalize,
        )
        self.train_loader = DataLoader(
            self.train_ds, train_cfg.batch_size, shuffle=True,
            collate_fn=OpenFrameDataset.collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, train_cfg.batch_size, shuffle=False,
            collate_fn=OpenFrameDataset.collate,
        )
        print(f"Data: train={len(self.train_ds)}, val={len(self.val_ds)}")

    def _train_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        for x, y, mask in self.train_loader:
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total, n = 0.0, 0
        all_pred, all_gt, all_mask = [], [], []

        for x, y, mask in self.val_loader:
            x, y, mask = x.to(self.device), y.to(self.device), mask.to(self.device)
            logits = self.model(x)
            loss = self.criterion(logits, y, mask)
            total += loss.item()
            n += 1

            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.append(preds)
            all_gt.append(y.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

        val_loss = total / max(n, 1)

        # Frame accuracy
        pred_flat = np.concatenate([p.flatten() for p in all_pred])
        gt_flat = np.concatenate([g.flatten() for g in all_gt])
        mask_flat = np.concatenate([m.flatten() for m in all_mask])
        fa = frame_accuracy(pred_flat, gt_flat, mask_flat)

        return val_loss, fa

    def train(self) -> str:
        best_loss = float('inf')
        best_epoch = 0
        patience = 0
        ckpt_path = str(self.out / 'best.pt')
        log_path = self.out / 'log.txt'

        print(f"\nTraining for {self.tcfg.num_epochs} epochs...")

        for ep in range(1, self.tcfg.num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch()
            val_loss, fa = self._validate()
            elapsed = time.time() - t0
            self.scheduler.step(val_loss)

            line = (f"Ep {ep:3d}/{self.tcfg.num_epochs} | "
                    f"trn={train_loss:.4f} val={val_loss:.4f} | "
                    f"acc={fa.get('overall', 0):.3f} "
                    f"ks={fa.get('per_class', {}).get(1, 0):.3f} "
                    f"sep={fa.get('per_class', {}).get(2, 0):.3f} | "
                    f"{elapsed:.1f}s")

            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = ep
                patience = 0
                torch.save({
                    'epoch': ep,
                    'model': self.model.state_dict(),
                    'val_loss': best_loss,
                    'frame_acc': fa,
                    'model_cfg': self.mcfg,
                    'decoder_cfg': self.dcfg,
                }, ckpt_path)
                line += " ★"
            else:
                patience += 1

            print(line)
            with open(log_path, 'a') as f:
                f.write(line + '\n')

            if patience >= self.tcfg.patience:
                print(f"Early stop at ep {ep} (best {best_epoch})")
                break

        torch.save({'epoch': ep, 'model': self.model.state_dict(),
                     'model_cfg': self.mcfg},
                    str(self.out / 'last.pt'))

        print(f"\nBest: ep {best_epoch}, val_loss={best_loss:.4f}")
        return ckpt_path
