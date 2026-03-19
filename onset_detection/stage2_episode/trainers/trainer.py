"""
Training loop for Episode-based Stage 2: dual-head frame-level TCN.

Dual-head model:
  - typing_head: 2-class (silence vs typing) — same as before
  - onset_head:  Gaussian impulse regression — NEW

The onset_head loss is only computed when the batch has non-zero onset_targets
(i.e., data generated with the new synthesis pipeline). Old data (all-zero
onset_targets) skips the onset loss automatically, so mixed old+new training
works out of the box.

Key changes from the single-head trainer:
  - collate returns (x, y, onset_targets, mask) instead of (x, y, mask)
  - model.forward() returns (typing_logits, onset_logits)
  - loss takes typing_logits + onset_logits + onset_targets
  - episode eval uses onset_probs from onset_head when available
"""
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict

from models.tcn import EpisodeTCN
from models.losses import EpisodeFrameLoss
from data.datasets import EpisodeFrameDataset
from utils.decoder import decode_episodes
from utils.metrics import frame_accuracy_2class, full_eval
from configs.config import ModelConfig, TrainConfig, SignalConfig, EpisodeConfig


class EpisodeTrainer:

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 signal_cfg: SignalConfig, episode_cfg: EpisodeConfig,
                 data_dir: str, output_dir: str, device='auto'):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.scfg = signal_cfg
        self.ecfg = episode_cfg
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available()
            else device if device != 'auto' else 'cpu'
        )

        # Dual-head model (use_onset_head=True by default for new training)
        self.model = EpisodeTCN(
            in_ch=model_cfg.input_channels,
            hidden=model_cfg.hidden_channels,
            num_layers=model_cfg.num_layers,
            kernel=model_cfg.kernel_size,
            dropout=model_cfg.dropout,
            num_classes=model_cfg.num_classes,
            use_onset_head=True,
        ).to(self.device)

        nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"EpisodeTCN (dual-head): {nparams:,} params on {self.device}")

        self.criterion = EpisodeFrameLoss(
            class_weights=train_cfg.class_weights,
            smooth_weight=train_cfg.smoothing_weight,
            onset_weight=train_cfg.onset_loss_weight,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=7
        )

        self.train_ds = EpisodeFrameDataset(
            data_dir, 'train', signal_cfg.sample_rate,
            signal_cfg.use_magnitude, signal_cfg.normalize,
        )
        self.val_ds = EpisodeFrameDataset(
            data_dir, 'val', signal_cfg.sample_rate,
            signal_cfg.use_magnitude, signal_cfg.normalize,
        )
        self.train_loader = DataLoader(
            self.train_ds, train_cfg.batch_size, shuffle=True,
            collate_fn=EpisodeFrameDataset.collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, train_cfg.batch_size, shuffle=False,
            collate_fn=EpisodeFrameDataset.collate,
        )
        print(f"Data: train={len(self.train_ds)}, val={len(self.val_ds)}")

    def _train_epoch(self):
        self.model.train()
        total, n = 0.0, 0
        loss_parts = {'typing_loss': 0.0, 'onset_loss': 0.0}

        # collate now returns (x, y, onset_targets, mask)
        for x, y, onset_targets, mask in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            onset_targets = onset_targets.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()
            typing_logits, onset_logits = self.model(x)

            # Only pass onset targets to loss if this batch has real targets
            # (all-zero onset_targets = old data without onset supervision)
            has_onset_sup = onset_targets.sum() > 0
            loss, parts = self.criterion(
                typing_logits, y, mask,
                onset_logits=onset_logits if has_onset_sup else None,
                onset_targets=onset_targets if has_onset_sup else None,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            total += loss.item()
            for k in loss_parts:
                loss_parts[k] += parts.get(k, 0.0)
            n += 1

        avg = total / max(n, 1)
        avg_parts = {k: v / max(n, 1) for k, v in loss_parts.items()}
        return avg, avg_parts

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        total, n = 0.0, 0
        all_pred, all_gt, all_mask = [], [], []

        for x, y, onset_targets, mask in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            onset_targets = onset_targets.to(self.device)
            mask = mask.to(self.device)

            typing_logits, onset_logits = self.model(x)

            has_onset_sup = onset_targets.sum() > 0
            loss, _ = self.criterion(
                typing_logits, y, mask,
                onset_logits=onset_logits if has_onset_sup else None,
                onset_targets=onset_targets if has_onset_sup else None,
            )
            total += loss.item()
            n += 1

            preds = typing_logits.argmax(dim=1).cpu().numpy()
            all_pred.append(preds)
            all_gt.append(y.cpu().numpy())
            all_mask.append(mask.cpu().numpy())

        val_loss = total / max(n, 1)

        pred_flat = np.concatenate([p.flatten() for p in all_pred])
        gt_flat = np.concatenate([g.flatten() for g in all_gt])
        mask_flat = np.concatenate([m.flatten() for m in all_mask])
        fa = frame_accuracy_2class(pred_flat, gt_flat, mask_flat)

        return val_loss, fa

    @torch.no_grad()
    def _episode_eval(self, max_samples: int = 20):
        """
        Run full episode decode on a subset of val set.
        Uses onset_head probs for peak picking when available.
        Returns average episode detection rate and onset F1.
        """
        self.model.eval()
        sr = self.scfg.sample_rate

        det_rates = []
        onset_f1s = []
        count = 0

        for f in self.val_ds.files[:max_samples]:
            d = np.load(f, allow_pickle=True)
            imu = d['imu']
            labels = d['frame_labels']

            if labels.max() > 1:
                labels = (labels == 1).astype(np.int64)

            gt_episodes = []
            for key in ['episodes_json', 'groups_json']:
                if key in d:
                    gt_raw = json.loads(str(d[key]))
                    for g in gt_raw:
                        gt_episodes.append({
                            'start': g['start'],
                            'end': g['end'],
                            'onsets': g.get('onsets', []),
                            'num_keys': g.get('num_keys', len(g.get('onsets', []))),
                        })
                    break

            if not gt_episodes:
                continue

            from utils.signal_processing import preprocess
            proc, _ = preprocess(imu, sr, self.scfg.use_magnitude, self.scfg.normalize)
            x = torch.from_numpy(proc.T).float().unsqueeze(0).to(self.device)

            typing_logits, onset_logits = self.model(x)
            preds = typing_logits.argmax(dim=1)[0].cpu().numpy()
            typing_probs = torch.softmax(typing_logits, dim=1)[0, 1].cpu().numpy()

            # Use onset head probs if available
            onset_probs = None
            if onset_logits is not None:
                onset_probs = torch.sigmoid(onset_logits)[0, 0].cpu().numpy()

            dec = decode_episodes(
                preds,
                raw_imu=imu,
                typing_probs=typing_probs,
                onset_probs=onset_probs,
                sample_rate=sr,
                median_kernel=self.ecfg.median_kernel,
                min_typing_run_ms=self.ecfg.min_typing_run_ms,
                episode_gap_ms=self.ecfg.episode_gap_ms,
                min_onset_gap_ms=self.ecfg.min_onset_gap_ms,
                min_episode_keys=self.ecfg.min_episode_keys,
                min_episode_duration_ms=self.ecfg.min_episode_duration_ms,
            )

            ev = full_eval(dec['episodes'], gt_episodes, sr)
            det_rates.append(ev['episode_detection_rate'])
            onset_f1s.append(ev['avg_onset_f1'])
            count += 1

        if count == 0:
            return 0.0, 0.0

        return float(np.mean(det_rates)), float(np.mean(onset_f1s))

    def train(self) -> str:
        best_loss = float('inf')
        best_epoch = 0
        patience = 0
        ckpt_path = str(self.out / 'best.pt')
        log_path = self.out / 'log.txt'

        print(f"\nTraining for {self.tcfg.num_epochs} epochs...")

        for ep in range(1, self.tcfg.num_epochs + 1):
            t0 = time.time()
            train_loss, train_parts = self._train_epoch()
            val_loss, fa = self._validate()
            elapsed = time.time() - t0
            self.scheduler.step(val_loss)

            pc = fa.get('per_class', {})
            line = (f"Ep {ep:3d}/{self.tcfg.num_epochs} | "
                    f"trn={train_loss:.4f} "
                    f"(typ={train_parts['typing_loss']:.3f} "
                    f"onset={train_parts['onset_loss']:.3f}) "
                    f"val={val_loss:.4f} | "
                    f"acc={fa.get('overall', 0):.3f} "
                    f"typing={pc.get('typing', 0):.3f} "
                    f"silence={pc.get('silence', 0):.3f} | "
                    f"{elapsed:.1f}s")

            if ep % 10 == 0 or ep == 1:
                det_rate, onset_f1 = self._episode_eval()
                line += f" | det={det_rate:.3f} f1={onset_f1:.3f}"

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
                    'episode_cfg': self.ecfg,
                    'has_onset_head': True,
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
