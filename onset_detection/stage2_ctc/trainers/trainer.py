"""
Training loop for Frame-level CTC character decoder.

Key differences from stage2_episode trainer:
  - Loss: frame-CE + CTC (not typing-CE + onset-MSE)
  - Validation: CTC greedy decode + CER (not episode detection rate)
  - No onset head, no typing/silence frame accuracy
  - Each sample is one password episode, not a whole session
"""
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from models.frame_ctc import FrameCTCModel, init_from_onset_checkpoint
from models.losses import FrameCTCLoss
from data.datasets import CTCEpisodeDataset
from utils.decode import greedy_decode
from utils.metrics import cer, levenshtein
from utils.vocab import BLANK_IDX, NUM_CLASSES
from configs.config import ModelConfig, TrainConfig, SignalConfig


class CTCTrainer:

    def _estimate_class_weights(self):
        counts = np.ones(NUM_CLASSES, dtype=np.float64)
        for i in range(len(self.train_ds)):
            _, _, _, ct = self.train_ds[i]
            for idx in ct.tolist():
                counts[int(idx)] += 1.0

        # Mild inverse-sqrt reweighting to reduce high-frequency character bias
        # without making rare classes explode.
        weights = counts.mean() / np.sqrt(counts)
        weights = weights / np.mean(weights[1:])  # normalize nonblank around 1
        weights[BLANK_IDX] = self.tcfg.blank_ce_weight
        return torch.tensor(weights, dtype=torch.float32)

    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig,
                 signal_cfg: SignalConfig, data_dir: str, output_dir: str,
                 device='auto'):
        self.mcfg = model_cfg
        self.tcfg = train_cfg
        self.scfg = signal_cfg
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Model
        self.model = FrameCTCModel(
            in_ch=model_cfg.input_channels,
            hidden=model_cfg.hidden_channels,
            num_layers=model_cfg.num_layers,
            kernel=model_cfg.kernel_size,
            dropout=model_cfg.dropout,
            num_classes=model_cfg.num_classes,
        ).to(self.device)

        # Optional: init backbone from onset checkpoint
        if train_cfg.onset_checkpoint:
            self.model = init_from_onset_checkpoint(
                self.model, train_cfg.onset_checkpoint, self.device
            )
        if getattr(train_cfg, 'resume_checkpoint', ''):
            ckpt = torch.load(train_cfg.resume_checkpoint, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
            print(f"[FrameCTCModel] Resumed model weights from {train_cfg.resume_checkpoint}")
        if getattr(train_cfg, 'freeze_backbone', False):
            for name, param in self.model.named_parameters():
                if not name.startswith('char_head.'):
                    param.requires_grad = False
            print("[FrameCTCModel] Backbone frozen; finetuning char_head only")

        nparams = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"FrameCTCModel: {nparams:,} params on {self.device}")

        # Data
        sigma_ms = train_cfg.keystroke_sigma_ms
        self.train_ds = CTCEpisodeDataset(
            data_dir, 'train', signal_cfg.sample_rate,
            signal_cfg.use_magnitude, signal_cfg.normalize,
            sigma_ms=sigma_ms,
        )
        self.val_ds = CTCEpisodeDataset(
            data_dir, 'val', signal_cfg.sample_rate,
            signal_cfg.use_magnitude, signal_cfg.normalize,
            sigma_ms=sigma_ms,
        )
        self.train_loader = DataLoader(
            self.train_ds, train_cfg.batch_size, shuffle=True,
            collate_fn=CTCEpisodeDataset.collate, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_ds, train_cfg.batch_size, shuffle=False,
            collate_fn=CTCEpisodeDataset.collate,
        )
        print(f"Data: train={len(self.train_ds)}, val={len(self.val_ds)}")

        class_weights = self._estimate_class_weights()
        print("Frame CE class weights (sample):",
              {i: round(float(class_weights[i]), 3) for i in range(min(12, len(class_weights)))})

        # Loss
        self.criterion = FrameCTCLoss(
            num_classes=model_cfg.num_classes,
            frame_weight=train_cfg.frame_ce_weight,
            ctc_weight=train_cfg.ctc_weight,
            blank_ce_weight=train_cfg.blank_ce_weight,
            label_smoothing=train_cfg.label_smoothing,
            class_weights=class_weights,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8
        )

    def _scheduled_ctc_weight(self, epoch: int) -> float:
        base = float(self.tcfg.ctc_weight)
        warm = int(getattr(self.tcfg, 'ctc_warmup_epochs', 0))
        ramp = int(getattr(self.tcfg, 'ctc_ramp_epochs', 1))
        if epoch <= warm:
            return 0.0
        if ramp <= 0:
            return base
        progress = min(1.0, max(0.0, (epoch - warm) / float(ramp)))
        return base * progress

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss, n = 0.0, 0
        parts_sum = {'frame_ce': 0.0, 'ctc': 0.0}
        ctc_w = self._scheduled_ctc_weight(epoch)

        for batch in self.train_loader:
            (x, ht, sw, mask, ctc_tgt, ctc_tgt_len, inp_len) = batch
            x = x.to(self.device)
            ht = ht.to(self.device)
            sw = sw.to(self.device)
            mask = mask.to(self.device)
            ctc_tgt = ctc_tgt.to(self.device)
            ctc_tgt_len = ctc_tgt_len.to(self.device)
            inp_len = inp_len.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x)  # [B, C, T]

            result = self.criterion(
                logits, ht, sw, mask,
                ctc_tgt, ctc_tgt_len, inp_len,
                ctc_weight_override=ctc_w,
            )
            loss = result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            total_loss += loss.item()
            parts_sum['frame_ce'] += result['frame_ce'].item()
            parts_sum['ctc'] += result['ctc'].item()
            n += 1

        avg = total_loss / max(n, 1)
        avg_parts = {k: v / max(n, 1) for k, v in parts_sum.items()}
        return avg, avg_parts, ctc_w

    @torch.no_grad()
    def _validate(self, epoch: int):
        self.model.eval()
        total_loss, n = 0.0, 0
        total_edits, total_chars = 0, 0
        examples = []
        ctc_w = self._scheduled_ctc_weight(epoch)

        for batch in self.val_loader:
            (x, ht, sw, mask, ctc_tgt, ctc_tgt_len, inp_len) = batch
            x = x.to(self.device)
            ht = ht.to(self.device)
            sw = sw.to(self.device)
            mask = mask.to(self.device)
            ctc_tgt = ctc_tgt.to(self.device)
            ctc_tgt_len = ctc_tgt_len.to(self.device)
            inp_len = inp_len.to(self.device)

            logits = self.model(x)

            result = self.criterion(
                logits, ht, sw, mask,
                ctc_tgt, ctc_tgt_len, inp_len,
                ctc_weight_override=ctc_w,
            )
            total_loss += result['loss'].item()
            n += 1

            # Decode and compute CER
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            B = x.shape[0]
            offset = 0
            for i in range(B):
                T_i = int(inp_len[i])
                K_i = int(ctc_tgt_len[i])
                lp = log_probs[i, :, :T_i].cpu().numpy()  # [C, T_i]
                hyp = greedy_decode(lp)

                ref_indices = ctc_tgt[offset:offset + K_i].cpu().numpy()
                ref = ''.join(
                    chr(ord('a') + idx - 1) if 1 <= idx <= 26
                    else str(idx - 27) if 27 <= idx <= 36
                    else '?' for idx in ref_indices
                )
                offset += K_i

                edits = levenshtein(ref, hyp)
                total_edits += edits
                total_chars += len(ref)

                if len(examples) < 8:
                    examples.append((ref, hyp))

        val_loss = total_loss / max(n, 1)
        val_cer = total_edits / max(total_chars, 1)
        return val_loss, val_cer, examples, ctc_w

    def train(self) -> str:
        best_loss = float('inf')
        best_cer = float('inf')
        best_epoch = 0
        best_cer_epoch = 0
        patience = 0
        ckpt_path = str(self.out / 'best.pt')
        ckpt_cer_path = str(self.out / 'best_cer.pt')
        log_path = self.out / 'log.jsonl'

        print(f"\nTraining for {self.tcfg.num_epochs} epochs...\n")

        for ep in range(1, self.tcfg.num_epochs + 1):
            t0 = time.time()
            train_loss, train_parts, train_ctc_w = self._train_epoch(ep)
            val_loss, val_cer, examples, val_ctc_w = self._validate(ep)
            elapsed = time.time() - t0

            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']

            line = (f"Ep {ep:3d}/{self.tcfg.num_epochs} | "
                    f"trn={train_loss:.4f} "
                    f"(ce={train_parts['frame_ce']:.3f} "
                    f"ctc={train_parts['ctc']:.3f}) "
                    f"val={val_loss:.4f} "
                    f"CER={val_cer:.3f} "
                    f"ctc_w={train_ctc_w:.3f} "
                    f"lr={lr:.1e} | "
                    f"{elapsed:.1f}s")

            improved = False
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = ep
                patience = 0
                improved = True
                torch.save({
                    'epoch': ep,
                    'model': self.model.state_dict(),
                    'val_loss': best_loss,
                    'val_cer': val_cer,
                    'model_cfg': {
                        'input_channels': self.mcfg.input_channels,
                        'hidden_channels': self.mcfg.hidden_channels,
                        'num_layers': self.mcfg.num_layers,
                        'kernel_size': self.mcfg.kernel_size,
                        'dropout': self.mcfg.dropout,
                        'num_classes': self.mcfg.num_classes,
                    },
                }, ckpt_path)
                line += " *"
            else:
                patience += 1

            if val_cer < best_cer:
                best_cer = val_cer
                best_cer_epoch = ep
                torch.save({
                    'epoch': ep,
                    'model': self.model.state_dict(),
                    'val_loss': val_loss,
                    'val_cer': best_cer,
                    'model_cfg': {
                        'input_channels': self.mcfg.input_channels,
                        'hidden_channels': self.mcfg.hidden_channels,
                        'num_layers': self.mcfg.num_layers,
                        'kernel_size': self.mcfg.kernel_size,
                        'dropout': self.mcfg.dropout,
                        'num_classes': self.mcfg.num_classes,
                    },
                }, ckpt_cer_path)

            # Show examples periodically
            if ep % 10 == 0 or ep == 1 or improved:
                for ref, hyp in examples[:3]:
                    print(f"    ref={ref}  hyp={hyp}")

            print(line)

            log_entry = {
                'epoch': ep, 'train_loss': train_loss,
                'val_loss': val_loss, 'val_cer': val_cer,
                'lr': lr, 'elapsed': elapsed, 'ctc_weight': train_ctc_w,
            }
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            if patience >= self.tcfg.patience:
                print(f"Early stop at ep {ep} (best {best_epoch})")
                break

        # Save last checkpoint
        torch.save({
            'epoch': ep, 'model': self.model.state_dict(),
            'model_cfg': {
                'input_channels': self.mcfg.input_channels,
                'hidden_channels': self.mcfg.hidden_channels,
                'num_layers': self.mcfg.num_layers,
                'kernel_size': self.mcfg.kernel_size,
                'dropout': self.mcfg.dropout,
                'num_classes': self.mcfg.num_classes,
            },
        }, str(self.out / 'last.pt'))

        print(f"\nBest loss: ep {best_epoch}, val_loss={best_loss:.4f}")
        print(f"Best CER: ep {best_cer_epoch}, CER={best_cer:.3f}")
        return ckpt_path
