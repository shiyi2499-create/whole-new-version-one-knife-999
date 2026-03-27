from __future__ import annotations

import copy
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "phase3_password_inception")):
    if p not in sys.path:
        sys.path.insert(0, p)

from phase3_password_inception.run_password_closure_inception import (
    InceptionTimeClassifier,
    augment_batch,
)
from phase3_password_inception.stage3_diff_channels import append_diff_channels


@dataclass
class SegmentalConfig:
    input_channels: int = 6
    encoder_hidden: int = 96
    encoder_blocks: int = 6
    encoder_kernel: int = 7
    encoder_dropout: float = 0.10
    target_len: int = 57
    trigger_ratio: float = 1.0 / 3.0
    prior_pre_ms: float = 100.0
    prior_post_ms: float = 200.0
    min_boundary_frac: float = 0.15
    max_margin_scale: float = 2.5
    loss_char: float = 1.0
    loss_ratio: float = 0.20
    loss_duration: float = 0.15
    loss_alpha: float = 0.05


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class EpisodeEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden: int, n_blocks: int, kernel: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        dilations = [2 ** (i % 4) for i in range(n_blocks)]
        self.blocks = nn.ModuleList([
            ResidualDilatedBlock(hidden, kernel=kernel, dilation=d, dropout=dropout)
            for d in dilations
        ])

    def forward(self, x_bt_c: torch.Tensor) -> torch.Tensor:
        x = x_bt_c.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)


class SegmentalClassifier(nn.Module):
    def __init__(
        self,
        target_len: int,
        classes: list[str],
        means: np.ndarray,
        stds: np.ndarray,
        use_diff_channels: bool = False,
    ):
        super().__init__()
        self.target_len = int(target_len)
        self.classes = [str(x) for x in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.use_diff_channels = bool(use_diff_channels)
        self.model = InceptionTimeClassifier(
            n_timesteps=self.target_len,
            n_channels=int(len(means)),
            n_classes=len(self.classes),
        )
        self.register_buffer("means", torch.as_tensor(np.asarray(means, dtype=np.float32)))
        self.register_buffer("stds", torch.as_tensor(np.asarray(stds, dtype=np.float32)))

    def normalize(self, windows: torch.Tensor) -> torch.Tensor:
        if self.use_diff_channels and windows.shape[-1] * 2 == self.means.shape[0]:
            windows_np = append_diff_channels(windows.detach().cpu().numpy())
            windows = torch.as_tensor(windows_np, dtype=windows.dtype, device=windows.device)
        return (windows - self.means.view(1, 1, -1)) / (self.stds.view(1, 1, -1) + 1e-6)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(windows))

    def meta(self) -> dict:
        return {
            "target_len": self.target_len,
            "classes": self.classes,
            "means": self.means.detach().cpu().numpy().tolist(),
            "stds": self.stds.detach().cpu().numpy().tolist(),
            "use_diff_channels": self.use_diff_channels,
        }


class SegmentalSequenceModel(nn.Module):
    def __init__(self, config: SegmentalConfig, classifier: SegmentalClassifier):
        super().__init__()
        self.cfg = config
        self.classifier = classifier
        self.encoder = EpisodeEncoder(
            input_channels=config.input_channels,
            hidden=config.encoder_hidden,
            n_blocks=config.encoder_blocks,
            kernel=config.encoder_kernel,
            dropout=config.encoder_dropout,
        )
        h = config.encoder_hidden
        self.start_head = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1))
        self.end_head = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1))
        self.boundary_head = nn.Sequential(nn.Linear(h * 4 + 1, h), nn.GELU(), nn.Linear(h, 1))

    def freeze_classifier(self, freeze: bool = True):
        for p in self.classifier.parameters():
            p.requires_grad = not freeze

    @staticmethod
    def _gather_feature(feats: torch.Tensor, index: float) -> torch.Tensor:
        T = feats.shape[0]
        idx = int(round(float(index)))
        idx = min(max(idx, 0), T - 1)
        return feats[idx]

    def _margin_from_raw(self, raw: torch.Tensor, prior: float, cap: float) -> torch.Tensor:
        cap = max(cap, 1.0)
        prior = min(max(prior, 1.0), cap)
        scale_cap = max(cap / max(prior, 1e-6), 1.01)
        return torch.clamp(prior * torch.exp(torch.tanh(raw) * math.log(scale_cap)), min=1.0, max=cap)

    def predict_boundaries(self, feats: torch.Tensor, key_frames: torch.Tensor, sample_rate_hz: float) -> dict:
        device = feats.device
        t = key_frames.float().to(device)
        T = feats.shape[0]
        K = int(len(t))
        global_feat = feats.mean(dim=0)
        prior_pre = self.cfg.prior_pre_ms / 1000.0 * sample_rate_hz
        prior_post = self.cfg.prior_post_ms / 1000.0 * sample_rate_hz
        prior_total = prior_pre + prior_post

        key_feats = [self._gather_feature(feats, float(x)) for x in t]
        boundaries: list[torch.Tensor] = []
        alpha_list: list[torch.Tensor] = []

        if K > 1:
            first_gap = float(max(t[1] - t[0], 1.0))
        else:
            first_gap = float(max(prior_total, 2.0))
        start_cap = min(float(t[0]) - 1e-3 if float(t[0]) > 1 else 1.0, max(prior_pre, 1.0) * self.cfg.max_margin_scale)
        start_cap = max(start_cap, 1.0)
        start_raw = self.start_head(torch.cat([global_feat, key_feats[0]], dim=0)).squeeze(-1)
        start_margin = self._margin_from_raw(start_raw, prior_pre, start_cap)
        boundaries.append(torch.clamp(t[0] - start_margin, min=0.0, max=float(t[0]) - 1e-3 if float(t[0]) > 0.5 else 0.0))

        for i in range(K - 1):
            gap = torch.clamp(t[i + 1] - t[i], min=1.0)
            mid = 0.5 * (t[i + 1] + t[i])
            gap_feat = self._gather_feature(feats, float(mid))
            raw = self.boundary_head(
                torch.cat(
                    [key_feats[i], gap_feat, key_feats[i + 1], global_feat, gap.view(1)],
                    dim=0,
                )
            ).squeeze(-1)
            alpha = self.cfg.min_boundary_frac + (1.0 - 2.0 * self.cfg.min_boundary_frac) * torch.sigmoid(raw)
            alpha_list.append(alpha)
            boundaries.append(t[i] + alpha * gap)

        if K > 1:
            last_gap = float(max(t[-1] - t[-2], 1.0))
        else:
            last_gap = float(max(prior_total, 2.0))
        end_cap = min(float(T - 1 - t[-1]), max(prior_post, 1.0) * self.cfg.max_margin_scale)
        end_cap = max(end_cap, 1.0)
        end_raw = self.end_head(torch.cat([global_feat, key_feats[-1]], dim=0)).squeeze(-1)
        end_margin = self._margin_from_raw(end_raw, prior_post, end_cap)
        boundaries.append(torch.clamp(t[-1] + end_margin, min=float(t[-1]) + 1e-3, max=float(T - 1)))

        boundary_tensor = torch.stack(boundaries)
        durations = torch.clamp(boundary_tensor[1:] - boundary_tensor[:-1], min=1.0)
        rel_pos = torch.clamp((t - boundary_tensor[:-1]) / durations, 0.0, 1.0)
        return {
            "boundaries": boundary_tensor,
            "durations": durations,
            "rel_pos": rel_pos,
            "alphas": torch.stack(alpha_list) if alpha_list else torch.zeros(0, device=device),
            "prior_total_frames": torch.tensor(prior_total, dtype=torch.float32, device=device),
        }

    def _sample_segment(self, signal_tc: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
        T = signal_tc.shape[0]
        signal = signal_tc.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        steps = torch.linspace(0.0, 1.0, self.cfg.target_len, device=signal_tc.device)
        positions = start + steps * torch.clamp(end - start, min=1.0)
        norm_x = 2.0 * positions / max(T - 1, 1) - 1.0
        grid = torch.stack([norm_x, torch.zeros_like(norm_x)], dim=-1).view(1, 1, self.cfg.target_len, 2)
        sampled = F.grid_sample(signal, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled[0, :, 0, :].transpose(0, 1)

    def extract_windows(self, signal_tc: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        windows = []
        for i in range(len(boundaries) - 1):
            windows.append(self._sample_segment(signal_tc, boundaries[i], boundaries[i + 1]))
        return torch.stack(windows, dim=0)

    def forward_episode(self, imu: torch.Tensor, key_frames: torch.Tensor, sample_rate_hz: float) -> dict:
        feats = self.encoder(imu.unsqueeze(0))[0]
        boundary_out = self.predict_boundaries(feats, key_frames, sample_rate_hz)
        windows = self.extract_windows(imu, boundary_out["boundaries"])
        logits = self.classifier(windows)
        return {
            **boundary_out,
            "windows": windows,
            "logits": logits,
        }

    def compute_loss(self, out: dict, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        labels = labels.long().to(out["logits"].device)
        char_loss = F.cross_entropy(out["logits"], labels)
        ratio_loss = F.smooth_l1_loss(out["rel_pos"], torch.full_like(out["rel_pos"], self.cfg.trigger_ratio))
        duration_loss = F.smooth_l1_loss(
            torch.log(out["durations"] + 1.0),
            torch.full_like(out["durations"], torch.log(out["prior_total_frames"] + 1.0)),
        )
        if len(out["alphas"]) > 0:
            alpha_loss = F.smooth_l1_loss(out["alphas"], torch.full_like(out["alphas"], 0.5))
        else:
            alpha_loss = torch.zeros((), device=out["logits"].device)
        total = (
            self.cfg.loss_char * char_loss
            + self.cfg.loss_ratio * ratio_loss
            + self.cfg.loss_duration * duration_loss
            + self.cfg.loss_alpha * alpha_loss
        )
        metrics = {
            "loss": float(total.detach().cpu()),
            "char_loss": float(char_loss.detach().cpu()),
            "ratio_loss": float(ratio_loss.detach().cpu()),
            "duration_loss": float(duration_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
        }
        return total, metrics

    def checkpoint_payload(self) -> dict:
        return {
            "model_state": self.state_dict(),
            "config": self.cfg.__dict__,
            "classifier_meta": self.classifier.meta(),
        }


def build_classifier(
    target_len: int,
    classes: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> SegmentalClassifier:
    return SegmentalClassifier(target_len=target_len, classes=classes, means=means, stds=stds)


def save_classifier_checkpoint(classifier: SegmentalClassifier, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state": classifier.model.state_dict(),
            "n_timesteps": classifier.target_len,
            "n_channels": int(classifier.means.numel()),
            "n_classes": len(classifier.classes),
            "classes": np.asarray(classifier.classes, dtype="<U1"),
            "means": classifier.means.detach().cpu().numpy(),
            "stds": classifier.stds.detach().cpu().numpy(),
            "use_diff_channels": bool(classifier.use_diff_channels),
        },
        path,
    )


def load_classifier_from_checkpoint(path: str, device: torch.device) -> SegmentalClassifier:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    classifier = SegmentalClassifier(
        target_len=int(ckpt["n_timesteps"]),
        classes=[str(x) for x in np.asarray(ckpt["classes"]).tolist()],
        means=np.asarray(ckpt["means"], dtype=np.float32),
        stds=np.asarray(ckpt["stds"], dtype=np.float32),
        use_diff_channels=bool(ckpt.get("use_diff_channels", False)),
    ).to(device)
    classifier.model.load_state_dict(ckpt["model_state"])
    return classifier


def load_external_inception(checkpoint_path: str, scaler_path: str, device: torch.device) -> SegmentalClassifier:
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler = np.load(scaler_path)
    classifier = SegmentalClassifier(
        target_len=int(raw["n_timesteps"]),
        classes=[str(x) for x in np.asarray(raw["classes"]).astype(str).tolist()],
        means=np.asarray(scaler["means"], dtype=np.float32),
        stds=np.asarray(scaler["stds"], dtype=np.float32),
        use_diff_channels=bool(raw.get("use_diff_channels", False)),
    ).to(device)
    classifier.model.load_state_dict(raw["model_state"])
    return classifier


def train_classifier(
    classifier: SegmentalClassifier,
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    device: torch.device,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 8e-4,
    patience: int = 25,
) -> tuple[SegmentalClassifier, dict]:
    X_train = torch.tensor(train_windows, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.tensor(val_windows, dtype=torch.float32, device=device)
    y_val = torch.tensor(val_labels, dtype=torch.long, device=device)

    dl_gen = torch.Generator()
    dl_gen.manual_seed(42)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, generator=dl_gen)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    classifier = classifier.to(device)

    best_val = -1.0
    best_state = None
    patience_ctr = 0
    history = []

    for epoch in range(epochs):
        classifier.train()
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = augment_batch(xb, p=0.5)
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            correct += int((logits.argmax(1) == yb).sum().item())
            total += int(len(yb))
        scheduler.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(X_val)
            val_acc = float((val_logits.argmax(1) == y_val).float().mean().item())

        history.append({
            "epoch": epoch + 1,
            "train_acc": correct / max(total, 1),
            "val_acc": val_acc,
        })
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(classifier.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(f"[classifier] epoch={epoch+1:03d} train_acc={history[-1]['train_acc']:.4f} val_acc={val_acc:.4f}", flush=True)
        if patience_ctr >= patience:
            break

    if best_state is not None:
        classifier.load_state_dict(best_state)
    return classifier, {"best_val_acc": best_val, "history": history}


def load_segmental_checkpoint(path: str, device: torch.device) -> SegmentalSequenceModel:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = SegmentalConfig(**ckpt["config"])
    meta = ckpt["classifier_meta"]
    classifier = build_classifier(
        target_len=int(meta["target_len"]),
        classes=[str(x) for x in meta["classes"]],
        means=np.asarray(meta["means"], dtype=np.float32),
        stds=np.asarray(meta["stds"], dtype=np.float32),
    ).to(device)
    model = SegmentalSequenceModel(cfg, classifier).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model
