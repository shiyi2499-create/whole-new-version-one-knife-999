"""
Dual-head TCN for episode-based Stage 2.

Head A (typing_head): frame-wise 2-class (silence vs typing).
  - Same as before: learns "is there keystroke activity here?"
  - Used by decoder to find episode boundaries.

Head B (onset_head): frame-wise scalar sigmoid.
  - NEW: learns a Gaussian-smoothed impulse at each key center.
  - Trained with MSE against a soft Gaussian target (sigma ~ 15-25ms).
  - Used by decoder to pick onset peaks INSIDE an episode.

Why two heads instead of one?
  The "typing" plateau signal and the "per-key impulse" signal are fundamentally
  different targets. A single head trying to do both will average them out.
  Decoupling the heads:
    - typing_head: kept smooth by TMSE loss → good for episode segmentation
    - onset_head: kept sharp by MSE vs narrow Gaussian → peak-pickable

  This is equivalent to separating "activity detection" from "event localization",
  which is standard practice in action detection (e.g. STAD, BMN, AFSD).

Old single-head checkpoints load transparently (onset_head weights absent →
decoder falls back to energy-envelope heuristic automatically).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1, dropout=0.3):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.proj = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(ch)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = self.drop(self.proj(out))
        return self.norm(out + x)


class EpisodeTCN(nn.Module):
    """
    Dual-head TCN:
      - typing_head  → [B, 2, T] logits (silence vs typing)
      - onset_head   → [B, 1, T] sigmoid (per-key Gaussian impulse)

    The onset_head is optional at inference time — if the checkpoint was
    trained without it (old single-head checkpoint), onset decoding falls
    back to the energy envelope heuristic transparently.
    """

    def __init__(self, in_ch=8, hidden=64, num_layers=10, kernel=3,
                 dropout=0.3, num_classes=2, use_onset_head=True):
        super().__init__()
        self.use_onset_head = use_onset_head

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([
            DilatedResBlock(hidden, kernel, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])

        # Head A: typing vs silence (unchanged from single-head version)
        self.typing_head = nn.Conv1d(hidden, num_classes, 1)

        # Head B: per-key onset probability (NEW)
        # A small 2-layer bottleneck prevents the onset gradient from
        # corrupting shared features needed for the typing head.
        if use_onset_head:
            self.onset_head = nn.Sequential(
                nn.Conv1d(hidden, hidden // 2, 1),
                nn.ReLU(),
                nn.Conv1d(hidden // 2, 1, 1),
            )
        else:
            self.onset_head = None

    def forward(self, x):
        """
        x: [B, C, T]
        Returns:
          typing_logits: [B, 2, T]
          onset_logits:  [B, 1, T] or None
        """
        h = self.input_conv(x)
        for layer in self.layers:
            h = layer(h)

        typing_logits = self.typing_head(h)
        onset_logits = self.onset_head(h) if self.use_onset_head else None
        return typing_logits, onset_logits

    def predict(self, x):
        """Returns typing class predictions [B, T] and typing probs [B, 2, T]."""
        self.eval()
        with torch.no_grad():
            typing_logits, _ = self.forward(x)
            probs = F.softmax(typing_logits, dim=1)
            preds = typing_logits.argmax(dim=1)
        return preds, probs

    def predict_typing_prob(self, x):
        """Returns just the typing probability [B, T]."""
        self.eval()
        with torch.no_grad():
            typing_logits, _ = self.forward(x)
            probs = F.softmax(typing_logits, dim=1)
        return probs[:, 1, :]  # class 1 = typing

    def predict_onset_prob(self, x):
        """
        Returns per-frame onset probability [B, T] from the onset head.
        Returns None if the onset head is disabled (old checkpoint fallback).
        """
        self.eval()
        with torch.no_grad():
            _, onset_logits = self.forward(x)
            if onset_logits is None:
                return None
            return torch.sigmoid(onset_logits).squeeze(1)

    def has_onset_head(self):
        return self.use_onset_head and self.onset_head is not None


# ---------------------------------------------------------------------------
# Backward-compatible loader — old checkpoints only have typing_head weights.
# load_state_dict(strict=False) silently skips missing onset_head weights.
# ---------------------------------------------------------------------------

def load_episode_tcn(ckpt_path, device, use_onset_head=True):
    """
    Load EpisodeTCN from checkpoint, handling both old (single-head) and
    new (dual-head) checkpoints transparently.
    """
    from configs.config import ModelConfig
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mcfg = ckpt.get('model_cfg')
    if mcfg is None:
        mcfg = ModelConfig()
    elif isinstance(mcfg, dict):
        mcfg = ModelConfig(**mcfg)

    state = ckpt.get('model', ckpt.get('model_state_dict', {}))
    # Remap old single-head key "output_conv.*" → "typing_head.*"
    remapped = {}
    for k, v in state.items():
        if k.startswith('output_conv.'):
            remapped['typing_head.' + k[len('output_conv.'):]] = v
        else:
            remapped[k] = v

    has_onset_weights = any('onset_head' in k for k in remapped.keys())
    _use_onset_head = use_onset_head and has_onset_weights

    model = EpisodeTCN(
        in_ch=mcfg.input_channels,
        hidden=mcfg.hidden_channels,
        num_layers=mcfg.num_layers,
        kernel=mcfg.kernel_size,
        dropout=mcfg.dropout,
        num_classes=mcfg.num_classes,
        use_onset_head=_use_onset_head,
    ).to(device)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    non_onset_missing = [k for k in missing if 'onset_head' not in k]
    if non_onset_missing:
        print(f"[EpisodeTCN] Unexpected missing keys: {non_onset_missing}")
    model.eval()
    return model, mcfg
