"""
Frame-level Character TCN for CTC decoding.

Replaces the dual-head (typing + onset) design with a single character head
that outputs P(char|frame) at every timestep.

Architecture:
  - Same dilated residual TCN backbone as EpisodeTCN
  - Wider (128 hidden) and deeper (12 layers) to handle 38-class output
  - Single output head: [B, 38, T] logits

Backbone weights can be initialized from an existing onset TCN checkpoint
to transfer "where are keystrokes" knowledge.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResBlock(nn.Module):
    """Same architecture as stage2_episode TCN block."""
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


class FrameCTCModel(nn.Module):
    """
    Frame-level character posterior model.

    Input:  [B, C_in, T] preprocessed IMU (8ch: 6 raw + 2 magnitude)
    Output: [B, num_classes, T] logits (38 classes: blank + a-z + 0-9 + unk)

    Use with CTC loss (on log-softmax of output) and/or frame-level CE.
    """

    def __init__(self, in_ch=8, hidden=128, num_layers=12, kernel=3,
                 dropout=0.25, num_classes=38):
        super().__init__()
        self.num_classes = num_classes

        self.input_conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        self.layers = nn.ModuleList([
            DilatedResBlock(hidden, kernel, dilation=2 ** (i % num_layers),
                            dropout=dropout)
            for i in range(num_layers)
        ])

        # Character output head: slightly deeper than a single conv
        # to give enough capacity for 38-way discrimination
        self.char_head = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, num_classes, 1),
        )

    def forward(self, x):
        """
        x: [B, C, T]
        Returns: logits [B, num_classes, T]
        """
        h = self.input_conv(x)
        for layer in self.layers:
            h = layer(h)
        return self.char_head(h)

    def log_probs(self, x):
        """Returns log-softmax output for CTC decoding."""
        return F.log_softmax(self.forward(x), dim=1)

    def predict(self, x):
        """Returns greedy per-frame predictions [B, T] and probs [B, C, T]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        return preds, probs


def init_from_onset_checkpoint(model: FrameCTCModel, onset_ckpt_path: str,
                               device: torch.device) -> FrameCTCModel:
    """
    Transfer backbone weights from an existing onset TCN checkpoint.

    The onset model already learned "where are keystrokes" — this is
    excellent initialization for "what character is at each frame".

    Handles both old single-head and new dual-head checkpoints.
    Skips layers with mismatched shapes (e.g. if hidden dims differ).
    """
    # Older stage2_episode checkpoints pickle config dataclasses under the
    # module path `configs.config.*`. When stage2_ctc is on sys.path, that path
    # resolves to THIS package instead, so we provide lightweight placeholders
    # to let torch unpickle and then immediately discard the config objects.
    import configs.config as current_cfg
    for _name in ('SignalConfig', 'SynthesisConfig', 'ModelConfig',
                  'TrainConfig', 'EpisodeConfig'):
        if not hasattr(current_cfg, _name):
            setattr(current_cfg, _name, type(_name, (), {}))

    ckpt = torch.load(onset_ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get('model', ckpt.get('model_state_dict', {}))

    # Remap old key names
    remapped = {}
    for k, v in state.items():
        if k.startswith('output_conv.'):
            continue  # skip old typing head
        if k.startswith('typing_head.') or k.startswith('onset_head.'):
            continue  # skip both heads from dual-head model
        remapped[k] = v

    new_state = model.state_dict()
    transferred = 0
    for name, param in remapped.items():
        if name in new_state and new_state[name].shape == param.shape:
            new_state[name] = param.clone()
            transferred += 1

    model.load_state_dict(new_state)
    total_backbone = sum(1 for k in new_state if 'char_head' not in k)
    print(f"[FrameCTCModel] Transferred {transferred}/{total_backbone} "
          f"backbone params from onset checkpoint")
    return model
