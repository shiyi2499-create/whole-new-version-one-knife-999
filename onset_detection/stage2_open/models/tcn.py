"""
Frame-wise 3-class TCN for open Stage 2.
Classes: 0=gap, 1=keystroke, 2=separator.
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


class OpenTCN(nn.Module):
    """
    Single-stage TCN with N dilated layers → 3-class frame output.
    No assumption about number of groups or onsets.
    """

    def __init__(self, in_ch=8, hidden=64, num_layers=10, kernel=3,
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([
            DilatedResBlock(hidden, kernel, dilation=2**i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_conv = nn.Conv1d(hidden, num_classes, 1)

    def forward(self, x):
        """x: [B, C, T] → logits [B, num_classes, T]"""
        h = self.input_conv(x)
        for layer in self.layers:
            h = layer(h)
        return self.output_conv(h)

    def predict(self, x):
        """Returns class predictions [B, T] and probabilities [B, 3, T]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        return preds, probs
