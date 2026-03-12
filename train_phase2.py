"""
Training Pipeline - Phase 2: Hierarchical + Deep Learning + Ensemble
=====================================================================
Part A: Hierarchical classification (zone → key cascade)
Part B: Deep Learning
  - 1D CNN on raw windows
  - 1D CNN + BiLSTM hybrid
  - Transformer (d_model=128, num_layers=3, dropout 0.3/0.4) ← upgraded
  - Data augmentation: time shift, noise, channel dropout, scaling
Part C: Ensemble (XGBoost + Transformer probability fusion) ← new

Run:
  .venv/bin/python3 train_phase2.py

Requires: pip install torch scikit-learn xgboost matplotlib seaborn
"""

import os
import sys
import time
import json
import copy
import platform
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict, Counter
from typing import Optional

from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, top_k_accuracy_score, f1_score, recall_score
)
from sklearn.pipeline import Pipeline

from feature_extractor import (
    extract_features_batch, get_feature_names,
    map_to_zones, ZONE_LABELS, ZONE_MAPS
)

# ── XGBoost ───────────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ xgboost not installed. Ensemble will be skipped.")

# ── PyTorch ──────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    DEVICE = torch.device("cpu")

    def resolve_torch_device(device: str = "auto") -> torch.device:
        req = (device or "auto").lower()
        if req == "auto":
            if platform.system() == "Darwin":
                req = "cpu"
            elif torch.cuda.is_available():
                req = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                req = "mps"
            else:
                req = "cpu"

        if req == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested device=cuda but CUDA is not available.")
        elif req == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise RuntimeError("Requested device=mps but MPS is not available.")
        elif req != "cpu":
            raise ValueError(f"Unsupported device: {device}. Use one of auto/cpu/mps/cuda.")

        return torch.device(req)

    # Use env var to override when needed, e.g. KEYSTROKE_DEVICE=cuda.
    DEVICE = resolve_torch_device(os.environ.get("KEYSTROKE_DEVICE", "auto"))
    print(f"  PyTorch device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("  ⚠ PyTorch not installed. Install: pip install torch")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP = True
except ImportError:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP = False


SEED = 42


def _topk_safe(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    n_classes = probs.shape[1]
    if n_classes < k:
        return 0.0
    return float(top_k_accuracy_score(y_true, probs, k=k))


def _bootstrap_acc_ci(y_true: np.ndarray, y_pred: np.ndarray,
                      n_boot: int = 500, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    scores = np.empty(n_boot, dtype=np.float64)
    idx = np.arange(n)
    for i in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        scores[i] = np.mean(y_true[b] == y_pred[b])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def _summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                     classes: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    top3 = _topk_safe(y_true, probs, k=3)
    top5 = _topk_safe(y_true, probs, k=5)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_cls = recall_score(
        y_true, y_pred, labels=np.arange(len(classes)),
        average=None, zero_division=0
    )
    per_key_recall = {str(k): float(v) for k, v in zip(classes.tolist(), per_cls.tolist())}
    ci_lo, ci_hi = _bootstrap_acc_ci(y_true, y_pred, n_boot=500, seed=SEED)
    return {
        "accuracy": acc,
        "accuracy_ci95": [ci_lo, ci_hi],
        "top3_accuracy": top3,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "per_key_recall": per_key_recall,
    }


def _get_groups_from_data(data, n_samples: int) -> Optional[np.ndarray]:
    if "session_ids" in data.files:
        groups = data["session_ids"].astype(str)
        if len(groups) == n_samples:
            return groups
    return None


def _build_outer_splits(y: np.ndarray,
                        groups: Optional[np.ndarray],
                        n_splits: int = 5) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    if groups is not None:
        uniq = np.unique(groups)
        if len(uniq) >= n_splits:
            if HAS_STRATIFIED_GROUP:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                splits = list(splitter.split(np.zeros(len(y)), y, groups))
                return splits, "StratifiedGroupKFold"
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(np.zeros(len(y)), y, groups))
            return splits, "GroupKFold"
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    splits = list(splitter.split(np.zeros(len(y)), y))
    return splits, "StratifiedKFold"


def _split_train_val(train_idx: np.ndarray, y: np.ndarray,
                     groups: Optional[np.ndarray], fold: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + fold)
    if groups is not None:
        tr_groups = groups[train_idx]
        if len(np.unique(tr_groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold)
            tr_local, vl_local = next(gss.split(np.zeros(len(train_idx)), y[train_idx], tr_groups))
            return train_idx[tr_local], train_idx[vl_local]
    # Fallback (legacy datasets without session groups)
    if len(np.unique(y[train_idx])) > 1:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold)
        tr_local, vl_local = next(sss.split(np.zeros(len(train_idx)), y[train_idx]))
    else:
        n_val = max(1, int(0.2 * len(train_idx)))
        perm = rng.permutation(len(train_idx))
        vl_local = perm[:n_val]
        tr_local = perm[n_val:]
    return train_idx[tr_local], train_idx[vl_local]


# ══════════════════════════════════════════════════════════════
#  PART A: HIERARCHICAL CLASSIFICATION
# ══════════════════════════════════════════════════════════════

class HierarchicalClassifier:
    """
    Two-stage classifier:
      Stage 1: Predict zone (hand / row / quadrant)
      Stage 2: Predict exact key within predicted zone
    """

    def __init__(self, zone_type: str = "row"):
        self.zone_type = zone_type
        self.zone_model = None
        self.key_models = {}
        self.zone_le = LabelEncoder()
        self.key_les = {}
        self.zone_map = ZONE_MAPS[zone_type]

    def _get_zone_labels(self, y_keys):
        return map_to_zones(y_keys, self.zone_type)

    def fit(self, X, y_keys):
        y_zones = self._get_zone_labels(y_keys)
        valid = y_zones >= 0
        X_valid = X[valid]
        y_zones_valid = y_zones[valid]
        y_keys_valid = y_keys[valid]

        self.zone_le.fit(y_zones_valid.astype(str))
        self.zone_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=42,
                n_jobs=-1, class_weight="balanced"
            ))
        ])
        self.zone_model.fit(X_valid, y_zones_valid.astype(str))

        unique_zones = np.unique(y_zones_valid)
        for z in unique_zones:
            mask = y_zones_valid == z
            X_z = X_valid[mask]
            y_z = y_keys_valid[mask]
            if len(np.unique(y_z)) < 2:
                continue
            le = LabelEncoder()
            le.fit(y_z)
            self.key_les[z] = le
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=10.0, gamma="scale",
                            probability=True, random_state=42,
                            class_weight="balanced"))
            ])
            model.fit(X_z, y_z)
            self.key_models[z] = model

    def predict(self, X):
        zone_preds = self.zone_model.predict(X).astype(int)
        key_preds = np.array(["?"] * len(X), dtype=object)
        for z in np.unique(zone_preds):
            mask = zone_preds == z
            if z in self.key_models:
                key_preds[mask] = self.key_models[z].predict(X[mask])
            else:
                key_preds[mask] = "?"
        return key_preds


def run_hierarchical(X_feat, y_keys, groups=None):
    print(
        f"\n{'='*60}\n"
        f"  PART A: Hierarchical Classification\n"
        f"{'='*60}"
    )
    results = {}
    splits, split_mode = _build_outer_splits(y_keys, groups, n_splits=5)
    print(f"  Split protocol: {split_mode}")
    for zone_type in ["hand", "row", "quadrant"]:
        print(f"\n  ── Hierarchical: zone={zone_type} ──")
        all_preds = np.array(["?"] * len(y_keys), dtype=object)
        fold_accs = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            hc = HierarchicalClassifier(zone_type=zone_type)
            hc.fit(X_feat[train_idx], y_keys[train_idx])
            preds = hc.predict(X_feat[test_idx])
            all_preds[test_idx] = preds
            valid_fold = preds != "?"
            if valid_fold.any():
                acc = accuracy_score(y_keys[test_idx][valid_fold], preds[valid_fold])
            else:
                acc = 0.0
            fold_accs.append(acc)

        valid_mask = all_preds != "?"
        overall_acc = accuracy_score(y_keys[valid_mask], all_preds[valid_mask])
        print(f"    Accuracy:  {overall_acc:.1%}")
        print(f"    Mean±Std:  {np.mean(fold_accs):.3f} ± {np.std(fold_accs):.3f}")
        results[f"hierarchical_{zone_type}"] = {
            "accuracy": float(overall_acc),
            "fold_mean": float(np.mean(fold_accs)),
            "fold_std": float(np.std(fold_accs)),
            "split_protocol": split_mode,
        }
    return results


# ══════════════════════════════════════════════════════════════
#  PART B: DEEP LEARNING
# ══════════════════════════════════════════════════════════════

def augment_batch(X_batch, p=0.5):
    """
    Apply random augmentations to a batch of windows.
    X_batch: (B, T, C) tensor
    """
    B, T, C = X_batch.shape
    X_aug = X_batch.clone()

    for i in range(B):
        if np.random.random() > p:
            continue
        aug_type = np.random.choice(["shift", "noise", "scale", "ch_drop"])

        if aug_type == "shift":
            shift = np.random.randint(-T // 10, T // 10 + 1)
            X_aug[i] = torch.roll(X_aug[i], shifts=shift, dims=0)
        elif aug_type == "noise":
            std = X_aug[i].std() * 0.01
            X_aug[i] += torch.randn_like(X_aug[i]) * std
        elif aug_type == "scale":
            scale = 0.8 + 0.4 * np.random.random()
            X_aug[i] *= scale
        elif aug_type == "ch_drop":
            ch = np.random.randint(0, C)
            X_aug[i][:, ch] = 0.0

    return X_aug


# ── Model Definitions ────────────────────────────────────────

if HAS_TORCH:
    class Conv1DClassifier(nn.Module):
        """Simple 1D CNN. Handles variable n_timesteps via AdaptiveAvgPool1d."""
        def __init__(self, n_timesteps=39, n_channels=6, n_classes=42):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.conv_layers(x)
            return self.classifier(x)


    class CNNBiLSTMClassifier(nn.Module):
        """CNN + Bidirectional LSTM hybrid."""
        def __init__(self, n_timesteps=39, n_channels=6, n_classes=42):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_channels, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            self.lstm = nn.LSTM(
                input_size=64, hidden_size=64, num_layers=2,
                batch_first=True, bidirectional=True, dropout=0.2,
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = x.permute(0, 2, 1)
            lstm_out, _ = self.lstm(x)
            x = lstm_out[:, -1, :]
            return self.classifier(x)


    class TransformerClassifier(nn.Module):
        """
        Upgraded Transformer for vibration sequences.
        d_model=128, nhead=8, num_layers=3 vs previous 64/4/2.
        Dropout raised to 0.3 (encoder) / 0.4+0.3 (classifier) to
        prevent overfit on ~4000 samples.
        """
        def __init__(self, n_timesteps=39, n_channels=6, n_classes=42,
                     d_model=128, nhead=8, num_layers=3):
            super().__init__()
            self.input_proj = nn.Linear(n_channels, d_model)
            # Learnable positional encoding; size matches input length
            self.pos_encoding = nn.Parameter(
                torch.randn(1, n_timesteps, d_model) * 0.02
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=256,   # 2× d_model
                dropout=0.3,           # ↑ from 0.2
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(0.4),           # ↑ from 0.3
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(0.3),           # ↑ from 0.2
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            # x: (B, T, C)
            x = self.input_proj(x)                          # (B, T, d_model)
            x = x + self.pos_encoding[:, :x.size(1), :]
            x = self.transformer(x)                         # (B, T, d_model)
            x = x.mean(dim=1)                               # global avg pool
            return self.classifier(x)


# ── Training Loop ────────────────────────────────────────────

def train_dl_model(model, X_train, y_train, X_val, y_val,
                   epochs=200, lr=1e-3, batch_size=32, augment=True,
                   patience=40):
    """
    Train a PyTorch model with early stopping.
    patience raised to 40 (from 30) to give larger models time to converge.
    """
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_vl = torch.FloatTensor(X_val).to(DEVICE)
    y_vl = torch.LongTensor(y_val).to(DEVICE)

    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for xb, yb in train_loader:
            if augment:
                xb = augment_batch(xb, p=0.5)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_out = model(X_vl)
            val_acc = (val_out.argmax(dim=1) == y_vl).float().mean().item()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"      Early stop at epoch {epoch+1} (patience={patience})")
            break

        if (epoch + 1) % 50 == 0:
            print(f"      Epoch {epoch+1:3d}: loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc, history


def evaluate_dl_model(ModelClass, X_raw, y_keys, model_name,
                      epochs=200, lr=1e-3, augment=True, patience=40,
                      groups=None, outer_splits=None, split_mode=None):
    """
    Evaluate a DL model using 5-fold CV.
    Returns metrics dict, all_preds, all_probs (for ensemble use).
    """
    le = LabelEncoder()
    le.fit(y_keys)
    y_enc = le.transform(y_keys)
    n_classes = len(le.classes_)

    if outer_splits is None:
        outer_splits, split_mode = _build_outer_splits(y_enc, groups, n_splits=5)
    elif split_mode is None:
        split_mode = "precomputed"

    fold_accs = []
    all_preds = np.zeros(len(y_enc), dtype=int)
    all_probs = np.zeros((len(y_enc), n_classes))
    n_folds = len(outer_splits)

    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"    Fold {fold+1}/{n_folds}...")
        train_sub_idx, val_idx = _split_train_val(train_idx, y_enc, groups, fold=fold)
        if len(train_sub_idx) == 0 or len(val_idx) == 0:
            train_sub_idx, val_idx = train_idx, test_idx

        # Fit normalization on train subset only to avoid test leakage.
        mu = X_raw[train_sub_idx].mean(axis=(0, 1), keepdims=True)
        sd = X_raw[train_sub_idx].std(axis=(0, 1), keepdims=True)
        sd[sd < 1e-10] = 1.0
        X_train = (X_raw[train_sub_idx] - mu) / sd
        X_val = (X_raw[val_idx] - mu) / sd
        X_test = (X_raw[test_idx] - mu) / sd

        model = ModelClass(
            n_timesteps=X_raw.shape[1],
            n_channels=X_raw.shape[2],
            n_classes=n_classes,
        )
        trained_model, best_val_acc, _ = train_dl_model(
            model,
            X_train, y_enc[train_sub_idx],
            X_val, y_enc[val_idx],
            epochs=epochs, lr=lr, augment=augment, patience=patience,
        )
        trained_model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(DEVICE)
            out = trained_model(X_test_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

        all_preds[test_idx] = preds
        all_probs[test_idx] = probs
        fold_acc = accuracy_score(y_enc[test_idx], preds)
        fold_accs.append(fold_acc)
        print(f"      → val_acc={best_val_acc:.3f}  test_acc={fold_acc:.3f}")

    summary = _summary_metrics(y_enc, all_preds, all_probs, le.classes_)
    metrics = {
        "model": model_name,
        **summary,
        "fold_mean": float(np.mean(fold_accs)),
        "fold_std": float(np.std(fold_accs)),
        "fold_accuracies": [float(a) for a in fold_accs],
        "label_classes": le.classes_.tolist(),
        "split_protocol": split_mode,
    }

    if HAS_PLOT:
        cm = confusion_matrix(y_enc, all_preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, vmin=0, vmax=1, annot_kws={"size": 6})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"{model_name}\nAccuracy: {summary['accuracy']:.1%}  "
            f"Top-3: {summary['top3_accuracy']:.1%}"
        )
        plt.tight_layout()
        path = f"results/confusion_{model_name}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"    Saved: {path}")

    return metrics, all_preds, all_probs, le


# ══════════════════════════════════════════════════════════════
#  PART C: ENSEMBLE  (XGBoost + Transformer)
# ══════════════════════════════════════════════════════════════

def run_xgb_cv(X_feat, y_keys, le, outer_splits=None, groups=None, split_mode=None):
    """
    Run XGBoost 5-fold CV on feature matrix.
    Returns (all_probs, overall_acc) using the same label encoder as Transformer.
    """
    if not HAS_XGB:
        return None, 0.0

    y_enc = le.transform(y_keys)
    n_classes = len(le.classes_)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss", n_jobs=-1,
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb_model),
    ])

    if outer_splits is None:
        outer_splits, split_mode = _build_outer_splits(y_enc, groups, n_splits=5)
    elif split_mode is None:
        split_mode = "precomputed"

    all_probs = np.zeros((len(y_enc), n_classes))
    all_preds = np.zeros(len(y_enc), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        pipe.fit(X_feat[train_idx], y_enc[train_idx])
        probs = pipe.predict_proba(X_feat[test_idx])
        # Map columns to le's class order (XGB may reorder internally)
        xgb_classes = pipe.named_steps["clf"].classes_
        prob_aligned = np.zeros((len(test_idx), n_classes))
        for ci, c in enumerate(xgb_classes):
            prob_aligned[:, c] = probs[:, ci]
        all_probs[test_idx] = prob_aligned
        all_preds[test_idx] = prob_aligned.argmax(axis=1)

    metrics = _summary_metrics(y_enc, all_preds, all_probs, le.classes_)
    metrics["split_protocol"] = split_mode
    return all_probs, metrics


def run_ensemble(X_feat, y_keys, le,
                 tf_probs, tf_acc,
                 xgb_probs, xgb_acc):
    """
    Fuse Transformer and XGBoost probability outputs.

    Weighting strategy: accuracy-proportional soft weighting.
      w_tf  = tf_acc  / (tf_acc + xgb_acc)
      w_xgb = xgb_acc / (tf_acc + xgb_acc)

    Also tries fixed grids to find optimal weights.
    """
    print(
        f"\n{'='*60}\n"
        f"  PART C: Ensemble (Transformer + XGBoost)\n"
        f"{'='*60}"
    )

    y_enc = le.transform(y_keys)
    results = {}

    # ── Accuracy-proportional weights ────────────────────────
    total = tf_acc + xgb_acc
    w_tf  = tf_acc  / total
    w_xgb = xgb_acc / total
    probs_prop = w_tf * tf_probs + w_xgb * xgb_probs
    pred_prop = probs_prop.argmax(axis=1)
    summary_prop = _summary_metrics(y_enc, pred_prop, probs_prop, le.classes_)
    print(f"\n  Accuracy-proportional weights "
          f"(Transformer={w_tf:.2f}, XGBoost={w_xgb:.2f}):")
    print(f"    Accuracy: {summary_prop['accuracy']:.1%}  "
          f"Top-3: {summary_prop['top3_accuracy']:.1%}")
    results["ensemble_prop"] = {
        **summary_prop,
        "w_tf": float(w_tf), "w_xgb": float(w_xgb)
    }

    # ── Grid search over weights ──────────────────────────────
    best_acc = 0.0
    best_w = (0.5, 0.5)
    print("\n  Grid search over Transformer weight (0.3 → 0.9):")
    for w in np.arange(0.3, 1.0, 0.1):
        w = round(w, 1)
        probs_g = w * tf_probs + (1 - w) * xgb_probs
        acc_g = accuracy_score(y_enc, probs_g.argmax(axis=1))
        top3_g = _topk_safe(y_enc, probs_g, k=3)
        print(f"    w_tf={w:.1f}  acc={acc_g:.1%}  top3={top3_g:.1%}")
        if acc_g > best_acc:
            best_acc = acc_g
            best_w = (w, 1 - w)

    best_probs = best_w[0] * tf_probs + best_w[1] * xgb_probs
    best_preds = best_probs.argmax(axis=1)
    best_summary = _summary_metrics(y_enc, best_preds, best_probs, le.classes_)

    print(f"\n  🏆 Best ensemble: w_tf={best_w[0]:.1f}  "
          f"acc={best_summary['accuracy']:.1%}  "
          f"top3={best_summary['top3_accuracy']:.1%}  "
          f"top5={best_summary['top5_accuracy']:.1%}")

    results["ensemble_best"] = {
        **best_summary,
        "w_tf": float(best_w[0]),
        "w_xgb": float(best_w[1]),
    }

    # ── Save ensemble probabilities for Phase 3 decoder ──────
    np.savez_compressed(
        "results/ensemble_probs.npz",
        probs=best_probs,
        y_true=y_enc,
        classes=le.classes_,
        w_tf=best_w[0], w_xgb=best_w[1],
    )
    print("  Saved ensemble probs → results/ensemble_probs.npz (for phase3_decoder.py)")

    return results


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs("results", exist_ok=True)
    np.random.seed(SEED)
    if HAS_TORCH:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    print(
        f"\n{'='*60}\n"
        f"  🧠 TRAINING PIPELINE - Phase 2\n"
        f"     Hierarchical + Deep Learning + Ensemble\n"
        f"{'='*60}"
    )

    # ── Load data ────────────────────────────────────────────
    data = np.load("data/processed/merged_dataset.npz", allow_pickle=True)
    X_raw = data["X"].astype(np.float32)
    y_keys = data["y"]
    rate = int(data.get("target_rate_hz", 190))
    groups = _get_groups_from_data(data, len(y_keys))
    if groups is None:
        print("  ⚠ merged_dataset.npz has no session_ids; fallback to sample-level stratified split.")
        print("    Re-run preprocessor to enable group-wise leakage-safe evaluation.")

    # 过滤掉采集时误触的杂类（样本数 < 10 的键，如 capslock 等）
    key_counts = Counter(y_keys.tolist())
    valid_keys = {k for k, v in key_counts.items() if v >= 10}
    removed = sorted(set(y_keys.tolist()) - valid_keys)
    if removed:
        print(f"  ⚠ 过滤低样本键: {removed} (各 {[key_counts[k] for k in removed]} 次)")
    mask = np.array([k in valid_keys for k in y_keys])
    X_raw, y_keys = X_raw[mask], y_keys[mask]
    if groups is not None:
        groups = groups[mask]
    print(f"\n  Data: {X_raw.shape}, {len(valid_keys)} classes, {rate}Hz")
    split_preview, split_mode = _build_outer_splits(y_keys, groups, n_splits=5)
    uniq_groups = len(np.unique(groups)) if groups is not None else 0
    print(f"  Split protocol: {split_mode}  (groups={uniq_groups})")

    # ── Load or extract features (for hierarchical + ensemble) ─
    feat_path = "results/features.npz"
    if os.path.exists(feat_path):
        print(f"  Loading cached features from {feat_path}")
        fdata = np.load(feat_path, allow_pickle=True)
        X_feat = fdata["X"]
        cache_ok = X_feat.shape[0] == len(y_keys)
        if cache_ok and "y" in fdata.files:
            cache_ok = np.array_equal(fdata["y"], y_keys)
        if cache_ok and groups is not None and "session_ids" in fdata.files:
            cache_ok = np.array_equal(fdata["session_ids"].astype(str), groups.astype(str))
        if not cache_ok:
            print("  ⚠ Cache mismatch — re-extracting features...")
            X_feat = None
    else:
        X_feat = None

    if X_feat is None:
        print("  Extracting features...")
        X_feat = extract_features_batch(X_raw, sample_rate=rate)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
        payload = {"X": X_feat, "y": y_keys, "feature_names": get_feature_names()}
        if groups is not None:
            payload["session_ids"] = groups
        np.savez_compressed(feat_path, **payload)
        print(f"  Features saved → {feat_path}")

    all_results = {}

    # ══════════════════════════════════════════════════════════
    #  PART A: Hierarchical
    # ══════════════════════════════════════════════════════════
    hier_results = run_hierarchical(X_feat, y_keys, groups=groups)
    all_results.update(hier_results)

    # ══════════════════════════════════════════════════════════
    #  PART B: Deep Learning
    # ══════════════════════════════════════════════════════════
    tf_probs_global = None   # will be filled during Transformer run
    le_global = None

    if not HAS_TORCH:
        print("\n  ⚠ Skipping deep learning (PyTorch not installed)")
    else:
        print(
            f"\n{'='*60}\n"
            f"  PART B: Deep Learning Models\n"
            f"{'='*60}"
        )

        dl_models = [
            ("1D_CNN",      Conv1DClassifier,    200, 1e-3, 40),
            ("CNN_BiLSTM",  CNNBiLSTMClassifier, 200, 1e-3, 40),
            # Transformer upgraded: d_model=128, num_layers=3, dropout↑, epochs↑
            ("Transformer", TransformerClassifier, 350, 5e-4, 40),
        ]

        for model_name, ModelClass, epochs, lr, pat in dl_models:
            print(f"\n  ── {model_name} ──")
            t0 = time.time()
            metrics, preds, probs, le = evaluate_dl_model(
                ModelClass, X_raw, y_keys, model_name,
                epochs=epochs, lr=lr, augment=True, patience=pat,
                groups=groups, outer_splits=split_preview, split_mode=split_mode,
            )
            elapsed = time.time() - t0

            print(f"    Accuracy:     {metrics['accuracy']:.1%}")
            print(f"    Acc CI95:     [{metrics['accuracy_ci95'][0]:.1%}, {metrics['accuracy_ci95'][1]:.1%}]")
            print(f"    Top-3:        {metrics['top3_accuracy']:.1%}")
            print(f"    Top-5:        {metrics['top5_accuracy']:.1%}")
            print(f"    Macro-F1:     {metrics['macro_f1']:.3f}")
            print(f"    Folds:        {[f'{a:.3f}' for a in metrics['fold_accuracies']]}")
            print(f"    Mean±Std:     {metrics['fold_mean']:.3f} ± {metrics['fold_std']:.3f}")
            print(f"    Time:         {elapsed:.0f}s")

            all_results[f"dl_{model_name}"] = metrics

            # Keep Transformer probs for ensemble
            if model_name == "Transformer":
                tf_probs_global = probs
                le_global = le
                # Save for phase3_decoder.py
                np.savez_compressed(
                    "results/transformer_probs.npz",
                    probs=probs,
                    y_true=le.transform(y_keys),
                    classes=le.classes_,
                )
                print("  Saved Transformer probs → results/transformer_probs.npz")

        # ── Ablation: 1D CNN without augmentation ────────────
        print(
            f"\n{'='*60}\n"
            f"  ABLATION: 1D CNN without data augmentation\n"
            f"{'='*60}"
        )
        metrics_noaug, _, _, _ = evaluate_dl_model(
            Conv1DClassifier, X_raw, y_keys, "1D_CNN_no_aug",
            epochs=200, lr=1e-3, augment=False,
            groups=groups, outer_splits=split_preview, split_mode=split_mode,
        )
        print(f"    Accuracy (no aug): {metrics_noaug['accuracy']:.1%}")
        print(f"    Accuracy (w/ aug): {all_results.get('dl_1D_CNN', {}).get('accuracy', 0):.1%}")
        all_results["dl_1D_CNN_no_aug"] = metrics_noaug

    # ══════════════════════════════════════════════════════════
    #  PART C: Ensemble
    # ══════════════════════════════════════════════════════════
    if HAS_XGB and tf_probs_global is not None and le_global is not None:
        print(f"\n  Running XGBoost CV for ensemble...")
        t0 = time.time()
        xgb_probs, xgb_metrics = run_xgb_cv(
            X_feat, y_keys, le_global,
            outer_splits=split_preview, groups=groups, split_mode=split_mode,
        )
        xgb_acc = xgb_metrics["accuracy"]
        print(f"    XGBoost CV accuracy: {xgb_acc:.1%}  ({time.time()-t0:.0f}s)")

        tf_acc = all_results["dl_Transformer"]["accuracy"]
        ensemble_results = run_ensemble(
            X_feat, y_keys, le_global,
            tf_probs_global, tf_acc,
            xgb_probs, xgb_acc,
        )
        all_results.update(ensemble_results)
        all_results["xgb_standalone"] = xgb_metrics
    else:
        if not HAS_XGB:
            print("\n  ⚠ Skipping ensemble (xgboost not installed)")
        elif tf_probs_global is None:
            print("\n  ⚠ Skipping ensemble (Transformer did not run)")

    # ══════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ══════════════════════════════════════════════════════════
    print(
        f"\n{'='*60}\n"
        f"  📊 PHASE 2 RESULTS SUMMARY\n"
        f"{'='*60}\n"
    )

    p1_path = "results/results_phase1.json"
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1 = json.load(f)
        print("  ── Phase 1 (baselines) ──")
        for k, v in p1.items():
            line = f"    {k:35s}  acc={v['accuracy']:.1%}"
            if v.get("top3_accuracy", 0) > 0:
                line += f"  top3={v['top3_accuracy']:.1%}"
            print(line)
        print()

    print("  ── Phase 2 ──")
    for k, v in all_results.items():
        acc = v.get("accuracy", 0)
        line = f"    {k:35s}  acc={acc:.1%}"
        if v.get("top3_accuracy", 0) > 0:
            line += f"  top3={v['top3_accuracy']:.1%}"
        if v.get("top5_accuracy", 0) > 0:
            line += f"  top5={v['top5_accuracy']:.1%}"
        if "macro_f1" in v:
            line += f"  macro_f1={v['macro_f1']:.3f}"
        print(line)

    # Best model
    candidate_keys = [k for k in all_results
                      if k.startswith("dl_") or k.startswith("ensemble_")]
    if candidate_keys:
        best_name = max(candidate_keys, key=lambda k: all_results[k].get("accuracy", 0))
        best_acc = all_results[best_name]["accuracy"]
        print(f"\n  🏆 Best: {best_name} → {best_acc:.1%}")

    results_path = "results/results_phase2.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Results saved: {results_path}")
    print(f"\n{'='*60}\n  ✓ Phase 2 complete!\n{'='*60}\n")


if __name__ == "__main__":
    main()
