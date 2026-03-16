"""
Transformer-only Runner
=======================
Runs only the Transformer model with switchable runtime profile/device,
then merges results with the already-finished CNN/BiLSTM results
from results/results_phase2.json, and runs the XGBoost ensemble.

Assumes train_phase2.py has already been run and produced:
  results/results_phase2.json   ← 1D_CNN + CNN_BiLSTM results
  results/features.npz          ← cached feature matrix (for ensemble)

Run:
  .venv/bin/python3 run_transformer_only.py
"""

import os
import sys
import time
import json
import copy
import platform
import random
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from typing import Optional

from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, GroupShuffleSplit, StratifiedShuffleSplit
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, top_k_accuracy_score, f1_score, recall_score
)

from feature_extractor import extract_features_batch, get_feature_names

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable
    print("  ⚠ tqdm not installed. Install with: pip install tqdm")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ xgboost not installed — ensemble will be skipped")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True

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

        if req == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available.")
        if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested device=mps but MPS is not available.")
        if req not in {"cpu", "mps", "cuda"}:
            raise ValueError("Unsupported device. Use one of auto/cpu/mps/cuda.")
        return torch.device(req)

    ACTIVE_DEVICE = resolve_torch_device(os.environ.get("KEYSTROKE_DEVICE", "auto"))
    print(f"  PyTorch device (initial): {ACTIVE_DEVICE}")
except ImportError:
    HAS_TORCH = False
    print("  ⚠ PyTorch not installed")
    sys.exit(1)

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
RUNTIME_NUM_WORKERS = 0
RUNTIME_TORCH_THREADS = 1
RUNTIME_DETERMINISTIC = True
RUNTIME_XGB_JOBS = 1
RUNTIME_SPLIT_MODE = "auto"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer-only training with runtime profile/device switches"
    )
    parser.add_argument(
        "--profile",
        choices=["mac", "server"],
        default="mac",
        help="Runtime preset: mac=CPU-safe deterministic, server=GPU-oriented defaults."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Torch device override (default: auto; profile still sets sensible defaults)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers override."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="PyTorch CPU thread count override."
    )
    parser.add_argument(
        "--xgb-jobs",
        type=int,
        default=None,
        help="n_jobs for XGBoost override."
    )
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Disable strict deterministic settings for max throughput."
    )
    parser.add_argument(
        "--split-mode",
        choices=["auto", "group", "sample"],
        default="auto",
        help="CV split mode: auto (default), group (session-wise), sample (stratified sample-wise)."
    )
    return parser.parse_args()


def configure_runtime(args):
    global ACTIVE_DEVICE
    global RUNTIME_NUM_WORKERS, RUNTIME_TORCH_THREADS, RUNTIME_DETERMINISTIC, RUNTIME_XGB_JOBS
    global RUNTIME_SPLIT_MODE

    if args.profile == "mac":
        default_device = "cpu"
        default_workers = 0
        default_threads = 1
        default_xgb_jobs = 1
    else:
        default_device = "auto"
        default_workers = 4
        default_threads = max(1, (os.cpu_count() or 8) // 2)
        default_xgb_jobs = -1

    req_device = args.device if args.device != "auto" else default_device
    ACTIVE_DEVICE = resolve_torch_device(req_device)
    RUNTIME_NUM_WORKERS = args.num_workers if args.num_workers is not None else default_workers
    RUNTIME_TORCH_THREADS = args.threads if args.threads is not None else default_threads
    RUNTIME_XGB_JOBS = args.xgb_jobs if args.xgb_jobs is not None else default_xgb_jobs
    RUNTIME_DETERMINISTIC = not args.nondeterministic
    RUNTIME_SPLIT_MODE = args.split_mode

    print(
        "  Runtime config:\n"
        f"    profile={args.profile}\n"
        f"    device={ACTIVE_DEVICE}\n"
        f"    num_workers={RUNTIME_NUM_WORKERS}\n"
        f"    torch_threads={RUNTIME_TORCH_THREADS}\n"
        f"    xgb_jobs={RUNTIME_XGB_JOBS}\n"
        f"    deterministic={RUNTIME_DETERMINISTIC}\n"
        f"    split_mode={RUNTIME_SPLIT_MODE}"
    )


def set_global_determinism(seed: int = SEED):
    np.random.seed(seed)
    random.seed(seed)
    if not HAS_TORCH:
        return
    torch.set_num_threads(max(1, int(RUNTIME_TORCH_THREADS)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if RUNTIME_DETERMINISTIC:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        torch.use_deterministic_algorithms(RUNTIME_DETERMINISTIC, warn_only=True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = RUNTIME_DETERMINISTIC
        torch.backends.cudnn.benchmark = (not RUNTIME_DETERMINISTIC)


def _seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _topk_safe(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    if probs.shape[1] < k:
        return 0.0
    return float(top_k_accuracy_score(y_true, probs, k=k))


def _bootstrap_acc_ci(y_true: np.ndarray, y_pred: np.ndarray,
                      n_boot: int = 500, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0
    idx = np.arange(n)
    scores = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        b = rng.choice(idx, size=n, replace=True)
        scores[i] = np.mean(y_true[b] == y_pred[b])
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def _summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                     classes: np.ndarray) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    ci_lo, ci_hi = _bootstrap_acc_ci(y_true, y_pred, n_boot=500, seed=SEED)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    per_cls = recall_score(
        y_true, y_pred, labels=np.arange(len(classes)),
        average=None, zero_division=0
    )
    per_key_recall = {str(k): float(v) for k, v in zip(classes.tolist(), per_cls.tolist())}
    return {
        "accuracy": acc,
        "accuracy_ci95": [ci_lo, ci_hi],
        "top3_accuracy": _topk_safe(y_true, probs, k=3),
        "top5_accuracy": _topk_safe(y_true, probs, k=5),
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
                        n_splits: int = 5,
                        split_mode: str = "auto") -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    mode = (split_mode or "auto").lower()

    if mode == "sample":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(splitter.split(np.zeros(len(y)), y)), "StratifiedKFold(sample)"

    if mode == "group":
        if groups is None:
            raise ValueError("split_mode=group requires session_ids in merged_dataset.npz")
        uniq = np.unique(groups)
        if len(uniq) < n_splits:
            raise ValueError(f"split_mode=group requires at least {n_splits} unique groups, got {len(uniq)}")
        if HAS_STRATIFIED_GROUP:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            return list(splitter.split(np.zeros(len(y)), y, groups)), "StratifiedGroupKFold(group)"
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(y)), y, groups)), "GroupKFold(group)"

    if groups is not None:
        uniq = np.unique(groups)
        if len(uniq) >= n_splits:
            if HAS_STRATIFIED_GROUP:
                splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
                return list(splitter.split(np.zeros(len(y)), y, groups)), "StratifiedGroupKFold(auto)"
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(np.zeros(len(y)), y, groups)), "GroupKFold(auto)"
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(splitter.split(np.zeros(len(y)), y)), "StratifiedKFold(auto)"


def _split_train_val(train_idx: np.ndarray, y: np.ndarray,
                     groups: Optional[np.ndarray], fold: int,
                     split_protocol: str = "") -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED + fold)
    proto = (split_protocol or "").lower()
    prefer_group = ("group" in proto) and ("sample" not in proto)

    if prefer_group and groups is not None:
        tr_groups = groups[train_idx]
        if len(np.unique(tr_groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED + fold)
            tr_local, vl_local = next(gss.split(np.zeros(len(train_idx)), y[train_idx], tr_groups))
            return train_idx[tr_local], train_idx[vl_local]

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
#  DATA AUGMENTATION  (same as train_phase2.py)
# ══════════════════════════════════════════════════════════════

def augment_batch(X_batch, p=0.5):
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
            X_aug[i] *= 0.8 + 0.4 * np.random.random()
        elif aug_type == "ch_drop":
            X_aug[i][:, np.random.randint(0, C)] = 0.0
    return X_aug


# ══════════════════════════════════════════════════════════════
#  TRANSFORMER MODEL  (identical to train_phase2.py)
# ══════════════════════════════════════════════════════════════

class TransformerClassifier(nn.Module):
    """
    Transformer for vibration sequences.
    d_model=64, nhead=4, num_layers=3 — stable on macOS CPU/MPS.
    Deeper than original (2→3 layers) with higher dropout to prevent overfit.
    Note: d_model=128 triggers a PyTorch SDPA segfault on macOS; d_model=64
    is the safe ceiling and still outperforms the 2-layer version.
    """
    def __init__(self, n_timesteps=39, n_channels=6, n_classes=42,
                 d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_timesteps, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,   # 2× d_model
            dropout=0.3,           # ↑ from original 0.2
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.35),      # ↑ from original 0.3
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.25),      # ↑ from original 0.2
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=350, lr=5e-4, batch_size=32,
                augment=True, patience=40, device=None, seed=SEED,
                num_workers=None, progress_desc="Epoch"):
    if device is None:
        device = ACTIVE_DEVICE
    if num_workers is None:
        num_workers = RUNTIME_NUM_WORKERS
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_vl = torch.FloatTensor(X_val).to(device)
    y_vl = torch.LongTensor(y_val).to(device)

    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=dl_gen,
        persistent_workers=(num_workers > 0),
    )

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    epoch_iter = tqdm(
        range(epochs),
        desc=progress_desc,
        unit="ep",
        dynamic_ncols=True,
        leave=False,
        mininterval=0.5,
    )
    for epoch in epoch_iter:
        model.train()
        total_loss, correct, total = 0, 0, 0
        for xb, yb in loader:
            if augment:
                xb = augment_batch(xb, p=0.5)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_vl)
            val_acc = (val_out.argmax(dim=1) == y_vl).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        epoch_iter.set_postfix({
            "val": f"{val_acc:.3f}",
            "best": f"{best_val_acc:.3f}",
            "pat": f"{patience_counter}/{patience}",
        }, refresh=False)

        if patience_counter >= patience:
            print(f"      Early stop at epoch {epoch+1} (patience={patience})")
            break

        if (epoch + 1) % 50 == 0:
            train_acc = correct / total
            print(f"      Epoch {epoch+1:3d}: loss={total_loss/total:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    if hasattr(epoch_iter, "close"):
        epoch_iter.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# ══════════════════════════════════════════════════════════════
#  5-FOLD CV FOR TRANSFORMER
# ══════════════════════════════════════════════════════════════

def run_transformer_cv(X_raw, y_keys, groups=None, outer_splits=None, split_mode=None):
    le = LabelEncoder()
    le.fit(y_keys)
    y_enc = le.transform(y_keys)
    n_classes = len(le.classes_)

    if outer_splits is None:
        outer_splits, split_mode = _build_outer_splits(
            y_enc, groups, n_splits=5, split_mode=RUNTIME_SPLIT_MODE
        )
    elif split_mode is None:
        split_mode = "precomputed"

    fold_accs = []
    all_preds = np.zeros(len(y_enc), dtype=int)
    all_probs = np.zeros((len(y_enc), n_classes))
    n_folds = len(outer_splits)

    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        print(f"    Fold {fold+1}/{n_folds}...")
        fold_seed = SEED + fold
        set_global_determinism(fold_seed)
        train_sub_idx, val_idx = _split_train_val(
            train_idx, y_enc, groups, fold=fold, split_protocol=split_mode
        )
        if len(train_sub_idx) == 0 or len(val_idx) == 0:
            train_sub_idx, val_idx = train_idx, test_idx

        mu = X_raw[train_sub_idx].mean(axis=(0, 1), keepdims=True)
        sd = X_raw[train_sub_idx].std(axis=(0, 1), keepdims=True)
        sd[sd < 1e-10] = 1.0
        X_train = (X_raw[train_sub_idx] - mu) / sd
        X_val = (X_raw[val_idx] - mu) / sd
        X_test = (X_raw[test_idx] - mu) / sd

        model = TransformerClassifier(
            n_timesteps=X_raw.shape[1],
            n_channels=X_raw.shape[2],
            n_classes=n_classes,
        )
        trained, best_val, = train_model(
            model,
            X_train, y_enc[train_sub_idx],
            X_val, y_enc[val_idx],
            device=ACTIVE_DEVICE,
            seed=fold_seed, num_workers=RUNTIME_NUM_WORKERS,
            progress_desc=f"Transformer F{fold+1}/{n_folds}",
        )
        trained.eval()
        with torch.no_grad():
            out = trained(torch.FloatTensor(X_test).to(ACTIVE_DEVICE))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

        all_preds[test_idx] = preds
        all_probs[test_idx] = probs
        fold_acc = accuracy_score(y_enc[test_idx], preds)
        fold_accs.append(fold_acc)
        print(f"      → val_acc={best_val:.3f}  test_acc={fold_acc:.3f}")

    summary = _summary_metrics(y_enc, all_preds, all_probs, le.classes_)
    metrics = {
        "model": "Transformer",
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
        ax.set_title(
            f"Transformer  Accuracy: {summary['accuracy']:.1%}  "
            f"Top-3: {summary['top3_accuracy']:.1%}"
        )
        plt.tight_layout()
        plt.savefig("results/confusion_Transformer.png", dpi=150)
        plt.close()
        print("    Saved: results/confusion_Transformer.png")

    return metrics, all_probs, le


# ══════════════════════════════════════════════════════════════
#  ENSEMBLE  (XGBoost + Transformer)
# ══════════════════════════════════════════════════════════════

def run_xgb_cv(X_feat, y_keys, le, outer_splits=None, groups=None, split_mode=None):
    y_enc = le.transform(y_keys)
    n_classes = len(le.classes_)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss", n_jobs=RUNTIME_XGB_JOBS,
        )),
    ])
    if outer_splits is None:
        outer_splits, split_mode = _build_outer_splits(
            y_enc, groups, n_splits=5, split_mode=RUNTIME_SPLIT_MODE
        )
    elif split_mode is None:
        split_mode = "precomputed"

    all_probs = np.zeros((len(y_enc), n_classes))
    all_preds = np.zeros(len(y_enc), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(outer_splits):
        pipe.fit(X_feat[train_idx], y_enc[train_idx])
        probs = pipe.predict_proba(X_feat[test_idx])
        xgb_classes = pipe.named_steps["clf"].classes_
        prob_aligned = np.zeros((len(test_idx), n_classes))
        for ci, c in enumerate(xgb_classes):
            prob_aligned[:, c] = probs[:, ci]
        all_probs[test_idx] = prob_aligned
        all_preds[test_idx] = prob_aligned.argmax(axis=1)

    metrics = _summary_metrics(y_enc, all_preds, all_probs, le.classes_)
    metrics["split_protocol"] = split_mode
    return all_probs, metrics


def run_ensemble(tf_probs, tf_acc, xgb_probs, xgb_acc, y_keys, le):
    print(f"\n{'='*60}\n  ENSEMBLE (Transformer + XGBoost)\n{'='*60}")
    y_enc = le.transform(y_keys)

    # Accuracy-proportional weights
    total = tf_acc + xgb_acc
    w_tf, w_xgb = tf_acc / total, xgb_acc / total
    probs_prop = w_tf * tf_probs + w_xgb * xgb_probs
    preds_prop = probs_prop.argmax(axis=1)
    summary_prop = _summary_metrics(y_enc, preds_prop, probs_prop, le.classes_)
    print(f"\n  Proportional (Transformer={w_tf:.2f}, XGBoost={w_xgb:.2f}):")
    print(f"    Accuracy: {summary_prop['accuracy']:.1%}  Top-3: {summary_prop['top3_accuracy']:.1%}")

    # Grid search
    best_acc, best_w = 0.0, (0.5, 0.5)
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
    summary_best = _summary_metrics(y_enc, best_preds, best_probs, le.classes_)
    print(f"\n  🏆 Best ensemble: w_tf={best_w[0]:.1f}  "
          f"acc={summary_best['accuracy']:.1%}  "
          f"top3={summary_best['top3_accuracy']:.1%}  "
          f"top5={summary_best['top5_accuracy']:.1%}")

    # Save for phase3_decoder.py
    np.savez_compressed(
        "results/ensemble_probs.npz",
        probs=best_probs, y_true=y_enc,
        classes=le.classes_, w_tf=best_w[0], w_xgb=best_w[1],
    )
    print("  Saved → results/ensemble_probs.npz  (for phase3_decoder.py)")

    return {
        "ensemble_prop": {
            **summary_prop,
            "w_tf": float(w_tf), "w_xgb": float(w_xgb)
        },
        "ensemble_best": {
            **summary_best,
            "w_tf": float(best_w[0]), "w_xgb": float(best_w[1])
        },
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    configure_runtime(args)
    os.makedirs("results", exist_ok=True)
    set_global_determinism(SEED)

    # On macOS/CPU, disable optimized SDPA kernels to avoid historical segfaults.
    # Keep CUDA kernels enabled on server GPUs for speed.
    if ACTIVE_DEVICE.type != "cuda":
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        except Exception:
            pass  # older torch versions don't have these flags

    print(f"\n{'='*60}\n  🤖 TRANSFORMER-ONLY RUNNER\n{'='*60}")

    # ── Load data ────────────────────────────────────────────
    data = np.load("data/processed/merged_dataset.npz", allow_pickle=True)
    X_raw = data["X"].astype(np.float32)
    y_keys = data["y"]
    rate = int(data.get("target_rate_hz", 190))
    groups = _get_groups_from_data(data, len(y_keys))
    if groups is None:
        print("  ⚠ merged_dataset.npz has no session_ids; using legacy stratified split.")
        print("    Re-run preprocessor to enable session-group split.")

    # Filter spurious keys (< 10 samples)
    key_counts = Counter(y_keys.tolist())
    valid_keys = {k for k, v in key_counts.items() if v >= 10}
    removed = sorted(set(y_keys.tolist()) - valid_keys)
    if removed:
        print(f"  ⚠ 过滤低样本键: {removed}")
    mask = np.array([k in valid_keys for k in y_keys])
    X_raw, y_keys = X_raw[mask], y_keys[mask]
    if groups is not None:
        groups = groups[mask]
    print(f"\n  Data: {X_raw.shape}, {len(valid_keys)} classes, {rate}Hz")
    outer_splits, split_mode = _build_outer_splits(
        y_keys, groups, n_splits=5, split_mode=RUNTIME_SPLIT_MODE
    )
    uniq_groups = len(np.unique(groups)) if groups is not None else 0
    print(f"  Split protocol: {split_mode}  (groups={uniq_groups})")

    # ── Load features (for ensemble XGBoost) ─────────────────
    feat_path = "results/features.npz"
    if not os.path.exists(feat_path):
        print("  ⚠ features.npz not found — extracting now (takes ~1 min)...")
        X_feat = extract_features_batch(X_raw, sample_rate=rate)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
        payload = {"X": X_feat, "y": y_keys, "feature_names": get_feature_names()}
        if groups is not None:
            payload["session_ids"] = groups
        np.savez_compressed(feat_path, **payload)
    else:
        fdata = np.load(feat_path, allow_pickle=True)
        X_feat = fdata["X"]
        cache_ok = X_feat.shape[0] == len(y_keys)
        if cache_ok and "y" in fdata.files:
            cache_ok = np.array_equal(fdata["y"], y_keys)
        if cache_ok and groups is not None and "session_ids" in fdata.files:
            cache_ok = np.array_equal(fdata["session_ids"].astype(str), groups.astype(str))
        if not cache_ok:
            print("  ⚠ Feature cache shape mismatch — re-extracting...")
            X_feat = extract_features_batch(X_raw, sample_rate=rate)
            X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
            payload = {"X": X_feat, "y": y_keys, "feature_names": get_feature_names()}
            if groups is not None:
                payload["session_ids"] = groups
            np.savez_compressed(feat_path, **payload)
        else:
            print(f"  Loaded cached features: {X_feat.shape}")

    # ── Run Transformer CV ───────────────────────────────────
    print(f"\n{'='*60}\n  TRANSFORMER ({ACTIVE_DEVICE}, d_model=64, layers=3)\n{'='*60}")
    t0 = time.time()
    tf_metrics, tf_probs, le = run_transformer_cv(
        X_raw, y_keys,
        groups=groups, outer_splits=outer_splits, split_mode=split_mode,
    )
    elapsed = time.time() - t0

    print(f"\n  Accuracy:  {tf_metrics['accuracy']:.1%}")
    print(f"  Acc CI95:  [{tf_metrics['accuracy_ci95'][0]:.1%}, {tf_metrics['accuracy_ci95'][1]:.1%}]")
    print(f"  Top-3:     {tf_metrics['top3_accuracy']:.1%}")
    print(f"  Top-5:     {tf_metrics['top5_accuracy']:.1%}")
    print(f"  Macro-F1:  {tf_metrics['macro_f1']:.3f}")
    print(f"  Folds:     {[f'{a:.3f}' for a in tf_metrics['fold_accuracies']]}")
    print(f"  Mean±Std:  {tf_metrics['fold_mean']:.3f} ± {tf_metrics['fold_std']:.3f}")
    print(f"  Time:      {elapsed:.0f}s")

    # Save Transformer probs for phase3_decoder.py
    np.savez_compressed(
        "results/transformer_probs.npz",
        probs=tf_probs,
        y_true=le.transform(y_keys),
        classes=le.classes_,
    )
    print("  Saved → results/transformer_probs.npz")

    # ── Merge with existing phase2 results ───────────────────
    p2_path = "results/results_phase2.json"
    if os.path.exists(p2_path):
        with open(p2_path) as f:
            all_results = json.load(f)
        print(f"\n  Loaded existing phase2 results ({len(all_results)} entries)")
    else:
        all_results = {}
        print("  ⚠ results_phase2.json not found — starting fresh")

    all_results["dl_Transformer"] = tf_metrics

    # ── Ensemble ─────────────────────────────────────────────
    if HAS_XGB:
        print(f"\n  Running XGBoost CV for ensemble...")
        t0 = time.time()
        xgb_probs, xgb_metrics = run_xgb_cv(
            X_feat, y_keys, le,
            outer_splits=outer_splits, groups=groups, split_mode=split_mode,
        )
        xgb_acc = xgb_metrics["accuracy"]
        print(f"  XGBoost CV accuracy: {xgb_acc:.1%}  ({time.time()-t0:.0f}s)")

        ensemble_results = run_ensemble(
            tf_probs, tf_metrics["accuracy"],
            xgb_probs, xgb_acc,
            y_keys, le,
        )
        all_results.update(ensemble_results)
        all_results["xgb_standalone"] = xgb_metrics
    else:
        print("\n  ⚠ Skipping ensemble (xgboost not installed)")

    # ── Final summary ────────────────────────────────────────
    print(f"\n{'='*60}\n  📊 FULL RESULTS SUMMARY\n{'='*60}\n")

    # Phase 1
    p1_path = "results/results_phase1.json"
    if os.path.exists(p1_path):
        with open(p1_path) as f:
            p1 = json.load(f)
        print("  ── Phase 1 ──")
        for k, v in p1.items():
            line = f"    {k:35s}  acc={v['accuracy']:.1%}"
            if v.get("top3_accuracy", 0) > 0:
                line += f"  top3={v['top3_accuracy']:.1%}"
            print(line)

    print("\n  ── Phase 2 ──")
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

    candidates = {k: v["accuracy"] for k, v in all_results.items()
                  if k.startswith("dl_") or k.startswith("ensemble_")}
    if candidates:
        best_k = max(candidates, key=candidates.get)
        print(f"\n  🏆 Best: {best_k} → {candidates[best_k]:.1%}")

    # Save merged results back
    with open(p2_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n  Results saved → {p2_path}")
    print(f"\n{'='*60}\n  ✓ Done!\n{'='*60}\n")


if __name__ == "__main__":
    main()
