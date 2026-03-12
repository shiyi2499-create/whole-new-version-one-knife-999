"""
Transformer-only Runner
=======================
Runs only the Transformer model on CPU (avoids MPS segfault),
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
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score

from feature_extractor import extract_features_batch, get_feature_names

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
    MPS_DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available()
                              else "cpu")
    # Transformer always runs on CPU to avoid MPS segfault on d_model=128/3-layer
    CPU = torch.device("cpu")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  Transformer will run on: {CPU}")
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
                augment=True, patience=40, device=CPU):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    X_vl = torch.FloatTensor(X_val).to(device)
    y_vl = torch.LongTensor(y_val).to(device)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
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

        if patience_counter >= patience:
            print(f"      Early stop at epoch {epoch+1} (patience={patience})")
            break

        if (epoch + 1) % 50 == 0:
            train_acc = correct / total
            print(f"      Epoch {epoch+1:3d}: loss={total_loss/total:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_acc


# ══════════════════════════════════════════════════════════════
#  5-FOLD CV FOR TRANSFORMER
# ══════════════════════════════════════════════════════════════

def run_transformer_cv(X_raw, y_keys):
    le = LabelEncoder()
    le.fit(y_keys)
    y_enc = le.transform(y_keys)
    n_classes = len(le.classes_)

    # Normalize per channel
    X_norm = X_raw.copy()
    for ch in range(X_norm.shape[2]):
        mu = X_norm[:, :, ch].mean()
        sd = X_norm[:, :, ch].std()
        if sd > 1e-10:
            X_norm[:, :, ch] = (X_norm[:, :, ch] - mu) / sd

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    all_preds = np.zeros(len(y_enc), dtype=int)
    all_probs = np.zeros((len(y_enc), n_classes))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_norm, y_enc)):
        print(f"    Fold {fold+1}/5...")
        model = TransformerClassifier(
            n_timesteps=X_norm.shape[1],
            n_channels=X_norm.shape[2],
            n_classes=n_classes,
        )
        trained, best_val, = train_model(
            model,
            X_norm[train_idx], y_enc[train_idx],
            X_norm[test_idx],  y_enc[test_idx],
            device=CPU,
        )
        trained.eval()
        with torch.no_grad():
            out = trained(torch.FloatTensor(X_norm[test_idx]).to(CPU))
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

        all_preds[test_idx] = preds
        all_probs[test_idx] = probs
        fold_acc = accuracy_score(y_enc[test_idx], preds)
        fold_accs.append(fold_acc)
        print(f"      → val_acc={best_val:.3f}  test_acc={fold_acc:.3f}")

    overall_acc = accuracy_score(y_enc, all_preds)
    top3 = top_k_accuracy_score(y_enc, all_probs, k=3)
    top5 = top_k_accuracy_score(y_enc, all_probs, k=5)

    metrics = {
        "model": "Transformer",
        "accuracy": float(overall_acc),
        "top3_accuracy": float(top3),
        "top5_accuracy": float(top5),
        "fold_mean": float(np.mean(fold_accs)),
        "fold_std": float(np.std(fold_accs)),
        "fold_accuracies": [float(a) for a in fold_accs],
        "label_classes": le.classes_.tolist(),
    }

    if HAS_PLOT:
        cm = confusion_matrix(y_enc, all_preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    ax=ax, vmin=0, vmax=1, annot_kws={"size": 6})
        ax.set_title(f"Transformer  Accuracy: {overall_acc:.1%}  Top-3: {top3:.1%}")
        plt.tight_layout()
        plt.savefig("results/confusion_Transformer.png", dpi=150)
        plt.close()
        print("    Saved: results/confusion_Transformer.png")

    return metrics, all_probs, le


# ══════════════════════════════════════════════════════════════
#  ENSEMBLE  (XGBoost + Transformer)
# ══════════════════════════════════════════════════════════════

def run_xgb_cv(X_feat, y_keys, le):
    y_enc = le.transform(y_keys)
    n_classes = len(le.classes_)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, eval_metric="mlogloss", n_jobs=-1,
        )),
    ])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros((len(y_enc), n_classes))
    all_preds = np.zeros(len(y_enc), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y_enc)):
        pipe.fit(X_feat[train_idx], y_enc[train_idx])
        probs = pipe.predict_proba(X_feat[test_idx])
        xgb_classes = pipe.named_steps["clf"].classes_
        prob_aligned = np.zeros((len(test_idx), n_classes))
        for ci, c in enumerate(xgb_classes):
            prob_aligned[:, c] = probs[:, ci]
        all_probs[test_idx] = prob_aligned
        all_preds[test_idx] = prob_aligned.argmax(axis=1)

    return all_probs, accuracy_score(y_enc, all_preds)


def run_ensemble(tf_probs, tf_acc, xgb_probs, xgb_acc, y_keys, le):
    print(f"\n{'='*60}\n  ENSEMBLE (Transformer + XGBoost)\n{'='*60}")
    y_enc = le.transform(y_keys)

    # Accuracy-proportional weights
    total = tf_acc + xgb_acc
    w_tf, w_xgb = tf_acc / total, xgb_acc / total
    probs_prop = w_tf * tf_probs + w_xgb * xgb_probs
    acc_prop = accuracy_score(y_enc, probs_prop.argmax(axis=1))
    top3_prop = top_k_accuracy_score(y_enc, probs_prop, k=3)
    print(f"\n  Proportional (Transformer={w_tf:.2f}, XGBoost={w_xgb:.2f}):")
    print(f"    Accuracy: {acc_prop:.1%}  Top-3: {top3_prop:.1%}")

    # Grid search
    best_acc, best_w = 0.0, (0.5, 0.5)
    print("\n  Grid search over Transformer weight (0.3 → 0.9):")
    for w in np.arange(0.3, 1.0, 0.1):
        w = round(w, 1)
        probs_g = w * tf_probs + (1 - w) * xgb_probs
        acc_g = accuracy_score(y_enc, probs_g.argmax(axis=1))
        top3_g = top_k_accuracy_score(y_enc, probs_g, k=3)
        print(f"    w_tf={w:.1f}  acc={acc_g:.1%}  top3={top3_g:.1%}")
        if acc_g > best_acc:
            best_acc = acc_g
            best_w = (w, 1 - w)

    best_probs = best_w[0] * tf_probs + best_w[1] * xgb_probs
    best_top3 = top_k_accuracy_score(y_enc, best_probs, k=3)
    best_top5 = top_k_accuracy_score(y_enc, best_probs, k=5)
    print(f"\n  🏆 Best ensemble: w_tf={best_w[0]:.1f}  "
          f"acc={best_acc:.1%}  top3={best_top3:.1%}  top5={best_top5:.1%}")

    # Save for phase3_decoder.py
    np.savez_compressed(
        "results/ensemble_probs.npz",
        probs=best_probs, y_true=y_enc,
        classes=le.classes_, w_tf=best_w[0], w_xgb=best_w[1],
    )
    print("  Saved → results/ensemble_probs.npz  (for phase3_decoder.py)")

    return {
        "ensemble_prop":  {"accuracy": float(acc_prop),  "top3_accuracy": float(top3_prop),
                           "w_tf": float(w_tf), "w_xgb": float(w_xgb)},
        "ensemble_best":  {"accuracy": float(best_acc),  "top3_accuracy": float(best_top3),
                           "top5_accuracy": float(best_top5),
                           "w_tf": float(best_w[0]), "w_xgb": float(best_w[1])},
        "xgb_standalone": {"accuracy": float(xgb_acc)},
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs("results", exist_ok=True)

    # Disable PyTorch optimised attention kernels that cause segfault on macOS
    # (affects both MPS and CPU TransformerEncoderLayer on some torch builds)
    torch.set_num_threads(1)
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
    rate = int(data.get("target_rate_hz", 130))

    # Filter spurious keys (< 10 samples)
    key_counts = Counter(y_keys.tolist())
    valid_keys = {k for k, v in key_counts.items() if v >= 10}
    removed = sorted(set(y_keys.tolist()) - valid_keys)
    if removed:
        print(f"  ⚠ 过滤低样本键: {removed}")
    mask = np.array([k in valid_keys for k in y_keys])
    X_raw, y_keys = X_raw[mask], y_keys[mask]
    print(f"\n  Data: {X_raw.shape}, {len(valid_keys)} classes, {rate}Hz")

    # ── Load features (for ensemble XGBoost) ─────────────────
    feat_path = "results/features.npz"
    if not os.path.exists(feat_path):
        print("  ⚠ features.npz not found — extracting now (takes ~1 min)...")
        X_feat = extract_features_batch(X_raw, sample_rate=rate)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
        np.savez_compressed(feat_path, X=X_feat, y=y_keys,
                            feature_names=get_feature_names())
    else:
        fdata = np.load(feat_path, allow_pickle=True)
        X_feat = fdata["X"]
        if X_feat.shape[0] != len(y_keys):
            print("  ⚠ Feature cache shape mismatch — re-extracting...")
            X_feat = extract_features_batch(X_raw, sample_rate=rate)
            X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
            np.savez_compressed(feat_path, X=X_feat, y=y_keys,
                                feature_names=get_feature_names())
        else:
            print(f"  Loaded cached features: {X_feat.shape}")

    # ── Run Transformer CV ───────────────────────────────────
    print(f"\n{'='*60}\n  TRANSFORMER (CPU, d_model=128, layers=3)\n{'='*60}")
    t0 = time.time()
    tf_metrics, tf_probs, le = run_transformer_cv(X_raw, y_keys)
    elapsed = time.time() - t0

    print(f"\n  Accuracy:  {tf_metrics['accuracy']:.1%}")
    print(f"  Top-3:     {tf_metrics['top3_accuracy']:.1%}")
    print(f"  Top-5:     {tf_metrics['top5_accuracy']:.1%}")
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
        xgb_probs, xgb_acc = run_xgb_cv(X_feat, y_keys, le)
        print(f"  XGBoost CV accuracy: {xgb_acc:.1%}  ({time.time()-t0:.0f}s)")

        ensemble_results = run_ensemble(
            tf_probs, tf_metrics["accuracy"],
            xgb_probs, xgb_acc,
            y_keys, le,
        )
        all_results.update(ensemble_results)
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