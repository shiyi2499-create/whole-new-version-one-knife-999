"""
Training Pipeline - Phase 1: Traditional ML Baselines
======================================================
Loads merged dataset, extracts features, trains and evaluates:
  1. Per-key classification (36 classes) with RF, XGBoost, SVM
  2. Zone classification (row / hand / quadrant)
  3. Feature importance analysis
  4. Confusion matrix visualization

Run:
  cd /Users/shiyi/keystroke_collector
  .venv/bin/python3 train_baseline.py

Requires: pip install scikit-learn xgboost matplotlib seaborn
"""

import os
import sys
import time
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    top_k_accuracy_score
)
from sklearn.pipeline import Pipeline

from feature_extractor import (
    extract_features_batch, get_feature_names,
    map_to_zones, ZONE_LABELS, ZONE_MAPS
)

# ── Try optional imports ─────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ xgboost not installed, skipping XGBoost. Install: pip install xgboost")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("  ⚠ matplotlib/seaborn not installed, skipping plots.")


# ══════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════

def load_dataset(path: str = "data/processed/merged_dataset.npz"):
    """Load the merged dataset."""
    print(f"\n  Loading dataset: {path}")
    data = np.load(path, allow_pickle=True)

    X_raw = data["X"]       # (1800, 30, 6)
    y = data["y"]           # (1800,) string labels
    rate = int(data.get("target_rate_hz", 100))

    print(f"    X shape: {X_raw.shape}")
    print(f"    y shape: {y.shape}")
    print(f"    Classes: {len(set(y.tolist()))}")
    print(f"    Rate: {rate} Hz")

    return X_raw, y, rate


# ══════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

def get_models():
    """Return dict of model name → sklearn pipeline."""
    models = {}

    # Random Forest
    models["RandomForest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=-1,
            ))
        ])

    # SVM (RBF kernel) - de Souza Faria's primary model
    models["SVM_RBF"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,      # needed for top-k accuracy
            random_state=42,
        ))
    ])

    return models


# ══════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X, y, label_encoder, n_folds=5, task_name="key"):
    """
    Evaluate a model using stratified k-fold cross-validation.
    Returns predictions, probabilities, and metrics dict.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_enc = label_encoder.transform(y)
    n_classes = len(label_encoder.classes_)

    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros((len(y), n_classes))

    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        all_preds[test_idx] = preds

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            # Handle case where model might not output all classes
            if probs.shape[1] == n_classes:
                all_probs[test_idx] = probs
            else:
                # Map columns to correct class indices
                classes_in_fold = model.classes_ if hasattr(model, 'classes_') else model[-1].classes_
                for ci, c in enumerate(classes_in_fold):
                    all_probs[test_idx, c] = probs[:, ci]

        fold_acc = accuracy_score(y_test, preds)
        fold_accs.append(fold_acc)

    # Overall metrics
    overall_acc = accuracy_score(y_enc, all_preds)

    # Top-K accuracy
    top3_acc = 0.0
    top5_acc = 0.0
    if np.any(all_probs > 0):
        if n_classes >= 3:
            top3_acc = top_k_accuracy_score(y_enc, all_probs, k=min(3, n_classes))
        if n_classes >= 5:
            top5_acc = top_k_accuracy_score(y_enc, all_probs, k=min(5, n_classes))

    metrics = {
        "task": task_name,
        "accuracy": overall_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "fold_accuracies": fold_accs,
        "fold_mean": np.mean(fold_accs),
        "fold_std": np.std(fold_accs),
    }

    return all_preds, all_probs, metrics


# ══════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════

def analyze_feature_importance(model, feature_names, output_dir, top_n=25):
    """Extract and plot feature importances from tree-based models."""
    clf = model.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        print("    (model doesn't support feature_importances_)")
        return

    indices = np.argsort(importances)[::-1][:top_n]

    print(f"\n  Top {top_n} features:")
    for rank, idx in enumerate(indices):
        print(f"    {rank+1:2d}. {feature_names[idx]:35s}  {importances[idx]:.4f}")

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(10, 8))
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]
        ax.barh(range(top_n), top_vals[::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances")
        plt.tight_layout()
        path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"    Saved: {path}")


# ══════════════════════════════════════════════════════════════
#  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, y_pred, classes, title, output_path):
    """Plot and save confusion matrix."""
    if not HAS_PLOT:
        return

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                vmin=0, vmax=1, annot_kws={"size": 6})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"    Saved: {output_path}")


# ══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"\n{'='*60}\n"
        f"  🧠 TRAINING PIPELINE - Phase 1: Traditional ML Baselines\n"
        f"{'='*60}"
    )

    # ── 1. Load data ─────────────────────────────────────────
    X_raw, y_keys, rate = load_dataset()

    # 过滤掉采集时误触的杂类（样本数 < 10 的键，如 capslock 等）
    from collections import Counter
    key_counts = Counter(y_keys.tolist())
    valid_keys = {k for k, v in key_counts.items() if v >= 10}
    removed = sorted(set(y_keys.tolist()) - valid_keys)
    if removed:
        print(f"  ⚠ 过滤低样本键: {removed} (各 {[key_counts[k] for k in removed]} 次)")
    mask = np.array([k in valid_keys for k in y_keys])
    X_raw, y_keys = X_raw[mask], y_keys[mask]
    print(f"  过滤后: {len(y_keys)} 样本, {len(valid_keys)} 类")

    # ── 2. Extract features ──────────────────────────────────
    n_features = len(get_feature_names())
    print(f"\n  Extracting {n_features} features per sample...")
    t0 = time.time()
    X_feat = extract_features_batch(X_raw, sample_rate=rate)
    feat_time = time.time() - t0
    print(f"    Time: {feat_time:.1f}s")

    feature_names = get_feature_names()

    # Check for NaN/Inf
    nan_count = np.sum(np.isnan(X_feat))
    inf_count = np.sum(np.isinf(X_feat))
    if nan_count > 0 or inf_count > 0:
        print(f"    ⚠ Found {nan_count} NaN, {inf_count} Inf values. Replacing with 0.")
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

    # Save features for reuse
    feat_path = os.path.join(output_dir, "features.npz")
    np.savez_compressed(feat_path, X=X_feat, y=y_keys,
                        feature_names=feature_names)
    print(f"    Saved features: {feat_path}")

    # ── 3. Key classification (36 classes) ───────────────────
    print(
        f"\n{'='*60}\n"
        f"  TASK 1: Per-Key Classification ({len(valid_keys)} classes)\n"
        f"{'='*60}"
    )

    le_key = LabelEncoder()
    le_key.fit(y_keys)

    models = get_models()
    all_results = {}

    for name, model in models.items():
        print(f"\n  ── {name} ──")
        t0 = time.time()
        preds, probs, metrics = evaluate_model(
            model, X_feat, y_keys, le_key, n_folds=5, task_name="key_36"
        )
        elapsed = time.time() - t0

        print(f"    Accuracy:     {metrics['accuracy']:.3f}  "
              f"({metrics['accuracy']*100:.1f}%)")
        print(f"    Top-3 acc:    {metrics['top3_accuracy']:.3f}  "
              f"({metrics['top3_accuracy']*100:.1f}%)")
        print(f"    Top-5 acc:    {metrics['top5_accuracy']:.3f}  "
              f"({metrics['top5_accuracy']*100:.1f}%)")
        print(f"    Fold accs:    {[f'{a:.3f}' for a in metrics['fold_accuracies']]}")
        print(f"    Mean±Std:     {metrics['fold_mean']:.3f} ± {metrics['fold_std']:.3f}")
        print(f"    Time:         {elapsed:.1f}s")

        all_results[f"key36_{name}"] = metrics

        # Confusion matrix for best model
        plot_confusion_matrix(
            le_key.transform(y_keys), preds,
            le_key.classes_,
            f"Key Classification - {name}\nAccuracy: {metrics['accuracy']:.1%}",
            os.path.join(output_dir, f"confusion_key36_{name}.png")
        )

    # ── 4. Feature importance (from RF) ──────────────────────
    print(
        f"\n{'='*60}\n"
        f"  FEATURE IMPORTANCE ANALYSIS\n"
        f"{'='*60}"
    )

    # Re-fit RF on all data for importance analysis
    rf_model = models["RandomForest"]
    le_temp = LabelEncoder()
    y_enc_all = le_temp.fit_transform(y_keys)
    rf_model.fit(X_feat, y_enc_all)
    analyze_feature_importance(rf_model, feature_names, output_dir)

    # Importance by sensor type
    importances = rf_model.named_steps["clf"].feature_importances_
    sensor_importance = defaultdict(float)
    for i, fname in enumerate(feature_names):
        if fname.startswith("accel_x"): sensor_importance["accel_x"] += importances[i]
        elif fname.startswith("accel_y"): sensor_importance["accel_y"] += importances[i]
        elif fname.startswith("accel_z"): sensor_importance["accel_z"] += importances[i]
        elif fname.startswith("gyro_x"): sensor_importance["gyro_x"] += importances[i]
        elif fname.startswith("gyro_y"): sensor_importance["gyro_y"] += importances[i]
        elif fname.startswith("gyro_z"): sensor_importance["gyro_z"] += importances[i]
        elif fname.startswith("ncc_"): sensor_importance["cross_corr"] += importances[i]
        elif fname.startswith("accel_mag"): sensor_importance["accel_magnitude"] += importances[i]

    print("\n  Importance by sensor/feature group:")
    for name, imp in sorted(sensor_importance.items(), key=lambda x: -x[1]):
        print(f"    {name:20s}  {imp:.4f}")

    # ── 5. Zone classification ───────────────────────────────
    print(
        f"\n{'='*60}\n"
        f"  TASK 2: Zone Classification\n"
        f"{'='*60}"
    )

    # Use best model (RF is fast and good for this)
    for zone_type in ["hand", "row", "quadrant"]:
        y_zone = map_to_zones(y_keys, zone_type)

        # Skip samples that don't map (-1)
        valid = y_zone >= 0
        X_zone = X_feat[valid]
        y_zone_valid = y_zone[valid]

        n_classes = len(set(y_zone_valid))
        zone_names = ZONE_LABELS[zone_type]

        print(f"\n  ── Zone: {zone_type} ({n_classes} classes) ──")
        for zid, zname in zone_names.items():
            count = np.sum(y_zone_valid == zid)
            print(f"    {zname}: {count} samples")

        le_zone = LabelEncoder()
        le_zone.fit(y_zone_valid)

        # Use RF for zone classification
        rf_zone = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1,
                class_weight="balanced",
            ))
        ])

        preds, probs, metrics = evaluate_model(
            rf_zone, X_zone, y_zone_valid.astype(str), le_zone,
            n_folds=5, task_name=f"zone_{zone_type}"
        )

        print(f"    Accuracy:  {metrics['accuracy']:.3f}  "
              f"({metrics['accuracy']*100:.1f}%)")
        print(f"    Fold:      {metrics['fold_mean']:.3f} ± {metrics['fold_std']:.3f}")

        all_results[f"zone_{zone_type}"] = metrics

        # Confusion matrix
        class_labels = [zone_names.get(int(c), str(c)) for c in le_zone.classes_]
        plot_confusion_matrix(
            le_zone.transform(y_zone_valid.astype(str)), preds,
            class_labels,
            f"Zone Classification: {zone_type}\nAccuracy: {metrics['accuracy']:.1%}",
            os.path.join(output_dir, f"confusion_zone_{zone_type}.png")
        )

    # ── 6. Per-key accuracy breakdown ────────────────────────
    print(
        f"\n{'='*60}\n"
        f"  PER-KEY ACCURACY BREAKDOWN (best model)\n"
        f"{'='*60}"
    )

    # Re-run best model to get per-key results
    best_model_name = max(
        [(k, v["accuracy"]) for k, v in all_results.items() if k.startswith("key36_")],
        key=lambda x: x[1]
    )[0]
    best_model_short = best_model_name.replace("key36_", "")
    print(f"  Best model: {best_model_short}")

    best_model = models[best_model_short]
    preds, probs, metrics = evaluate_model(
        best_model, X_feat, y_keys, le_key, n_folds=5
    )
    y_enc = le_key.transform(y_keys)

    key_accs = {}
    for ki, key in enumerate(le_key.classes_):
        mask = y_enc == ki
        if np.sum(mask) > 0:
            key_acc = accuracy_score(y_enc[mask], preds[mask])
            key_accs[key] = key_acc

    # Sort by accuracy
    sorted_keys = sorted(key_accs.items(), key=lambda x: -x[1])
    print("\n  Easiest keys (highest accuracy):")
    for k, a in sorted_keys[:10]:
        print(f"    '{k}': {a:.1%}")
    print("\n  Hardest keys (lowest accuracy):")
    for k, a in sorted_keys[-10:]:
        print(f"    '{k}': {a:.1%}")

    # ── 7. Summary ───────────────────────────────────────────
    print(
        f"\n{'='*60}\n"
        f"  📊 RESULTS SUMMARY\n"
        f"{'='*60}\n"
    )

    for task_name, metrics in all_results.items():
        line = f"  {task_name:30s}  acc={metrics['accuracy']:.1%}"
        if metrics.get("top3_accuracy", 0) > 0:
            line += f"  top3={metrics['top3_accuracy']:.1%}"
        if metrics.get("top5_accuracy", 0) > 0:
            line += f"  top5={metrics['top5_accuracy']:.1%}"
        print(line)

    # Save all results
    results_path = os.path.join(output_dir, "results_phase1.json")
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            kk: (vv if not isinstance(vv, np.floating) else float(vv))
            for kk, vv in v.items()
            if kk != "fold_accuracies"
        }
        serializable[k]["fold_accuracies"] = [float(x) for x in v["fold_accuracies"]]

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # Save per-key accuracies
    key_acc_path = os.path.join(output_dir, "per_key_accuracy.json")
    with open(key_acc_path, "w") as f:
        json.dump({k: float(v) for k, v in key_accs.items()}, f, indent=2)
    print(f"  Per-key accuracy saved: {key_acc_path}")

    print(
        f"\n{'='*60}\n"
        f"  ✓ Phase 1 complete!\n"
        f"  Output files in: {output_dir}/\n"
        f"{'='*60}\n"
    )


if __name__ == "__main__":
    main()