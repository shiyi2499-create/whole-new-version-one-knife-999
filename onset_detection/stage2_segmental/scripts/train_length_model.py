#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(THIS_DIR)
ONSET_ROOT = os.path.dirname(PKG_ROOT)
REPO_ROOT = os.path.dirname(ONSET_ROOT)
for p in (REPO_ROOT, ONSET_ROOT, PKG_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_segmental.length_model import (
    extract_attempt_length_examples,
    extract_mixed_episode_length_examples,
    save_length_model,
)


def parse_dataset_spec(spec: str):
    # format: LEN:DIR:TRAIN_END:TEST_START
    parts = spec.split(':')
    if len(parts) != 4:
        raise ValueError(f'Bad --dataset spec: {spec}')
    length = int(parts[0])
    directory = parts[1]
    train_end = int(parts[2])
    test_start = int(parts[3])
    return length, directory, train_end, test_start


def parse_mixed_dataset_spec(spec: str):
    # format: LEN:DIR
    parts = spec.split(':', 1)
    if len(parts) != 2:
        raise ValueError(f'Bad --mixed-dataset spec: {spec}')
    return int(parts[0]), parts[1]


def discover_prefixes(password_dir: str):
    d = Path(password_dir)
    out = []
    for p in sorted(d.glob('p01_free_type_password_part*_sensor.csv')):
        prefix = str(p).replace('_sensor.csv', '')
        if Path(prefix + '_attempts.csv').exists() and Path(prefix + '_events.csv').exists():
            out.append(prefix)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', action='append', required=True, help='LEN:DIR:TRAIN_END:TEST_START')
    ap.add_argument('--mixed-dataset', action='append', default=[], help='LEN:DIR (mixed GT-context positive examples)')
    ap.add_argument('--model-out', required=True)
    ap.add_argument('--report-out', required=True)
    ap.add_argument('--feature-mode', default='no_time', choices=['no_time', 'legacy_time'])
    ap.add_argument(
        '--mixed-use-cluster-subregion',
        action='store_true',
        help='Train mixed examples on the same peak-cluster subregion used during inference.',
    )
    args = ap.parse_args()

    train_x, train_y, test_x, test_y = [], [], [], []
    dataset_rows = []
    for spec in args.dataset:
        length, password_dir, train_end, test_start = parse_dataset_spec(spec)
        prefixes = discover_prefixes(password_dir)
        used = 0
        for prefix in prefixes:
            part = int(prefix.split('_part')[1].split('_')[0])
            xs, ys = extract_attempt_length_examples(prefix, true_len=length, feature_mode=args.feature_mode)
            if part <= train_end:
                train_x.extend(xs); train_y.extend(ys)
            elif part >= test_start:
                test_x.extend(xs); test_y.extend(ys)
            used += len(xs)
        dataset_rows.append({
            'length': length,
            'password_dir': password_dir,
            'num_examples': used,
            'train_end': train_end,
            'test_start': test_start,
        })

    mixed_rows = []
    for spec in args.mixed_dataset:
        length, input_dir = parse_mixed_dataset_spec(spec)
        xs, ys = extract_mixed_episode_length_examples(
            input_dir,
            true_len=length,
            feature_mode=args.feature_mode,
            use_cluster_subregion=bool(args.mixed_use_cluster_subregion),
        )
        train_x.extend(xs)
        train_y.extend(ys)
        mixed_rows.append({
            'length': length,
            'mixed_input_dir': input_dir,
            'num_examples': len(xs),
            'use_cluster_subregion': bool(args.mixed_use_cluster_subregion),
        })

    Xtr = np.asarray(train_x, dtype=np.float32)
    ytr = np.asarray(train_y)
    Xte = np.asarray(test_x, dtype=np.float32)
    yte = np.asarray(test_y)

    models = {
        'logreg': LogisticRegression(max_iter=5000),
        'rf': RandomForestClassifier(n_estimators=300, random_state=42),
        'extra': ExtraTreesClassifier(n_estimators=300, random_state=42),
        'gb': GradientBoostingClassifier(random_state=42),
    }
    results = {
        'datasets': dataset_rows,
        'mixed_datasets': mixed_rows,
        'feature_mode': args.feature_mode,
        'train_size': int(len(ytr)),
        'test_size': int(len(yte)),
        'train_counts': {str(k): int(v) for k, v in Counter(ytr).items()},
        'test_counts': {str(k): int(v) for k, v in Counter(yte).items()},
        'models': {},
    }

    best_name = None
    best_acc = -1.0
    best_model = None
    labels = sorted({int(x) for x in np.unique(np.concatenate([ytr, yte]))})

    for name, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = float(accuracy_score(yte, pred))
        results['models'][name] = {
            'accuracy': acc,
            'confusion': confusion_matrix(yte, pred, labels=labels).tolist(),
            'report': classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0),
        }
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    results['best_model'] = best_name
    results['best_accuracy'] = best_acc
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    save_length_model(best_model, labels, args.model_out, feature_mode=args.feature_mode)
    print(json.dumps({'best_model': best_name, 'best_accuracy': best_acc, 'labels': labels}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
