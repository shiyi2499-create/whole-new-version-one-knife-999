#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-/home/shiyi/备份（mac_vs专用）}"
RUNS="$BASE/results/stage1_exp5_new_retry_27fold_runs_20260324"
SPLITS="$BASE/data/raw/stage1_exp5_new_retry_27fold_splits_20260324"
OUTROOT="$BASE/results/stage1_exp5_new_retry_27fold_posthoc_completefirst_20260325"
SCRIPT="$BASE/onset_detection/stage2_segmental/scripts/recompute_posthoc_completehit.py"
MAIN="$BASE/onset_detection/stage2_segmental/scripts/train_eval_stage1_dense_labeling.py"

mkdir -p "$OUTROOT"

for i in $(seq -w 1 27); do
  fold="fold${i}"
  echo "[$(date '+%F %T')] start $fold"
  python3 "$SCRIPT" \
    --main_script "$MAIN" \
    --checkpoint "$RUNS/$fold/best_dense_labeling.pt" \
    --eval_dir "$SPLITS/$fold/eval" \
    --output_dir "$OUTROOT/$fold" \
    --device cuda \
    > "$OUTROOT/$fold.stdout.log" \
    2> "$OUTROOT/$fold.stderr.log"
  echo "[$(date '+%F %T')] done $fold"
done
