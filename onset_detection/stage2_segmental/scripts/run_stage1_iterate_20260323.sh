#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python3}"
QUEUE_LOG="${QUEUE_LOG:-results/stage1_iterate_20260323.queue.log}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

wait_for_clear() {
  while pgrep -f "train_eval_segment_bagrank_ctx_v2.py" >/dev/null 2>&1; do
    echo "[wait] another stage1 run is active; sleeping ${WAIT_SECONDS}s" | tee -a "$QUEUE_LOG"
    sleep "$WAIT_SECONDS"
  done
}

run_exp() {
  local out_dir="$1"
  shift
  echo "[start] $(date -Is) $out_dir" | tee -a "$QUEUE_LOG"
  "$PYTHON_BIN" onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py \
    --train_mixed_dirs data/raw/mixed_training \
    --eval_dirs data/raw/mixed_single_training data/raw/mixed_single_len9 \
    --classifier_checkpoint results/inception_password_len8_len9_len10_quick.pt \
    --classifier_scaler results/inception_password_len8_len9_len10_quick_scaler.npz \
    --overlap_checkpoint runs/stage2_overlap_candidate_finetune_v1/best_overlap.pt \
    --length_model results/length_model_len8_len9_len10_notime_mixed_cluster_v2.pkl \
    --disable_rhythm_aux \
    --output_dir "$out_dir" \
    --device auto \
    "$@" >>"$QUEUE_LOG" 2>&1
  echo "[end]   $(date -Is) $out_dir" | tee -a "$QUEUE_LOG"
}

summarize_reports() {
  "$PYTHON_BIN" - <<'PY' | tee -a "$QUEUE_LOG"
import json
from pathlib import Path

dirs = [
    Path("results/stage2_segment_bagrank_ctx_v2_keynesspool_proposerfix_no_rhythmaux_20260323_remote"),
    Path("results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_proposerfix_no_rhythmaux_20260323_remote"),
    Path("results/stage2_segment_bagrank_ctx_v2_keynesspool_thr050_no_rhythmaux_20260323_remote"),
    Path("results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr050_no_rhythmaux_20260323_remote"),
    Path("results/stage2_segment_bagrank_ctx_v2_keynesspool_thr060_no_rhythmaux_20260323_remote"),
    Path("results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr060_no_rhythmaux_20260323_remote"),
    Path("results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr050_retrytrain_no_rhythmaux_20260323_remote"),
]

rows = []
for d in dirs:
    report_path = d / "report.json"
    if not report_path.exists():
      continue
    obj = json.loads(report_path.read_text())
    overlap = obj.get("overlap_refine", {})
    oracle = obj.get("train_summary", {}).get("eval_oracle", {})
    rows.append({
        "dir": d.name,
        "candidate_mode": obj.get("candidate_mode"),
        "keyness_strong_threshold": obj.get("keyness_strong_threshold"),
        "char_top1": overlap.get("char_top1", 0.0),
        "char_top5": overlap.get("char_top5", 0.0),
        "cer": overlap.get("cer", 1.0),
        "exact_match": overlap.get("exact_match", 0.0),
        "oracle_best": oracle.get("mean_best_target", 0.0),
    })

rows.sort(
    key=lambda r: (
        -float(r["char_top1"]),
        -float(r["exact_match"]),
        -float(r["char_top5"]),
        float(r["cer"]),
        -float(r["oracle_best"]),
    )
)

print("[summary] ranked results")
for row in rows:
    print(json.dumps(row, ensure_ascii=False))
PY
}

mkdir -p results
touch "$QUEUE_LOG"

wait_for_clear
run_exp "results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_proposerfix_no_rhythmaux_20260323_remote" \
  --candidate_mode keynesspool_oldpool
run_exp "results/stage2_segment_bagrank_ctx_v2_keynesspool_thr050_no_rhythmaux_20260323_remote" \
  --candidate_mode keynesspool \
  --keyness_strong_threshold 0.50
run_exp "results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr050_no_rhythmaux_20260323_remote" \
  --candidate_mode keynesspool_oldpool \
  --keyness_strong_threshold 0.50
run_exp "results/stage2_segment_bagrank_ctx_v2_keynesspool_thr060_no_rhythmaux_20260323_remote" \
  --candidate_mode keynesspool \
  --keyness_strong_threshold 0.60
run_exp "results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr060_no_rhythmaux_20260323_remote" \
  --candidate_mode keynesspool_oldpool \
  --keyness_strong_threshold 0.60
echo "[start] $(date -Is) results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr050_retrytrain_no_rhythmaux_20260323_remote" | tee -a "$QUEUE_LOG"
"$PYTHON_BIN" onset_detection/stage2_segmental/scripts/train_eval_segment_bagrank_ctx_v2.py \
  --train_mixed_dirs data/raw/mixed_training data/raw/mixed_retry_training \
  --eval_dirs data/raw/mixed_single_training data/raw/mixed_single_len9 \
  --classifier_checkpoint results/inception_password_len8_len9_len10_quick.pt \
  --classifier_scaler results/inception_password_len8_len9_len10_quick_scaler.npz \
  --overlap_checkpoint runs/stage2_overlap_candidate_finetune_v1/best_overlap.pt \
  --length_model results/length_model_len8_len9_len10_notime_mixed_cluster_v2.pkl \
  --candidate_mode keynesspool_oldpool \
  --disable_rhythm_aux \
  --keyness_strong_threshold 0.50 \
  --output_dir results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr050_retrytrain_no_rhythmaux_20260323_remote \
  --device auto >>"$QUEUE_LOG" 2>&1
echo "[end]   $(date -Is) results/stage2_segment_bagrank_ctx_v2_keynesspool_oldpool_thr050_retrytrain_no_rhythmaux_20260323_remote" | tee -a "$QUEUE_LOG"

summarize_reports
