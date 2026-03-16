#!/bin/zsh
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <key> <variant-tag> [repeats]"
  echo "Example: $0 r left_middle 80"
  exit 1
fi

KEY="$1"
VARIANT_TAG="$2"
REPEATS="${3:-80}"

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
RAW_SUBDIR="trial_nonroot_single_key_boost_${KEY}_${VARIANT_TAG}"

exec "$PYTHON_BIN" collector.py \
  --mode single_key \
  --raw-subdir "$RAW_SUBDIR" \
  --keys "$KEY" \
  --repeats "$REPEATS" \
  --single-gate-rate 190 \
  --precheck-sec 5
