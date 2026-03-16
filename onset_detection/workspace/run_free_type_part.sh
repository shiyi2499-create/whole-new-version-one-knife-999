#!/bin/zsh
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <part-number>"
  exit 1
fi

PART="$1"
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"

exec "$PYTHON_BIN" collector.py \
  --mode free_type \
  --raw-subdir trial_nonroot_free_type_refill \
  --part "$PART" \
  --free-groups 16 \
  --free-gate-rate 150 \
  --precheck-sec 5
