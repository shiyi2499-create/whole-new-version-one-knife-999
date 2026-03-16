#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"

exec "$PYTHON_BIN" collector.py \
  --mode free_type \
  --raw-subdir trial_nonroot_free_type_part1 \
  --part 1 \
  --free-groups 16 \
  --free-gate-rate 150 \
  --precheck-sec 5
