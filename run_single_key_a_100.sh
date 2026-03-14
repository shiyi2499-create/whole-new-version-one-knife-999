#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"

exec "$PYTHON_BIN" collector.py \
  --mode single_key \
  --raw-subdir trial_nonroot_single_key_a \
  --keys a \
  --repeats 100 \
  --single-gate-rate 190 \
  --precheck-sec 5
