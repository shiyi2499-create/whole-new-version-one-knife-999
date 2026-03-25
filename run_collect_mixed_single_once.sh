#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <participant> <password_length: 8|9|10> [seed]"
  exit 1
fi

PARTICIPANT="$1"
PW_LEN="$2"
SEED="${3:-}"

case "$PW_LEN" in
  8|9|10) ;;
  *)
    echo "password_length must be 8, 9, or 10"
    exit 1
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="onset_detection/onset_collector.py"

CMD=(
  "$PYTHON_BIN" "$SCRIPT"
  --mode mixed_single_training
  --participant "$PARTICIPANT"
  --password-length "$PW_LEN"
  --n-trials 1
)

if [ -n "$SEED" ]; then
  CMD+=(--seed "$SEED")
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}"
