#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <sentence|continuous|password> <part> <raw_subdir> [participant]"
  exit 1
fi

PROFILE="$1"
PART="$2"
RAW_SUBDIR="$3"
PARTICIPANT="${4:-p01}"

FREE_GROUPS=16
if [[ "$PROFILE" == "password" ]]; then
  FREE_GROUPS=10
fi

cd "$(dirname "$0")"

.venv/bin/python3 collector.py \
  --mode free_type \
  --prompt-profile "$PROFILE" \
  --participant "$PARTICIPANT" \
  --raw-subdir "$RAW_SUBDIR" \
  --part "$PART" \
  --free-groups "$FREE_GROUPS" \
  --free-gate-rate 150 \
  --precheck-sec 5
