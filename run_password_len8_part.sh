#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <part:1-10> [participant]"
  exit 1
fi

PART="$1"
PARTICIPANT="${2:-p01}"

cd "$(dirname "$0")"

./run_text_part.sh password "$PART" password/len_8 "$PARTICIPANT"
