#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
BATCH_SIZES="${BATCH_SIZES:-128 256 512 1024 2048}"
METHOD="${METHOD:-hamu-q}"
HARD_PROBABILITY="${HARD_PROBABILITY:-0.0}"

for batch_size in $BATCH_SIZES; do
  METHOD="$METHOD" HARD_PROBABILITY="$HARD_PROBABILITY" BATCH_SIZE="$batch_size" \
    "$(dirname "$0")/hamu_unlearn.sh" "$SEED"
done
