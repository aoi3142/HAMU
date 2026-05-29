#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-hamu-q}"
BATCH_SIZES="${BATCH_SIZES:-16 32 50 64 128}"

for method in $METHODS; do
  for batch_size in $BATCH_SIZES; do
    METHOD="$method" BATCH_SIZE="$batch_size" \
      "$(dirname "$0")/hamu_unlearn_waterdrum_tofu.sh" "$SEED"
  done
done
