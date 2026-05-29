#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHOD="${METHOD:-hamu-u}"
CONSTRAINT_RATIO="${CONSTRAINT_RATIO:-0.9}"
HARD_PROBABILITY="${HARD_PROBABILITY:-0.0}"

for use_optimizer in 0 1; do
  METHOD="$METHOD" \
  CONSTRAINT_RATIO="$CONSTRAINT_RATIO" \
  HARD_PROBABILITY="$HARD_PROBABILITY" \
  USE_OPTIMIZER="$use_optimizer" \
    "$(dirname "$0")/hamu_unlearn.sh" "$SEED"
done
