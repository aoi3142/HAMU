#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-hamu-q hamu-u}"
CONSTRAINT_RATIOS="${CONSTRAINT_RATIOS:-0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
HARD_PROBABILITY="${HARD_PROBABILITY:-0.0}"

for method in $METHODS; do
  for ratio in $CONSTRAINT_RATIOS; do
    METHOD="$method" CONSTRAINT_RATIO="$ratio" HARD_PROBABILITY="$HARD_PROBABILITY" \
      "$(dirname "$0")/hamu_unlearn.sh" "$SEED"
  done
done
