#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-hamu-q hamu-u}"
CONSTRAINT_RATIO="${CONSTRAINT_RATIO:-0.5}"

for method in $METHODS; do
  METHOD="$method" \
  CONSTRAINT_RATIO="$CONSTRAINT_RATIO" \
  STOP_ON_STOPPING_CRITERION="${STOP_ON_STOPPING_CRITERION:-1}" \
    "$(dirname "$0")/hamu_unlearn_waterdrum_tofu.sh" "$SEED"
done
