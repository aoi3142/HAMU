#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-ft ga gdiff kl scrub gru pcgrad hamu-q hamu-u}"

for method in $METHODS; do
  if [[ "$method" == hamu-* ]]; then
    METHOD="$method" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}" \
      "$(dirname "$0")/hamu_unlearn_waterdrum_tofu.sh" "$SEED"
  else
    METHOD="$method" \
    NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}" \
      "$(dirname "$0")/baseline_unlearn_waterdrum_tofu.sh" "$SEED"
  fi
done
