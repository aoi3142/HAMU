#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-ft ga gdiff kl scrub hamu-q hamu-u}"

for method in $METHODS; do
  if [[ "$method" == hamu-* ]]; then
    METHOD="$method" \
    NUM_TRAIN_EPOCHS=1 \
    DATASET_REPEAT=1 \
    EVAL_ON_SUBSETS=0 \
    EVAL_ON_START=0 \
    EVAL_STRATEGY=no \
      "$(dirname "$0")/hamu_unlearn.sh" "$SEED"
  else
    METHOD="$method" \
    NUM_TRAIN_EPOCHS=1 \
    DATASET_REPEAT=1 \
    EVAL_ON_SUBSETS=0 \
    EVAL_ON_START=0 \
    EVAL_STRATEGY=no \
      "$(dirname "$0")/baseline_unlearn.sh" "$SEED"
  fi
done
