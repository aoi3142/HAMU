#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
HAMU_METHODS="${HAMU_METHODS:-hamu-q hamu-u}"
BASELINE_METHODS="${BASELINE_METHODS:-ft ga gdiff kl scrub}"

for add_duplicate in 0 1; do
  for method in $HAMU_METHODS; do
    ADD_DUPLICATE="$add_duplicate" METHOD="$method" "$(dirname "$0")/hamu_unlearn_waterdrum_tofu.sh" "$SEED"
  done
  for method in $BASELINE_METHODS; do
    ADD_DUPLICATE="$add_duplicate" METHOD="$method" "$(dirname "$0")/baseline_unlearn_waterdrum_tofu.sh" "$SEED"
  done
done
