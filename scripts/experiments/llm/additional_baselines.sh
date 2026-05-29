#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-gru pcgrad ft ga gdiff kl scrub}"

for method in $METHODS; do
  METHOD="$method" "$(dirname "$0")/baseline_unlearn_waterdrum_tofu.sh" "$SEED"
done
