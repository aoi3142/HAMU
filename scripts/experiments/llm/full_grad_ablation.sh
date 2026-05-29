#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-hamu-q hamu-u}"

for method in $METHODS; do
  for full_grad in 0 1; do
    METHOD="$method" FULL_GRAD="$full_grad" \
      "$(dirname "$0")/hamu_unlearn_waterdrum_tofu.sh" "$SEED"
  done
done
