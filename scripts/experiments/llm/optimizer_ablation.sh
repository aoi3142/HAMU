#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
METHODS="${METHODS:-hamu-q hamu-u}"

for method in $METHODS; do
  for use_optimizer in 0 1; do
    METHOD="$method" USE_OPTIMIZER="$use_optimizer" \
      "$(dirname "$0")/hamu_unlearn_waterdrum_tofu.sh" "$SEED"
  done
done
