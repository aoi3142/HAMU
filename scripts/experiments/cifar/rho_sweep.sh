#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-42}"
HAMU_METHODS="${HAMU_METHODS:-hamu-q hamu-u}"
BASELINE_METHODS="${BASELINE_METHODS:-ft ga gdiff kl scrub}"
HARD_PROBABILITIES="${HARD_PROBABILITIES:-0.0 0.25 0.5 0.75 1.0}"

for rho in $HARD_PROBABILITIES; do
  for method in $HAMU_METHODS; do
    HARD_PROBABILITY="$rho" METHOD="$method" "$(dirname "$0")/hamu_unlearn.sh" "$SEED"
  done
  for method in $BASELINE_METHODS; do
    HARD_PROBABILITY="$rho" METHOD="$method" "$(dirname "$0")/baseline_unlearn.sh" "$SEED"
  done
done
