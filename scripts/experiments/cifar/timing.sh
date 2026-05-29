#!/usr/bin/env bash
set -euo pipefail

USE_WANDB="${USE_WANDB:-1}" \
HARD_PROBABILITY="${HARD_PROBABILITY:-0.0}" \
LR="${LR:-1e-4}" \
CONSTRAINT_RATIO="${CONSTRAINT_RATIO:-0.5}" \
BATCH_SIZE="${BATCH_SIZE:-5000}" \
"$(dirname "$0")/hamu_unlearn.sh" "$@"
