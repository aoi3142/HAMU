#!/usr/bin/env bash
set -euo pipefail

FULL_GRAD=1 METHOD="${METHOD:-hamu-q}" GRADIENT_PAIRING="${GRADIENT_PAIRING:-split-gpu}" \
  "$(dirname "$0")/hamu_unlearn.sh" "$@"
