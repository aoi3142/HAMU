#!/usr/bin/env bash
set -euo pipefail

JOBS_FILE="${1:?Usage: scripts/launch/run_jobs.sh JOBS_FILE [MAX_PROCESSES]}"
MAX_PROCESSES="${2:-3}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
NUM_GPUS="${NUM_GPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}"

if [[ ! -f "$JOBS_FILE" ]]; then
  echo "Jobs file not found: $JOBS_FILE" >&2
  exit 1
fi

i=0
while IFS= read -r args_line || [[ -n "$args_line" ]]; do
  [[ -z "$args_line" ]] && continue
  if [[ "$NUM_GPUS" == "1" ]]; then
    cmd=(python -m hamu.cli.train)
  else
    cmd=(
      accelerate launch
      --config_file configs/accelerate/default_config.yaml
      --num_processes "$NUM_GPUS"
      --main_process_port "5${PORT_PREFIX:-42}$((i % MAX_PROCESSES))"
      -m hamu.cli.train
    )
  fi
  while [[ "$(jobs -r | wc -l)" -ge "$MAX_PROCESSES" ]]; do
    sleep 1
  done
  # shellcheck disable=SC2086
  ARTIFACT_ROOT="$ARTIFACT_ROOT" "${cmd[@]}" $args_line &
  i=$((i + 1))
done < "$JOBS_FILE"

wait
