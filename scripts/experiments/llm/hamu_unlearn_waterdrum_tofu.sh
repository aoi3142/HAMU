#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
SEED="${1:-42}"
FORGET_PCT="${FORGET_PCT:-10}"
SUBSET="forget_${FORGET_PCT}"
METHOD="${METHOD:-hamu-q}"
LR="${LR:-1e-4}"
CONSTRAINT_RATIO="${CONSTRAINT_RATIO:-0.5}"
DUPLICATE_SPLIT="${DUPLICATE_SPLIT:-semantic_duplicate}"
MODEL_VARIANT="full"
if [[ "${ADD_DUPLICATE:-0}" == "1" ]]; then
  if [[ "$DUPLICATE_SPLIT" == "semantic_duplicate" ]]; then
    MODEL_VARIANT="full_semanticdup"
  elif [[ "$DUPLICATE_SPLIT" == "exact_duplicate" ]]; then
    MODEL_VARIANT="full_exactdup"
  else
    MODEL_VARIANT="full_duplicate"
  fi
fi
CONSTRAINT="$(python - <<PY
print(float("$LR") * float("$CONSTRAINT_RATIO"))
PY
)"
NUM_GPUS="${NUM_GPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}"
if [[ -n "${GRADIENT_PAIRING:-}" ]]; then
  PAIRING="$GRADIENT_PAIRING"
elif [[ "$NUM_GPUS" == "1" ]]; then
  PAIRING="sequential"
else
  PAIRING="split-gpu"
fi

MODEL_NAME="${MODEL_NAME:-$ARTIFACT_ROOT/checkpoints/waterdrum_tofu/seed_${SEED}/${SUBSET}/${MODEL_VARIANT}}"

ARGS=(
  --modality llm
  --no_torch_compile
  --unlearning_method "$METHOD"
  --gradient_pairing "$PAIRING"
  --model_name "$MODEL_NAME"
  --dataset_name "${DATASET_NAME:-Glow-AI/WaterDrum-TOFU}"
  --dataset_subset "$SUBSET"
  --dataset_split retain
  --forget_dataset_name "${DATASET_NAME:-Glow-AI/WaterDrum-TOFU}"
  --forget_dataset_subset "$SUBSET"
  --forget_dataset_split forget
  --duplicate_dataset_name "${DATASET_NAME:-Glow-AI/WaterDrum-TOFU}"
  --duplicate_dataset_subset "$SUBSET"
  --duplicate_dataset_split "$DUPLICATE_SPLIT"
  --dataset_prompt_field question
  --dataset_response_field answer
  --forget_dataset_prompt_field question
  --forget_dataset_response_field answer
  --duplicate_dataset_prompt_field question
  --duplicate_dataset_response_field answer
  --eval_on_start
  --eval_on_subsets
  --shuffle_dataset
  --seed "$SEED"
  --shuffle_seed "$SEED"
  --learning_rate "$LR"
  --hamu_constraint "$CONSTRAINT"
  --use_lr_radius
  --distribute_constraint_grad_norm
  --max_grad_norm 1.0
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}"
  --per_device_train_batch_size "${BATCH_SIZE:-50}"
  --per_device_eval_batch_size 200
  --logging_steps "${LOGGING_STEPS:-4}"
  --eval_steps "${EVAL_STEPS:-8}"
  --longer_eval_steps "${LONGER_EVAL_STEPS:-72}"
  --eval_strategy "${EVAL_STRATEGY:-no}"
  --artifact_root "$ARTIFACT_ROOT"
  --output_dir "$ARTIFACT_ROOT/results/waterdrum_tofu/hamu"
)

if [[ "${ADD_DUPLICATE:-0}" == "1" ]]; then
  ARGS+=(--add_duplicate_to_retain)
fi
if [[ "${USE_OPTIMIZER:-0}" == "1" ]]; then
  ARGS+=(--use_optimizer)
fi
if [[ "${FULL_GRAD:-0}" == "1" ]]; then
  ARGS+=(--full_grad)
fi
if [[ "${STOP_ON_STOPPING_CRITERION:-0}" == "1" ]]; then
  ARGS+=(--stop_on_stopping_criterion)
fi
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  ARGS+=(--use_wandb --wandb_project unlearn_TOFU --wandb_run_name "${SEED}_${SUBSET}_${MODEL_VARIANT}_${METHOD}_${PAIRING}_lr${LR}_c${CONSTRAINT}")
fi

if [[ "$PAIRING" == "sequential" || "$NUM_GPUS" == "1" ]]; then
  python -m hamu.cli.train "${ARGS[@]}"
else
  accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    --num_processes "$NUM_GPUS" \
    --main_process_port "${PORT:-31431}" \
    -m hamu.cli.train "${ARGS[@]}"
fi
