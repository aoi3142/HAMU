#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
SEED="${1:-42}"
METHOD="${METHOD:-gdiff}"
LR="${LR:-1e-4}"
FORGET_CLASS="${FORGET_CLASS:-0}"
HARD_PROBABILITY="${HARD_PROBABILITY:-0.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-25}"
DATASET_REPEAT="${DATASET_REPEAT:-25}"
EVAL_STRATEGY="${EVAL_STRATEGY:-steps}"
EVAL_STEPS="${EVAL_STEPS:-9}"
SAVE_STRATEGY="${SAVE_STRATEGY:-no}"
NUM_GPUS="${NUM_GPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}"
if [[ "$METHOD" == "ft" || "$METHOD" == "ga" ]]; then
  PAIRING="${GRADIENT_PAIRING:-split-gpu}"
elif [[ -n "${GRADIENT_PAIRING:-}" ]]; then
  PAIRING="$GRADIENT_PAIRING"
elif [[ "$NUM_GPUS" == "1" ]]; then
  PAIRING="sequential"
else
  PAIRING="split-gpu"
fi

MODEL_NAME="${MODEL_NAME:-$ARTIFACT_ROOT/checkpoints/cifar/seed_${SEED}/forget_${FORGET_CLASS}_0.0/full}"

WANDB_PROJECT="unlearn_cifar"

COMMON_ARGS=(
  --modality vision
  --unlearning_method "$METHOD"
  --gradient_pairing "$PAIRING"
  --model_name "$MODEL_NAME"
  --dataset_name uoft-cs/cifar10
  --forget_dataset_name uoft-cs/cifar10
  --duplicate_dataset_name uoft-cs/cifar10
  --dataset_subset "$FORGET_CLASS"
  --forget_dataset_subset "$FORGET_CLASS"
  --duplicate_dataset_subset ""
  --dataset_split train
  --forget_dataset_split train
  --duplicate_dataset_split test
  --dataset_prompt_field img
  --forget_dataset_prompt_field img
  --duplicate_dataset_prompt_field img
  --dataset_response_field label
  --forget_dataset_response_field label
  --duplicate_dataset_response_field label
  --hard_probability "$HARD_PROBABILITY"
  --shuffle_dataset
  --seed "$SEED"
  --shuffle_seed "$SEED"
  --learning_rate "$LR"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --dataset_repeat "$DATASET_REPEAT"
  --per_device_train_batch_size "${BATCH_SIZE:-5000}"
  --per_device_eval_batch_size 10000
  --logging_strategy steps
  --logging_steps 1
  --eval_strategy "$EVAL_STRATEGY"
  --eval_steps "$EVAL_STEPS"
  --save_strategy "$SAVE_STRATEGY"
  --freeze_batchnorm
  --artifact_root "$ARTIFACT_ROOT"
  --output_dir "$ARTIFACT_ROOT/results/cifar/baselines"
)

if [[ "${EVAL_ON_SUBSETS:-1}" == "1" ]]; then
  COMMON_ARGS+=(--eval_on_subsets)
fi
if [[ "${EVAL_ON_START:-1}" == "1" ]]; then
  COMMON_ARGS+=(--eval_on_start)
fi
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  COMMON_ARGS+=(--use_wandb --wandb_project "$WANDB_PROJECT" --wandb_run_name "${SEED}_${FORGET_CLASS}_${HARD_PROBABILITY}_${METHOD}_lr${LR}")
fi

if [[ "$NUM_GPUS" == "1" ]]; then
  python -m hamu.cli.train "${COMMON_ARGS[@]}"
else
  accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    --num_processes "$NUM_GPUS" \
    --main_process_port "${PORT:-31422}" \
    -m hamu.cli.train "${COMMON_ARGS[@]}"
fi
