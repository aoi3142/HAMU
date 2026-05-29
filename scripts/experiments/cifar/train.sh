#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
SEED="${1:-42}"
FORGET_CLASS="${FORGET_CLASS:-0}"
HARD_PROBABILITY="${HARD_PROBABILITY:-0.0}"
SPLIT="${SPLIT:-retain}"
NUM_GPUS="${NUM_GPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}"

COMMON_ARGS=(
  --modality vision
  --unlearning_method none
  --model_name resnet20-cifar
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
  --split_ratio 0.1
  --hard_probability "$HARD_PROBABILITY"
  --shuffle_dataset
  --seed "$SEED"
  --shuffle_seed "$SEED"
  --learning_rate 1e-3
  --num_train_epochs 50
  --dataset_repeat 50
  --per_device_train_batch_size 5000
  --per_device_eval_batch_size 10000
  --eval_on_subsets
  --logging_strategy steps
  --logging_steps 1
  --eval_strategy steps
  --eval_steps 0.1
  --save_strategy steps
  --save_steps 0.1
  --select_best_model_by_subset_accuracy
  --artifact_root "$ARTIFACT_ROOT"
  --output_dir "$ARTIFACT_ROOT/checkpoints/cifar/seed_${SEED}/forget_${FORGET_CLASS}_${HARD_PROBABILITY}/${SPLIT}"
  --final_model_output_dir "$ARTIFACT_ROOT/checkpoints/cifar/seed_${SEED}/forget_${FORGET_CLASS}_${HARD_PROBABILITY}/${SPLIT}"
)

if [[ "$SPLIT" == "full" ]]; then
  COMMON_ARGS+=(--add_forget_to_train)
fi
if [[ "${FREEZE_BATCHNORM:-0}" == "1" ]]; then
  COMMON_ARGS+=(--freeze_batchnorm)
fi

if [[ "${USE_WANDB:-1}" == "1" ]]; then
  COMMON_ARGS+=(--use_wandb --wandb_project train_cifar2 --wandb_run_name "${SEED}_${FORGET_CLASS}_${HARD_PROBABILITY}_${SPLIT}")
fi

if [[ "$NUM_GPUS" == "1" ]]; then
  python -m hamu.cli.train "${COMMON_ARGS[@]}"
else
  accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    --num_processes "$NUM_GPUS" \
    --main_process_port "${PORT:-31420}" \
    -m hamu.cli.train "${COMMON_ARGS[@]}"
fi
