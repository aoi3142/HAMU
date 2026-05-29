#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts}"
SEED="${1:-42}"
FORGET_PCT="${FORGET_PCT:-10}"
SPLIT="${SPLIT:-retain}"
DATASET_NAME="${DATASET_NAME:-Glow-AI/WaterDrum-TOFU}"
SUBSET="forget_${FORGET_PCT}"
DUPLICATE_SPLIT="${DUPLICATE_SPLIT:-semantic_duplicate}"
MODEL_VARIANT="$SPLIT"
if [[ "${ADD_DUPLICATE:-0}" == "1" ]]; then
  if [[ "$DUPLICATE_SPLIT" == "semantic_duplicate" ]]; then
    MODEL_VARIANT="${SPLIT}_semanticdup"
  elif [[ "$DUPLICATE_SPLIT" == "exact_duplicate" ]]; then
    MODEL_VARIANT="${SPLIT}_exactdup"
  else
    MODEL_VARIANT="${SPLIT}_duplicate"
  fi
fi
NUM_GPUS="${NUM_GPUS:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}"

ARGS=(
  --modality llm
  --no_torch_compile
  --unlearning_method none
  --model_name "${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
  --add_lora
  --lora_r 8
  --lora_alpha 16
  --dataset_name "$DATASET_NAME"
  --dataset_subset "$SUBSET"
  --dataset_split retain
  --forget_dataset_name "$DATASET_NAME"
  --forget_dataset_subset "$SUBSET"
  --forget_dataset_split forget
  --duplicate_dataset_name "$DATASET_NAME"
  --duplicate_dataset_subset "$SUBSET"
  --duplicate_dataset_split "$DUPLICATE_SPLIT"
  --dataset_prompt_field question
  --forget_dataset_prompt_field question
  --duplicate_dataset_prompt_field question
  --dataset_response_field answer
  --forget_dataset_response_field answer
  --duplicate_dataset_response_field answer
  --shuffle_dataset
  --seed "$SEED"
  --shuffle_seed "$SEED"
  --num_train_epochs 10
  --per_device_train_batch_size 64
  --per_device_eval_batch_size 64
  --logging_strategy epoch
  --logging_steps 1
  --eval_on_start
  --eval_on_subsets
  --artifact_root "$ARTIFACT_ROOT"
  --output_dir "$ARTIFACT_ROOT/checkpoints/waterdrum_tofu/seed_${SEED}/${SUBSET}/${MODEL_VARIANT}"
  --final_model_output_dir "$ARTIFACT_ROOT/checkpoints/waterdrum_tofu/seed_${SEED}/${SUBSET}/${MODEL_VARIANT}"
)

if [[ "$SPLIT" == "full" ]]; then
  ARGS+=(--add_forget_to_train)
fi
if [[ "${ADD_DUPLICATE:-0}" == "1" ]]; then
  ARGS+=(--add_duplicate_to_retain)
fi
if [[ "${USE_WANDB:-1}" == "1" ]]; then
  ARGS+=(--use_wandb --wandb_project train_TOFU --wandb_run_name "${SEED}_${SUBSET}_${MODEL_VARIANT}")
fi

if [[ "$NUM_GPUS" == "1" ]]; then
  python -m hamu.cli.train "${ARGS[@]}"
else
  accelerate launch \
    --config_file configs/accelerate/default_config.yaml \
    --num_processes "$NUM_GPUS" \
    --main_process_port "${PORT:-31430}" \
    -m hamu.cli.train "${ARGS[@]}"
fi
