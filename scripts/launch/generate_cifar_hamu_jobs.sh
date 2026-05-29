#!/usr/bin/env bash
set -euo pipefail

OUTPUT_FILE="${1:-artifacts/jobs/cifar_hamu_jobs.txt}"
mkdir -p "$(dirname "$OUTPUT_FILE")"
: > "$OUTPUT_FILE"

for seed in ${SEEDS:-41 42 43}; do
  for hard_probability in ${HARD_PROBABILITIES:-0.0 0.5 1.0}; do
    for lr in ${LEARNING_RATES:-1e-5 1e-4}; do
      for ratio in ${CONSTRAINT_RATIOS:-0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}; do
        for method in hamu-q hamu-u; do
          constraint="$(python - <<PY
print(float("$lr") * float("$ratio"))
PY
)"
          printf '%s\n' "--modality vision --unlearning_method $method --gradient_pairing split-gpu --model_name artifacts/checkpoints/cifar/seed_${seed}/forget_0_0.0/full --dataset_name uoft-cs/cifar10 --forget_dataset_name uoft-cs/cifar10 --duplicate_dataset_name uoft-cs/cifar10 --dataset_subset 0 --forget_dataset_subset 0 --duplicate_dataset_subset '' --dataset_split train --forget_dataset_split train --duplicate_dataset_split test --dataset_prompt_field img --forget_dataset_prompt_field img --duplicate_dataset_prompt_field img --dataset_response_field label --forget_dataset_response_field label --duplicate_dataset_response_field label --hard_probability $hard_probability --shuffle_dataset --seed $seed --shuffle_seed $seed --learning_rate $lr --hamu_constraint $constraint --use_lr_radius --distribute_constraint_grad_norm --num_train_epochs 50 --dataset_repeat 50 --per_device_train_batch_size 5000 --per_device_eval_batch_size 10000 --eval_on_subsets --eval_on_start --logging_strategy steps --logging_steps 1 --eval_strategy steps --eval_steps 0.02 --save_strategy no --freeze_batchnorm --artifact_root artifacts --output_dir artifacts/results/cifar/jobs" >> "$OUTPUT_FILE"
        done
      done
    done
  done
done

echo "Generated $(wc -l < "$OUTPUT_FILE") jobs in $OUTPUT_FILE"
