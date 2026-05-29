# HAMU: Hardness-Aware Multi-Objective Unlearning

Code for **How Hard Can It Be? Hardness-Aware Multi-Objective Unlearning**
(ICML 2026).

**Authors:** Jiangwei Chen*, Xinyuan Niu*, Rachael Hwee Ling Sim,
Zhengyuan Liu, Nancy F. Chen, Bryan Kian Hsiang Low

![Alt text](https://raw.githubusercontent.com/aoi3142/HAMU/main/images/hardness_diagram.svg "")

HAMU is a machine unlearning method that quantifies the local hardness of unlearning from the alignment between retain and forget gradients. Instead of optimizing a fixed weighted sum of objectives, HAMU solves a per-step constrained update: improve one unlearning objective by a specified amount while minimizing the cost to the other objective.

- `HAMU-Q` (`hamu-q`): enforce a minimum improvement in forget quality while minimizing retain utility degradation.
- `HAMU-U` (`hamu-u`): reciprocal variant that prioritizes the retain-side constraint.
- Supported tasks: CIFAR-10 vision experiments and WaterDrum-TOFU LLM QA experiments.
- Supported baselines: `ft`, `ga`, `gdiff`, `kl`, `scrub`, `gru`, `pcgrad`.

## Repository Layout

```text
hamu/
  cli/train.py              Unified training and unlearning CLI
  methods/                  HAMU and baseline gradient/update rules
  training/                 Trainer implementations and callbacks
  data/                     Vision and LLM dataset adapters
configs/accelerate/         Default accelerate launch config
scripts/experiments/        Paper experiment launchers
```

## Installation

```bash
pip install -e .
```

## Basic Workflow

Unlearning has two phases:

1. Train or fine-tune an original model on retain and forget data.
2. Start from that original model and unlearn with HAMU or a baseline on the retain/forget split.

By default, artifacts are written under `artifacts/`:

- CIFAR full checkpoints:
  `artifacts/checkpoints/cifar/seed_<seed>/forget_<class>_<rho>/full`
- CIFAR retain-only reference checkpoints:
  `artifacts/checkpoints/cifar/seed_<seed>/forget_<class>_<rho>/retain`
- WaterDrum-TOFU checkpoints:
  `artifacts/checkpoints/waterdrum_tofu/seed_<seed>/<subset>/<variant>`
- Unlearning outputs:
  `artifacts/results/cifar/` and `artifacts/results/waterdrum_tofu/`
- Dataset cache:
  `artifacts/dataset_cache/`

The experiment scripts enable Weights & Biases by default. Use `USE_WANDB=0` for local runs without logging.

## Quick Start: CIFAR-10

Train the full ResNet-20 model:

```bash
scripts/experiments/cifar/train_full.sh 42
```

Run HAMU-Q unlearning:

```bash
METHOD=hamu-q scripts/experiments/cifar/hamu_unlearn.sh 42
```

Run a baseline:

```bash
METHOD=gdiff scripts/experiments/cifar/baseline_unlearn.sh 42
```

Useful overrides:

```bash
USE_WANDB=0
FORGET_CLASS=0
HARD_PROBABILITY=0.5
LR=1e-4
CONSTRAINT_RATIO=0.5
NUM_GPUS=1
GRADIENT_PAIRING=sequential
MODEL_NAME=artifacts/checkpoints/cifar/seed_42/forget_0_0.0/full
```

For parameter sweeps and ablations, see:

```bash
scripts/experiments/cifar/rho_sweep.sh
scripts/experiments/cifar/constraint_sweep.sh
scripts/experiments/cifar/full_grad_ablation.sh
scripts/experiments/cifar/stopping_criterion.sh
scripts/experiments/cifar/batch_size_ablation.sh
scripts/experiments/cifar/optimizer_ablation.sh
scripts/experiments/cifar/timing_all.sh
scripts/experiments/cifar/additional_baselines.sh
```

## Quick Start: WaterDrum-TOFU LLM QA

Train the full LoRA-adapted Llama-2 checkpoint:

```bash
DATASET_NAME=Glow-AI/WaterDrum-TOFU  scripts/experiments/llm/train_full_waterdrum_tofu.sh 42
```

Run HAMU-Q unlearning:

```bash
METHOD=hamu-q scripts/experiments/llm/hamu_unlearn_waterdrum_tofu.sh 42
```

Run a baseline:

```bash
METHOD=gdiff scripts/experiments/llm/baseline_unlearn_waterdrum_tofu.sh 42
```

Useful overrides:

```bash
USE_WANDB=0
FORGET_PCT=10
DUPLICATE_SPLIT=semantic_duplicate
ADD_DUPLICATE=1
LR=1e-4
CONSTRAINT_RATIO=0.5
NUM_GPUS=1
GRADIENT_PAIRING=sequential
MODEL_NAME=artifacts/checkpoints/waterdrum_tofu/seed_42/forget_10/full
```

For parameter sweeps and ablations, see:

```bash
scripts/experiments/llm/semantic_similarity_sweep.sh
scripts/experiments/llm/constraint_sweep.sh
scripts/experiments/llm/full_grad_ablation.sh
scripts/experiments/llm/stopping_criterion.sh
scripts/experiments/llm/batch_size_ablation.sh
scripts/experiments/llm/optimizer_ablation.sh
scripts/experiments/llm/timing_all.sh
scripts/experiments/llm/additional_baselines.sh
```

## CLI Usage

All scripts call the same CLI:

```bash
python -m hamu.cli.train --help
```

Important arguments:

- `--modality {vision,llm}` selects the training stack.
- `--unlearning_method none` trains a normal model.
- `--unlearning_method hamu-q` or `hamu-u` runs HAMU.
- `--unlearning_method ft|ga|gdiff|kl|scrub|gru|pcgrad` runs a baseline.
- `--model_name` is either a Hugging Face model name or a saved checkpoint.
- `--hamu_constraint` sets the per-step HAMU constraint.
- `--use_lr_radius` sets the HAMU radius from learning rate and gradient norm.
- `--stop_on_stopping_criterion` stops HAMU once the aggregate stopping criterion is met. Without this flag, the criterion is logged only.

HAMU and paired-gradient baselines support two gradient pairing modes:

- `--gradient_pairing sequential`: single-process mode. This computes retain and forget gradients sequentially and currently requires `--gradient_accumulation_steps 1`.
- `--gradient_pairing split-gpu`: parallel mode for an even number of processes. Retain and forget batches are interleaved so alternate processes compute paired gradients in parallel.

## Notes

- `split-gpu` requires an even process count. For one GPU, set `NUM_GPUS=1 GRADIENT_PAIRING=sequential`.
- `hamu/methods/core.py` contains compact HAMU and baseline update formulas  to compare the methods without reading the trainer code.

## Citation

```bibtex
@inproceedings{chen2026icml,
  title={How hard can it be? Hardness-aware multi-objective unlearning},
  author={Chen, Jiangwei and Niu, Xinyuan and Sim, Rachael Hwee Ling and Liu, Zhengyuan and Chen, Nancy F. and Low, Bryan Kian Hsiang},
  booktitle={Proc. ICML},
  year={2026}
}
```
