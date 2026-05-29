"""Unified training and unlearning CLI for HAMU."""

from __future__ import annotations

import argparse
import atexit
import gc
import os
import random
from functools import partial
from itertools import cycle, islice
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Sequence
from datasets import concatenate_datasets

import torch

if TYPE_CHECKING:
    from datasets import Dataset


HAMU_METHODS = {"hamu-q", "hamu-u"}
BASELINE_METHODS = {"ft", "ga", "gdiff", "kl", "scrub", "gru", "pcgrad"}
METHOD_CHOICES = ["none", "hamu-q", "hamu-u", "ft", "ga", "gdiff", "kl", "scrub", "gru", "pcgrad"]
PAIRWISE_METHODS = HAMU_METHODS | {"gdiff", "kl", "scrub", "gru", "pcgrad"}
SEQUENTIAL_METHODS = PAIRWISE_METHODS


def batched(iterable: Iterable[Any], n: int) -> Iterator[tuple[Any, ...]]:
    iterator = iter(iterable)
    while True:
        batch = tuple(islice(iterator, n))
        if not batch:
            return
        yield batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train, fine-tune, or unlearn with HAMU.")

    parser.add_argument("--modality", choices=["vision", "llm"], required=True)
    parser.add_argument(
        "--unlearning_method",
        choices=METHOD_CHOICES,
        default="none",
    )
    parser.add_argument(
        "--gradient_pairing",
        choices=["split-gpu", "sequential"],
        default="split-gpu" if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) >= 2 else "sequential",
        help="How paired retain/forget gradients are produced for HAMU and paired baselines.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num_labels", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="artifacts/results")
    parser.add_argument("--final_model_output_dir", type=str, default=None)
    parser.add_argument("--artifact_root", type=str, default="artifacts")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)

    parser.add_argument("--split_ratio", type=float, default=0.1)
    parser.add_argument("--hard_probability", type=float, default=0.0)
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--shuffle_dataset", action="store_true", default=False)

    parser.add_argument("--dataset_name", type=str, default="locuslab/TOFU")
    parser.add_argument("--dataset_subset", type=str, default="full")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--dataset_prompt_field", type=str, default="question")
    parser.add_argument("--dataset_response_field", type=str, default="answer")

    parser.add_argument("--forget_dataset_name", type=str, default="locuslab/TOFU")
    parser.add_argument("--forget_dataset_subset", type=str, default="full")
    parser.add_argument("--forget_dataset_split", type=str, default="train")
    parser.add_argument("--forget_dataset_prompt_field", type=str, default="question")
    parser.add_argument("--forget_dataset_response_field", type=str, default="answer")

    parser.add_argument("--duplicate_dataset_name", type=str, default="locuslab/TOFU")
    parser.add_argument("--duplicate_dataset_subset", type=str, default="full")
    parser.add_argument("--duplicate_dataset_split", type=str, default="train")
    parser.add_argument("--duplicate_dataset_prompt_field", type=str, default="question")
    parser.add_argument("--duplicate_dataset_response_field", type=str, default="answer")

    parser.add_argument("--test_dataset_name", type=str, default="locuslab/TOFU")
    parser.add_argument("--test_dataset_subset", type=str, default="holdout10")
    parser.add_argument("--test_dataset_split", type=str, default="train")
    parser.add_argument("--test_dataset_prompt_field", type=str, default="question")
    parser.add_argument("--test_dataset_response_field", type=str, default="answer")
    parser.add_argument("--k_pct", type=float, default=0.1)

    parser.add_argument("--add_forget_to_train", action="store_true", default=False)
    parser.add_argument("--add_duplicate_to_retain", action="store_true", default=False)
    parser.add_argument("--eval_on_subsets", action="store_true", default=False)
    parser.add_argument("--eval_on_forget", action="store_true", default=False)
    parser.add_argument(
        "--select_best_model_by_subset_accuracy",
        action="store_true",
        default=False,
        help="For CV training, load the checkpoint with the best mean eval retain/forget accuracy before final save.",
    )

    parser.add_argument("--add_lora", action="store_true", default=False)
    parser.add_argument("--lora_name", type=str, default="added")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--hamu_constraint", type=float, default=0.0)
    parser.add_argument("--hamu_radius", type=float, default=None)
    parser.add_argument("--use_lr_radius", action="store_true", default=False)
    parser.add_argument("--continue_on_failed_feasibility", action="store_true", default=False)
    parser.add_argument(
        "--stop_on_stopping_criterion",
        action="store_true",
        default=False,
        help=(
            "Stop HAMU when the aggregate stopping criterion from the paper is met. "
            "By default the criterion is logged only and training proceeds to max epochs."
        ),
    )
    parser.add_argument("--stop_on_loss_thresholds", action="store_true", default=False)
    parser.add_argument("--retain_loss_stop_threshold", type=float, default=0.4)
    parser.add_argument("--forget_loss_stop_threshold", type=float, default=0.1)
    parser.add_argument("--full_grad", action="store_true", default=False)
    parser.add_argument("--distribute_constraint_softmax_dp", action="store_true", default=False)
    parser.add_argument("--distribute_constraint_grad_norm", action="store_true", default=False)
    parser.add_argument("--no_normalize_distributed_constraint", action="store_true", default=False)
    parser.add_argument("--use_optimizer", action="store_true", default=False)
    parser.add_argument("--freeze_batchnorm", action="store_true", default=False)
    parser.add_argument("--scrub_alpha", type=float, default=0.001)
    parser.add_argument("--scrub_gamma", type=float, default=0.99)

    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=float, default=0.25)
    parser.add_argument("--logging_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=float, default=None)
    parser.add_argument("--save_strategy", type=str, default="no")
    parser.add_argument("--eval_steps", type=float, default=None)
    parser.add_argument("--longer_eval_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--eval_on_start", action="store_true", default=False)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", default=False)
    parser.add_argument("--torch_empty_cache_steps", type=int, default=5)
    parser.add_argument("--no_torch_compile", action="store_true", default=False)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="hamu")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_notes", type=str, default=None)

    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dry_run", action="store_true", default=False)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    method = args.unlearning_method
    if args.dataset_cache_dir is None:
        args.dataset_cache_dir = str(Path(args.artifact_root) / "dataset_cache")

    if method in PAIRWISE_METHODS and args.gradient_pairing == "sequential":
        if world_size != 1:
            raise ValueError("--gradient_pairing sequential requires a single process.")
        if args.gradient_accumulation_steps != 1:
            raise ValueError("--gradient_pairing sequential currently requires --gradient_accumulation_steps 1.")
    elif method in PAIRWISE_METHODS:
        if world_size % 2 != 0 and not args.debug:
            raise ValueError("split-gpu gradient pairing requires an even number of processes.")

    if method.startswith("hamu"):
        radius = args.learning_rate * args.max_grad_norm if args.use_lr_radius else args.hamu_radius
        if radius is None:
            raise ValueError("--hamu_radius is required unless --use_lr_radius is set.")
        if args.hamu_constraint > radius * args.max_grad_norm:
            raise ValueError(
                f"hamu_constraint ({args.hamu_constraint}) must be <= radius * max_grad_norm "
                f"({radius * args.max_grad_norm})."
            )


def maybe_interlace_datasets(
    args: argparse.Namespace,
    train_dataset: Dataset,
    forget_dataset: Dataset,
    processor: Any,
    get_pad_dataset: Callable[..., Dataset],
) -> tuple[Dataset, int]:
    if args.unlearning_method not in PAIRWISE_METHODS or args.gradient_pairing != "split-gpu":
        return train_dataset, 0
    if args.unlearning_method in {"ft", "ga"}:
        return train_dataset, 0

    batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if batch_size == -1:
        batch_size = len(train_dataset) // int(os.environ.get("WORLD_SIZE", "1"))
    train_indices = range(len(train_dataset))
    forget_indices = range(len(forget_dataset))
    train_batched = (train_dataset.select(batch) for batch in batched(train_indices, batch_size))
    forget_batched = (forget_dataset.select(batch) for batch in batched(cycle(forget_indices), batch_size))
    interlaced_dataset = []
    dataset_pad = 0
    for train_batch, forget_batch in zip(train_batched, forget_batched):
        if len(train_batch) < batch_size:
            dataset_pad = batch_size - len(train_batch)
            pad_dataset = get_pad_dataset(dataset_pad, processor=processor, features=train_batch.features)
            train_batch = concatenate_datasets([train_batch, pad_dataset])
            forget_batch = concatenate_datasets([forget_batch.select(range(len(forget_batch) - dataset_pad)), pad_dataset])
        interlaced_dataset.append(train_batch)
        interlaced_dataset.append(forget_batch)
    return concatenate_datasets(interlaced_dataset), dataset_pad


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    os.environ["HAMU_MODALITY"] = args.modality
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from peft import PeftModelForCausalLM
    from transformers import TrainingArguments
    from trl.trainer.sft_config import SFTConfig

    from hamu.methods import GRUOptimizer, HAMUOptimizer, PCGradOptimizer
    from hamu.methods.hamu import GradientTransformOptimizer
    from hamu.training.callbacks import FractionalLoggingCallback, NaNStoppingCallback, RetainForgetAccuracyCallback
    from hamu.training.trainer import (
        BaseUnlearningTrainer,
        GATrainer,
        FTTrainer,
        GDiffTrainer,
        HAMUTrainer,
        KLTrainer,
        PairedGradientTrainer,
        SCRUBTrainer,
        ThresholdStoppingCallback,
    )

    if args.modality == "llm":
        from hamu.data import llm as data_module
    else:
        from hamu.data import vision as data_module

    unlearning = args.unlearning_method != "none"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.final_model_output_dir is not None:
        Path(args.final_model_output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.dataset_cache_dir).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        args.use_wandb = False
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
        if args.wandb_tags:
            os.environ["WANDB_TAGS"] = ",".join(args.wandb_tags)
        if args.wandb_notes:
            os.environ["WANDB_NOTES"] = args.wandb_notes

    load_original = args.unlearning_method in {"scrub", "kl"}
    if args.modality == "vision":
        load_model_and_processor_kwargs = {
            "num_labels": args.num_labels,
            "size": args.img_size,
            "freeze_batchnorm": args.freeze_batchnorm,
        }
    else:
        load_model_and_processor_kwargs = {}
    model, processor = data_module.load_model_and_processor(
        model_name=args.model_name,
        seed=args.seed,
        add_lora=args.add_lora,
        lora_name=args.lora_name,
        load_original=load_original,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        **load_model_and_processor_kwargs
    )

    dataset, duplicate_dataset, forget_dataset = data_module.load_datasets(processor, **vars(args))
    if args.modality == "llm":
        cache = Path(args.dataset_cache_dir)
        dataset = dataset.map(
            partial(data_module.preprocess_function, processor, args.dataset_prompt_field, args.dataset_response_field),
            remove_columns=dataset.column_names,
            load_from_cache_file=True,
            cache_file_name=str(cache / f"retain_{args.dataset_name.replace('/', '_')}_{args.dataset_subset}_{args.dataset_split}.arrow"),
        )
        if duplicate_dataset is not None and len(duplicate_dataset) > 0:
            duplicate_dataset = duplicate_dataset.map(
                partial(
                    data_module.preprocess_function,
                    processor,
                    args.duplicate_dataset_prompt_field,
                    args.duplicate_dataset_response_field,
                ),
                remove_columns=duplicate_dataset.column_names,
                load_from_cache_file=True,
                cache_file_name=str(cache / f"duplicate_{args.duplicate_dataset_name.replace('/', '_')}_{args.duplicate_dataset_subset}_{args.duplicate_dataset_split}.arrow"),
            )
        forget_dataset = forget_dataset.map(
            partial(data_module.preprocess_function, processor, args.forget_dataset_prompt_field, args.forget_dataset_response_field),
            remove_columns=forget_dataset.column_names,
            load_from_cache_file=True,
            cache_file_name=str(cache / f"forget_{args.forget_dataset_name.replace('/', '_')}_{args.forget_dataset_subset}_{args.forget_dataset_split}.arrow"),
        )
        test_dataset = data_module.my_load_dataset(args.test_dataset_name, args.test_dataset_subset, args.test_dataset_split)
        test_dataset = test_dataset.map(
            partial(data_module.preprocess_function, processor, args.test_dataset_prompt_field, args.test_dataset_response_field),
            remove_columns=test_dataset.column_names,
            load_from_cache_file=True,
            cache_file_name=str(cache / f"test_{args.test_dataset_name.replace('/', '_')}_{args.test_dataset_subset}_{args.test_dataset_split}.arrow"),
        )
    else:
        test_dataset = duplicate_dataset

    retain_dataset = concatenate_datasets([dataset, duplicate_dataset]) if args.add_duplicate_to_retain else dataset
    if args.unlearning_method == "ga":
        train_dataset = forget_dataset
    elif args.unlearning_method == "ft":
        train_dataset = retain_dataset
    elif args.unlearning_method in PAIRWISE_METHODS and args.gradient_pairing == "sequential":
        train_dataset = retain_dataset
    elif args.add_forget_to_train:
        train_dataset = concatenate_datasets([retain_dataset, forget_dataset])
    else:
        train_dataset = retain_dataset

    if args.shuffle_dataset:
        train_indices = list(range(len(train_dataset)))
        forget_indices = list(range(len(forget_dataset)))
        random.seed(args.shuffle_seed)
        random.shuffle(train_indices)
        random.shuffle(forget_indices)
        train_dataset = train_dataset.select(train_indices)
        forget_dataset = forget_dataset.select(forget_indices)

    train_dataset, dataset_pad = maybe_interlace_datasets(
        args,
        train_dataset,
        forget_dataset,
        processor,
        data_module.get_pad_dataset,
    )

    if args.unlearning_method == "ft":
        train_dataset = concatenate_datasets([train_dataset, train_dataset])
    if args.unlearning_method == "ga":
        copies = max(1, len(retain_dataset) // max(len(forget_dataset), 1))
        train_dataset = concatenate_datasets([forget_dataset, forget_dataset] * copies)
    if args.dataset_repeat > 1:
        train_dataset = concatenate_datasets([train_dataset] * args.dataset_repeat)
    if args.debug:
        train_dataset = train_dataset.select(range(min(32, len(train_dataset))))

    per_device_train_batch_size = args.per_device_train_batch_size
    if per_device_train_batch_size == -1:
        per_device_train_batch_size = len(train_dataset) // (
            int(os.environ.get("WORLD_SIZE", "1")) * args.gradient_accumulation_steps
        )

    training_args_dict = {
        "seed": args.seed,
        "output_dir": args.output_dir,
        "num_train_epochs": max(1, args.num_train_epochs // max(args.dataset_repeat, 1)),
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "logging_steps": args.logging_steps if args.logging_steps >= 1 else 1,
        "logging_strategy": args.logging_strategy,
        "eval_steps": args.eval_steps,
        "eval_strategy": args.eval_strategy if (args.eval_on_forget or args.eval_on_subsets) else "no",
        "eval_on_start": args.eval_on_start,
        "save_steps": args.save_steps,
        "save_strategy": args.save_strategy,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "fp16": args.fp16 and torch.cuda.is_available(),
        "bf16": args.bf16 and torch.cuda.is_available(),
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "report_to": "wandb" if args.use_wandb else "none",
        "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
        "torch_empty_cache_steps": args.torch_empty_cache_steps,
        "remove_unused_columns": False,
        "torch_compile": not args.no_torch_compile,
        "train_sampling_strategy": "sequential",
        "batch_eval_metrics": True,
    }
    callbacks = []
    if args.select_best_model_by_subset_accuracy:
        if args.modality != "vision":
            raise ValueError("--select_best_model_by_subset_accuracy is currently implemented for vision experiments.")
        if not args.eval_on_subsets:
            raise ValueError("--select_best_model_by_subset_accuracy requires --eval_on_subsets.")
        training_args_dict.update(
            {
                "load_best_model_at_end": True,
                "metric_for_best_model": RetainForgetAccuracyCallback.metric_name,
                "greater_is_better": True,
            }
        )
        callbacks.append(RetainForgetAccuracyCallback())
    if args.modality == "llm":
        training_args = SFTConfig(**training_args_dict, completion_only_loss=True, max_length=None)
    else:
        if not args.debug:
            num_workers = (os.cpu_count() or 8) // 8 - 1
            if num_workers > 1:
                training_args_dict |= {
                    "dataloader_num_workers": num_workers,
                    "dataloader_prefetch_factor": 4,
                    "dataloader_pin_memory": True,
                }
        training_args = TrainingArguments(**training_args_dict)

    base_optimizer = None
    if unlearning and args.use_optimizer:
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    optimizer = None
    method = args.unlearning_method.lower()
    if method in {"hamu-q", "hamu-u"}:
        optimizer = HAMUOptimizer(
            model,
            optimizer=base_optimizer,
            lr=args.learning_rate if args.use_lr_radius else None,
            radius=args.hamu_radius,
            constraint=args.hamu_constraint,
            variant=method,
            stop_on_failed_feasibility=not args.continue_on_failed_feasibility,
            stop_on_stopping_criterion=args.stop_on_stopping_criterion,
            full_grad=args.full_grad,
            distribute_constraint_softmax_dp=args.distribute_constraint_softmax_dp,
            distribute_constraint_grad_norm=args.distribute_constraint_grad_norm,
            normalize_distributed_constraint=not args.no_normalize_distributed_constraint,
            gradient_pairing=args.gradient_pairing,
        )
    elif method == "gru":
        optimizer = GRUOptimizer(model, optimizer=base_optimizer, lr=args.learning_rate, gradient_pairing=args.gradient_pairing)
    elif method == "pcgrad":
        optimizer = PCGradOptimizer(model, optimizer=base_optimizer, lr=args.learning_rate, gradient_pairing=args.gradient_pairing)
    elif method in {"ft", "ga"}:
        if base_optimizer is None:   # optimizer==None defaults to AdamW in Trainer, so we explicitly create an SGD optimizer for FT and GA
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.0)
        else:
            optimizer = base_optimizer
    elif method in {"gdiff", "kl", "scrub"}:
        optimizer = GradientTransformOptimizer(
            model,
            defaults={"lr": args.learning_rate},
            optimizer=base_optimizer,
            gradient_pairing=args.gradient_pairing,
        )

    trainer_class: type[BaseUnlearningTrainer]
    trainer_class = {
        "none": BaseUnlearningTrainer,
        "hamu-q": HAMUTrainer,
        "hamu-u": HAMUTrainer,
        "ft": FTTrainer,
        "ga": GATrainer,
        "gdiff": GDiffTrainer,
        "kl": KLTrainer,
        "scrub": SCRUBTrainer,
        "gru": PairedGradientTrainer,
        "pcgrad": PairedGradientTrainer,
    }[method]

    nan_callback = NaNStoppingCallback()
    callbacks.append(nan_callback)
    if unlearning:
        callbacks.append(
            ThresholdStoppingCallback(
                stop_on_failed_feasibility=not args.continue_on_failed_feasibility,
                stop_on_stopping_criterion=args.stop_on_stopping_criterion,
                stop_on_loss_thresholds=args.stop_on_loss_thresholds,
                retain_loss_threshold=args.retain_loss_stop_threshold,
                forget_loss_threshold=args.forget_loss_stop_threshold,
            )
        )
        callbacks.append(
            FractionalLoggingCallback(
                logging_interval=args.logging_steps,
                eval_interval=args.eval_steps,
                longer_eval_interval=args.longer_eval_steps,
            )
        )

    if args.eval_on_subsets:
        eval_dataset = {"retain": retain_dataset, "forget": forget_dataset, "test": test_dataset}
        if args.modality == "llm" and duplicate_dataset is not None and len(duplicate_dataset) > 0:
            eval_dataset["duplicate"] = duplicate_dataset
    elif args.eval_on_forget:
        eval_dataset = forget_dataset
    else:
        eval_dataset = None

    trainer_kwargs = {
        "gradient_pairing": args.gradient_pairing if method in PAIRWISE_METHODS else "none",
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": training_args,
        "optimizers": (optimizer, None),
        "callbacks": callbacks,
        "debug": args.debug,
        "processing_class": processor,
        "k_pct": args.k_pct,
        "forget_dataset_for_pairing": forget_dataset if args.gradient_pairing == "sequential" else None,
    }
    if trainer_class is SCRUBTrainer:
        trainer_kwargs["alpha"] = args.scrub_alpha
        trainer_kwargs["gamma"] = args.scrub_gamma
    trainer = trainer_class(**trainer_kwargs)

    if unlearning and isinstance(optimizer, GradientTransformOptimizer):
        optimizer.set_accelerator(trainer.accelerator)
    nan_callback.accelerator = trainer.accelerator
    if args.modality == "vision" and load_original:
        trainer.update_frozen_model()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run = None
    if trainer.is_world_process_zero():
        print("Training arguments:", args)
        if isinstance(model, PeftModelForCausalLM):
            model.print_trainable_parameters()
        effective_len = (len(train_dataset) - dataset_pad * (1 + int(method in PAIRWISE_METHODS))) // max(args.dataset_repeat, 1)
        print(f"Dataset size: {effective_len} (padded to {len(train_dataset) // max(args.dataset_repeat, 1)})")
        print("Starting training...")
        if args.use_wandb:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                tags=args.wandb_tags,
                notes=args.wandb_notes,
            )
            forget_pct = len(forget_dataset) / (len(dataset) + len(forget_dataset))
            wandb.config.update(
                {f"command/{k}": v for k, v in vars(args).items()}
                | {
                    "training_dataset_len": len(train_dataset),
                    "retain_dataset_len": len(dataset),
                    "forget_dataset_len": len(forget_dataset),
                    "duplicate_dataset_len": len(duplicate_dataset) if duplicate_dataset is not None else 0,
                    "forget_pct": int(round(100 * forget_pct)),
                    "ratio_b_lr": round(args.hamu_constraint / args.learning_rate, 8),
                    "ratio_b_r": round(args.hamu_constraint / args.hamu_radius, 8)
                    if args.hamu_radius is not None
                    else None,
                    "method": method if method != "none" else None,
                    "with_stopping": args.stop_on_stopping_criterion,
                },
                allow_val_change=True,
            )
            wandb.save(__file__)

    start_time = time()
    if not args.dry_run:
        trainer.train()
    training_time = time() - start_time

    if trainer.is_world_process_zero():
        print(f"Training completed in {training_time:.2f} seconds.")
        if args.use_wandb and run is not None:
            run.summary["training_time_seconds"] = training_time
            if torch.cuda.is_available():
                visible_cuda_devices = torch.cuda.device_count()
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                for rank in range(min(world_size, visible_cuda_devices)):
                    max_memory_reserved_mb = torch.cuda.max_memory_reserved(device=rank) / (1024 * 1024)
                    max_memory_allocated_mb = torch.cuda.max_memory_allocated(device=rank) / (1024 * 1024)
                    run.summary[f"max_memory_reserved_rank_{rank}"] = max_memory_reserved_mb
                    run.summary[f"max_memory_allocated_rank_{rank}"] = max_memory_allocated_mb

    training_skipped = trainer.state.epoch is None or trainer.state.epoch < 1.0
    if not args.dry_run and training_skipped:
        if trainer.is_world_process_zero():
            failed_msg = f"Training terminated at epoch {trainer.state.epoch}, before 1 epoch. Skipping model save."
            print(failed_msg)
            Path(args.output_dir, "training_skipped.txt").write_text(failed_msg)
            if args.use_wandb and run is not None:
                run.summary["training_skipped"] = True
    elif not args.dry_run and args.final_model_output_dir is not None:
        trainer.save_model(args.final_model_output_dir)
        if trainer.is_world_process_zero():
            print(f"Model saved to {args.final_model_output_dir}")


def cleanup() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    atexit.register(cleanup)
    main()
