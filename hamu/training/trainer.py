"""Trainer extensions for HAMU and baseline unlearning methods."""

from __future__ import annotations

from collections import defaultdict
from itertools import cycle
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from accelerate.optimizer import AcceleratedOptimizer
from peft import PeftType
from peft.peft_model import PeftModelForCausalLM
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import EvalPrediction, Trainer, TrainerCallback
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.utils import entropy_from_logits

from hamu.methods.hamu import GradientTransformOptimizer, HAMUOptimizer
from hamu.models.resnet20 import ResNet20ForCIFAR
from hamu.training.modality import is_llm_modality

IS_LLM = is_llm_modality()
BaseTrainer = SFTTrainer if IS_LLM else Trainer


def unwrap_gradient_optimizer(optimizer: Any) -> Optional[GradientTransformOptimizer]:
    if isinstance(optimizer, AcceleratedOptimizer):
        optimizer = optimizer.optimizer
    if isinstance(optimizer, GradientTransformOptimizer):
        return optimizer
    return None


class GradientTransformCallback(TrainerCallback):
    """Runs the gradient transform immediately before the optimizer step."""

    def __init__(self, additional_callbacks: Optional[list[Callable[..., Any]]] = None) -> None:
        super().__init__()
        self.optimizer: Optional[GradientTransformOptimizer] = None
        self.additional_callbacks = additional_callbacks or []

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs) -> Any:
        optimizer = unwrap_gradient_optimizer(self.optimizer)
        if optimizer is None:
            return control
        try:
            optimizer._step()
            optimizer.clear_retain_gradients()
        except ValueError as exc:
            print(f"Stopping training due to optimizer error: {exc}")
            control.should_log = True
            control.should_training_stop = True
        for callback in self.additional_callbacks:
            callback(args, state, control, **kwargs)
        if getattr(optimizer, "stop_reason", None) == "failed_feasibility":
            print("Stopping training because HAMU feasibility failed.")
            control.should_log = True
            control.should_training_stop = True
            optimizer.stop_reason = None
        elif getattr(optimizer, "stop_reason", None) == "stopping_criterion":
            print("Stopping training because the HAMU stopping criterion was met.")
            control.should_log = True
            control.should_training_stop = True
            optimizer.stop_reason = None
        return control


class BaseUnlearningTrainer(BaseTrainer):
    """Common Trainer functionality for HAMU and baselines."""

    def __init__(
        self,
        *,
        gradient_pairing: str = "none",
        debug: bool = False,
        k_pct: float = 0.2,
        forget_dataset_for_pairing: Optional[Dataset] = None,
        **kwargs,
    ) -> None:
        callbacks = kwargs.pop("callbacks", [])
        self.gradient_transform_callback = GradientTransformCallback(additional_callbacks=[self.my_log])
        callbacks.append(self.gradient_transform_callback)
        self.gradient_pairing = gradient_pairing
        self.alternate_gpu = gradient_pairing == "split-gpu"
        self.debug = debug
        self.k_pct = k_pct
        self.forget_dataset_for_pairing = forget_dataset_for_pairing
        self._forget_iterator = None
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self.eval_metrics = defaultdict(torch.Tensor)
        self.loss_func = torch.nn.CrossEntropyLoss(reduction="none")
        super().__init__(compute_metrics=self._compute_metrics, callbacks=callbacks, **kwargs)
        gradient_optimizer = unwrap_gradient_optimizer(self.optimizer)
        if gradient_optimizer is not None:
            self.gradient_transform_callback.optimizer = gradient_optimizer

    def set_optimizer(self, optimizer: Optional[torch.optim.Optimizer]) -> None:
        self.optimizer = optimizer
        gradient_optimizer = unwrap_gradient_optimizer(optimizer)
        if gradient_optimizer is not None:
            self.gradient_transform_callback.optimizer = gradient_optimizer

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        metric_key = f"{metric_key_prefix}_retain_forget_accuracy"
        if self.add_retain_forget_accuracy(metrics, metric_key_prefix):
            self.log({metric_key: metrics[metric_key]})
        return metrics

    @staticmethod
    def add_retain_forget_accuracy(metrics: dict[str, float], metric_key_prefix: str = "eval") -> bool:
        retain_key = f"{metric_key_prefix}_retain_accuracy"
        forget_key = f"{metric_key_prefix}_forget_accuracy"
        metric_key = f"{metric_key_prefix}_retain_forget_accuracy"
        if retain_key not in metrics or forget_key not in metrics:
            return False
        metrics[metric_key] = (metrics[retain_key] + metrics[forget_key]) / 2.0
        return True

    def _standard_compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        *,
        return_outputs: bool,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        if IS_LLM:
            inputs["use_cache"] = False
        return BaseTrainer.compute_loss(
            self,
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def update_frozen_model(self) -> None:
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        frozen_model = type(unwrapped_model)(unwrapped_model.config)
        frozen_model.load_state_dict(unwrapped_model.state_dict())
        frozen_model.to(self.accelerator.device)
        frozen_model.eval()
        frozen_model.requires_grad_(False)
        self.model.frozen_model = [frozen_model]
        unwrapped_model.frozen_model = [frozen_model]

    def log(self, logs: dict[str, float | torch.Tensor], start_time: float | None = None) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mode = "train" if self.model.training else "eval"
        if mode == "eval":
            min_k_logprobs = {k: logs.pop(k) for k in tuple(logs.keys()) if "min_k_logprob" in k}
            mean_losses = {k: logs.pop(k) for k in tuple(logs.keys()) if "mean_losses" in k}
            self.eval_metrics.update(min_k_logprobs)
            self.eval_metrics.update(mean_losses)
            self._append_auc_metrics()

        for key, values in self._metrics[mode].items():
            if any(isinstance(item, torch.Tensor) for item in values):
                self._metrics[mode][key] = [
                    item.item() if isinstance(item, torch.Tensor) and item.numel() == 1 else item
                    for item in values
                ]
        metrics = {
            key: sum(values) / len(values)
            for key, values in self._metrics[mode].items()
            if len(values) > 0 and not isinstance(values[0], torch.Tensor)
        }
        if mode == "eval":
            metrics = {f"eval_{key}": value for key, value in metrics.items()}
        logs.update(metrics)
        Trainer.log(self, logs, start_time)
        self._metrics[mode].clear()

    def _append_auc_metrics(self) -> None:
        if (
            "eval_retain_mean_losses" in self.eval_metrics
            and "eval_forget_mean_losses" in self.eval_metrics
            and "eval_test_mean_losses" in self.eval_metrics
        ):
            retain_losses = self.eval_metrics.pop("eval_retain_mean_losses")
            forget_losses = self.eval_metrics.pop("eval_forget_mean_losses")
            test_losses = self.eval_metrics.pop("eval_test_mean_losses")
            self._append_auc("retain_forget_losses_auc", retain_losses, forget_losses)
            self._append_auc("forget_test_losses_auc", forget_losses, test_losses)

        if (
            "eval_retain_min_k_logprobs" in self.eval_metrics
            and "eval_forget_min_k_logprobs" in self.eval_metrics
            and "eval_test_min_k_logprobs" in self.eval_metrics
        ):
            retain_scores = self.eval_metrics.pop("eval_retain_min_k_logprobs")
            forget_scores = self.eval_metrics.pop("eval_forget_min_k_logprobs")
            test_scores = self.eval_metrics.pop("eval_test_min_k_logprobs")
            self._append_auc("retain_forget_min_k_logprobs_auc", forget_scores, retain_scores)
            self._append_auc("forget_test_min_k_logprobs_auc", test_scores, forget_scores)

    def _append_auc(self, name: str, negatives: torch.Tensor, positives: torch.Tensor) -> None:
        y_true = np.array((0,) * len(negatives) + (1,) * len(positives))
        y_scores = torch.cat([negatives, positives])
        y_scores[torch.isnan(y_scores)] = 0.0
        try:
            auc = roc_auc_score(y_true, y_scores.numpy())
        except ValueError:
            auc = 0.5
        self._metrics["eval"][name].append(auc)

    def my_log(self, *args: Any, **kwargs) -> None:
        optimizer = unwrap_gradient_optimizer(self.optimizer)
        if optimizer is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        mode = "train" if self.model.training else "eval"
        for key, values in optimizer.logs.items():
            if values:
                self._metrics[mode][key].extend(values)
        optimizer.logs.clear()

    def _compute_metrics(self, pred: EvalPrediction, compute_result: bool = True) -> dict[str, float]:
        mode = "train" if self.model.training else "eval"
        logits = pred.predictions
        labels = pred.label_ids
        if not isinstance(logits, torch.Tensor):
            logits = torch.from_numpy(logits)
        if not isinstance(labels, torch.Tensor):
            labels = torch.from_numpy(labels)

        accelerator = getattr(self, "accelerator", None)
        device = accelerator.device if accelerator is not None else torch.device("cpu")
        logits = logits.to(device)
        labels = labels.to(device)

        if IS_LLM:
            logits = logits[:, :-1]
            labels = labels[:, 1:]

        full_losses = self.loss_func(logits.reshape(-1, logits.shape[-1]), labels.flatten()).reshape(labels.shape)
        if IS_LLM:
            tokens = (labels != -100).sum(dim=-1)
            tokens[tokens == 0] = 1
            mean_losses = full_losses.sum(dim=-1) / tokens
        else:
            mean_losses = full_losses
        self._metrics[mode]["mean_losses"].append(mean_losses.to("cpu", non_blocking=True))

        if IS_LLM:
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            logprobs = logprobs.gather(dim=-1, index=torch.clamp(labels.unsqueeze(-1), min=0)).squeeze(-1)
            logprobs[labels == -100] = torch.nan
            self._metrics[mode]["mean_logprobs"].append(logprobs.nanmean(dim=-1).mean().to("cpu", non_blocking=True))
            self._metrics[mode]["mean_probs"].append(torch.exp(logprobs).nanmean(dim=-1).mean().to("cpu", non_blocking=True))
            self._metrics[mode]["min_k_logprobs"].append(
                self.collate_min_k_logprob(logprobs, labels).to("cpu", non_blocking=True)
            )
        else:
            accuracy, entropy = self._classification_accuracy_entropy(logits, labels)
            self._metrics[mode]["accuracy"].append(accuracy)
            self._metrics[mode]["entropy"].append(entropy)

        if not compute_result:
            return {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        metrics = {}
        for key, values in self._metrics.get("eval", {}).items():
            if "min_k_logprob" in key or "mean_losses" in key:
                metrics[key] = torch.cat(values)
            elif values:
                metrics[key] = sum(values) / len(values)
                if isinstance(metrics[key], torch.Tensor):
                    metrics[key] = metrics[key].item()
        self._metrics["eval"].clear()
        return metrics

    def gather_alternate(self, tensor: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        tensor = self.accelerator.gather_for_metrics(tensor).reshape(-1, 2, *tensor.shape[1:])
        if reduction == "mean":
            return tensor.mean(dim=0)
        if reduction == "sum":
            return tensor.sum(dim=0)
        if reduction == "none":
            return tensor
        raise ValueError(f"Unsupported reduction: {reduction}")

    def collate_min_k_logprob(self, logprobs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        k = ((labels != -100).sum(dim=-1) * self.k_pct).int()
        k = torch.clamp(k, min=1)
        indices = (torch.topk(logprobs[i], k=k[i].item(), largest=False).indices for i in range(logprobs.size(0)))
        return torch.stack([logprobs[i][idx].nanmean() for i, idx in enumerate(indices)])

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        mode = "train" if self.model.training else "eval"
        if not self.model.training or not self.alternate_gpu:
            loss, outputs = self._standard_compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            self._record_standard_metrics(mode, loss, outputs, inputs)
            return (loss, outputs) if return_outputs else loss

        with self.accelerator.no_sync(model):
            loss, outputs = self._standard_compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
        self._record_split_metrics(mode, loss, outputs, inputs)
        self.compute_stats(mode)
        return (loss, outputs) if return_outputs else loss

    def _record_standard_metrics(self, mode: str, loss: torch.Tensor, outputs: Any, inputs: dict[str, Any]) -> None:
        if mode != "train":
            return
        if IS_LLM and "labels" in inputs:
            self._record_single_llm_token_accuracy(mode, outputs, inputs, "mean_token_accuracy")
        if not IS_LLM and "labels" in inputs:
            accuracy, entropy = self._classification_accuracy_entropy(outputs.logits, inputs["labels"])
            self._metrics[mode]["accuracy"].append(accuracy)
            self._metrics[mode]["entropy"].append(entropy)

    @staticmethod
    def _classification_accuracy_entropy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
        predictions = logits.argmax(dim=-1)
        mask = labels != -100
        total = mask.sum().item()
        accuracy = (((predictions == labels) & mask).sum().item() / total) if total > 0 else 0.0

        entropy = entropy_from_logits(logits)
        if entropy.shape == mask.shape:
            entropy = entropy[mask]
        entropy_value = entropy.mean().item() if entropy.numel() > 0 else 0.0
        return accuracy, entropy_value

    def _record_split_metrics(self, mode: str, loss: torch.Tensor, outputs: Any, inputs: dict[str, Any]) -> None:
        with torch.no_grad():
            gathered_loss = self.gather_alternate(loss)
            self._metrics[mode]["retain_loss"].append(gathered_loss[0].item())
            self._metrics[mode]["forget_loss"].append(gathered_loss[1].item())

            entropy = entropy_from_logits(outputs.logits)
            if "attention_mask" in inputs:
                entropy = torch.sum(entropy * inputs["attention_mask"]) / inputs["attention_mask"].sum()
            else:
                entropy = torch.mean(entropy)
            entropy = self.gather_alternate(entropy)
            self._metrics[mode]["retain_entropy"].append(entropy[0].item())
            self._metrics[mode]["forget_entropy"].append(entropy[1].item())

            if IS_LLM and "labels" in inputs:
                correct_tokens, total_tokens = self._llm_token_accuracy_counts(outputs.logits, inputs)
                correct_tokens = self.gather_alternate(correct_tokens, reduction="sum")
                total_tokens = self.gather_alternate(total_tokens, reduction="sum")
                self._metrics[mode]["retain_token_accuracy"].append(
                    (correct_tokens[0] / total_tokens[0]).item() if total_tokens[0] > 0 else 0.0
                )
                self._metrics[mode]["forget_token_accuracy"].append(
                    (correct_tokens[1] / total_tokens[1]).item() if total_tokens[1] > 0 else 0.0
                )
            elif not IS_LLM and "labels" in inputs:
                predictions = outputs.logits.argmax(dim=-1)
                labels = inputs["labels"]
                mask = labels != -100
                correct = ((predictions == labels) & mask).sum()
                total = mask.sum()
                correct = self.gather_alternate(correct, reduction="sum")
                total = self.gather_alternate(total, reduction="sum")
                self._metrics[mode]["retain_accuracy"].append((correct[0] / total[0]).item() if total[0] > 0 else 0.0)
                self._metrics[mode]["forget_accuracy"].append((correct[1] / total[1]).item() if total[1] > 0 else 0.0)

    def _llm_token_accuracy_counts(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "shift_labels" in inputs:
            shift_logits = logits.contiguous()
            shift_labels = inputs["shift_labels"].contiguous()
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if (
            getattr(self, "num_virtual_tokens", 0) > 0
            and hasattr(unwrapped_model, "peft_config")
            and unwrapped_model.peft_config[unwrapped_model.active_adapter].peft_type != PeftType.PREFIX_TUNING
        ):
            shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

        seq_len = min(shift_logits.shape[-2], shift_labels.shape[-1])
        shift_logits = shift_logits[..., :seq_len, :]
        shift_labels = shift_labels[..., :seq_len]

        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        correct_tokens = ((predictions == shift_labels) & mask).sum()
        total_tokens = mask.sum()
        return correct_tokens, total_tokens

    def _record_single_llm_token_accuracy(
        self,
        mode: str,
        outputs: Any,
        inputs: dict[str, Any],
        metric_key: str,
    ) -> None:
        with torch.no_grad():
            correct_tokens, total_tokens = self._llm_token_accuracy_counts(outputs.logits, inputs)
            correct_tokens = self.accelerator.gather_for_metrics(correct_tokens).sum()
            total_tokens = self.accelerator.gather_for_metrics(total_tokens).sum()
            accuracy = (correct_tokens / total_tokens).item() if total_tokens > 0 else 0.0
            self._metrics[mode][metric_key].append(accuracy)

    def compute_stats(self, mode: str) -> None:
        return

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.alternate_gpu:
            with self.accelerator.no_sync(model):
                return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def _compute_KL(
        self,
        model: PeftModelForCausalLM,
        inputs: dict[str, torch.Tensor | Any],
        logits: torch.Tensor,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            target = self.compute_original_outputs_logits(model, dict(inputs), num_items_in_batch=num_items_in_batch)
        logits_log_probs = logits.float().log_softmax(dim=-1)
        target_log_probs = target.float().log_softmax(dim=-1)
        kl = torch.nn.functional.kl_div(
            logits_log_probs,
            target_log_probs,
            log_target=True,
            reduction="none",
        ).sum(dim=-1)
        kl = kl.clamp_min(0.0)
        if IS_LLM and getattr(self, "num_virtual_tokens", 0) > 0 and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING:
            kl = kl[:, self.num_virtual_tokens :]
        if "attention_mask" in inputs:
            return torch.sum(kl * inputs["attention_mask"]) / inputs["attention_mask"].sum()
        if "position_ids" in inputs:
            return torch.mean(kl)
        if not IS_LLM:
            return torch.mean(kl)
        raise ValueError("Expected attention_mask or position_ids in inputs.")

    def compute_original_outputs_logits(
        self,
        model: PeftModelForCausalLM | DistributedDataParallel,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hasattr(model, "module"):
            model = model.module
        if IS_LLM:
            inputs.pop("labels", None)
            if self.model_accepts_loss_kwargs and num_items_in_batch is not None:
                inputs = {**inputs, "num_items_in_batch": num_items_in_batch}
            original_adapter = model.active_adapter
            model.set_adapter("original")
            # was_training = model.training
            # model.eval()
            outputs = model(**inputs)
            # if was_training:
            #     model.train()
            model.set_adapter(original_adapter)
        elif isinstance(model, ResNet20ForCIFAR):
            inputs.pop("labels", None)
            model.frozen_model[0].eval()
            outputs = model.frozen_model[0](**inputs)
        else:
            raise NotImplementedError(f"Original-model logits are not implemented for {type(model)}.")
        return outputs.logits

    def _get_forget_iterator(self) -> Iterator[dict[str, Any]]:
        if self.forget_dataset_for_pairing is None:
            raise RuntimeError("Sequential paired-gradient training requires forget_dataset_for_pairing.")
        if self._forget_iterator is None:
            dataloader = DataLoader(
                self.forget_dataset_for_pairing,
                sampler=SequentialSampler(self.forget_dataset_for_pairing),
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                drop_last=False,
            )
            dataloader = self.accelerator.prepare(dataloader)
            self._forget_iterator = cycle(dataloader)
        return self._forget_iterator

    def _next_forget_inputs(self) -> dict[str, Any]:
        return self._prepare_inputs(next(self._get_forget_iterator()))


class PairedGradientTrainer(BaseUnlearningTrainer):
    """Trainer that can compute paired retain/forget gradients sequentially."""

    def _clip_sequential_retain_gradients(self, model: nn.Module) -> None:
        max_grad_norm = self.args.max_grad_norm
        if max_grad_norm is None or max_grad_norm <= 0:
            return
        self.accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

    def _paired_retain_objective(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        loss: torch.Tensor,
        outputs: Any,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        return loss

    def _paired_forget_objective(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        loss: torch.Tensor,
        outputs: Any,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        return loss

    def _record_sequential_pair_metrics(
        self,
        retain_loss: torch.Tensor,
        forget_loss: torch.Tensor,
        retain_outputs: Any,
        forget_outputs: Any,
        retain_inputs: dict[str, Any],
        forget_inputs: dict[str, Any],
    ) -> None:
        self._metrics["train"]["retain_loss"].append(retain_loss.detach().item())
        self._metrics["train"]["forget_loss"].append(forget_loss.detach().item())
        if retain_outputs is not None:
            self._record_standard_metrics("train", retain_loss.detach(), retain_outputs, retain_inputs)
        self._record_standard_metrics("train", forget_loss.detach(), forget_outputs, forget_inputs)
        if IS_LLM and retain_outputs is not None and "labels" in retain_inputs:
            self._record_single_llm_token_accuracy("train", retain_outputs, retain_inputs, "retain_token_accuracy")
        if IS_LLM and forget_outputs is not None and "labels" in forget_inputs:
            self._record_single_llm_token_accuracy("train", forget_outputs, forget_inputs, "forget_token_accuracy")

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.gradient_pairing != "sequential":
            return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        optimizer = unwrap_gradient_optimizer(self.optimizer)
        if optimizer is None:
            raise RuntimeError("Sequential paired-gradient training requires a GradientTransformOptimizer.")

        model.train()
        retain_inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            retain_loss, retain_outputs = self._standard_compute_loss(
                model,
                retain_inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            retain_objective = self._paired_retain_objective(
                model,
                retain_inputs,
                retain_loss,
                retain_outputs,
                num_items_in_batch,
            )
            retain_objective_for_log = retain_objective.detach()
        # In Hugging Face's default Trainer, loss is automatically divided by self.args.gradient_accumulation_steps before backward() is called.
        # We need to do the same for the retain objective to ensure that the gradient magnitudes are correct when using gradient accumulation.
        if self.args.gradient_accumulation_steps > 1:
            retain_objective = retain_objective / self.args.gradient_accumulation_steps
        self.accelerator.backward(retain_objective)
        # Trainer clips the forget pass after training_step; clip retain before stashing it.
        self._clip_sequential_retain_gradients(model)
        optimizer.store_retain_gradients()
        self.optimizer.zero_grad(set_to_none=True)

        # Record standard metrics for retain and free outputs to save VRAM before the forget pass
        self._record_standard_metrics("train", retain_loss.detach(), retain_outputs, retain_inputs)
        if IS_LLM and retain_outputs is not None and "labels" in retain_inputs:
            self._record_single_llm_token_accuracy("train", retain_outputs, retain_inputs, "retain_token_accuracy")
        del retain_outputs
        retain_outputs = None

        forget_inputs = self._next_forget_inputs()
        with self.compute_loss_context_manager():
            forget_loss, forget_outputs = self._standard_compute_loss(
                model,
                forget_inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            forget_objective = self._paired_forget_objective(
                model,
                forget_inputs,
                forget_loss,
                forget_outputs,
                num_items_in_batch,
            )
            forget_objective_for_log = forget_objective.detach()
        # In Hugging Face's default Trainer, loss is automatically divided by self.args.gradient_accumulation_steps before backward() is called.
        # We need to do the same for the forget objective to ensure that the gradient magnitudes are correct when using gradient accumulation.
        if self.args.gradient_accumulation_steps > 1:
            forget_objective = forget_objective / self.args.gradient_accumulation_steps
        self.accelerator.backward(forget_objective)

        self._record_sequential_pair_metrics(
            retain_loss,
            forget_loss,
            retain_outputs,
            forget_outputs,
            retain_inputs,
            forget_inputs,
        )
        self.compute_stats("train")
        return ((retain_objective_for_log + forget_objective_for_log) / 2.0).detach()


class GATrainer(BaseUnlearningTrainer):
    def __init__(self, **kwargs) -> None:
        if kwargs.get("gradient_pairing", "none") != "none":
            raise ValueError("GA does not use paired gradients.")
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if self.model.training:
            metrics = self._metrics["train"]
            metrics["forget_loss"].append(loss.item())
            entropy_values = metrics.pop("entropy", None)
            if entropy_values is not None:
                metrics["forget_entropy"].extend(entropy_values)
            if IS_LLM:
                token_accuracy_values = metrics.pop("mean_token_accuracy", None)
                if token_accuracy_values is not None:
                    metrics["forget_token_accuracy"].extend(token_accuracy_values)
                elif "labels" in inputs:
                    self._record_single_llm_token_accuracy("train", outputs, inputs, "forget_token_accuracy")
            else:
                accuracy_values = metrics.pop("accuracy", None)
                if accuracy_values is not None:
                    metrics["forget_accuracy"].extend(accuracy_values)
            loss = -loss
        return (loss, outputs) if return_outputs else loss


class FTTrainer(BaseUnlearningTrainer):
    def __init__(self, **kwargs) -> None:
        if kwargs.get("gradient_pairing", "none") != "none":
            raise ValueError("FT does not use paired gradients.")
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if self.model.training:
            metrics = self._metrics["train"]
            metrics["retain_loss"].append(loss.item())
            entropy_values = metrics.pop("entropy", None)
            if entropy_values is not None:
                metrics["retain_entropy"].extend(entropy_values)
            if IS_LLM:
                token_accuracy_values = metrics.pop("mean_token_accuracy", None)
                if token_accuracy_values is not None:
                    metrics["retain_token_accuracy"].extend(token_accuracy_values)
                elif "labels" in inputs:
                    self._record_single_llm_token_accuracy("train", outputs, inputs, "retain_token_accuracy")
            else:
                accuracy_values = metrics.pop("accuracy", None)
                if accuracy_values is not None:
                    metrics["retain_accuracy"].extend(accuracy_values)
        return (loss, outputs) if return_outputs else loss


class GDiffTrainer(PairedGradientTrainer):
    def __init__(self, **kwargs) -> None:
        if kwargs.get("gradient_pairing") not in {"split-gpu", "sequential"}:
            raise ValueError("GDiff requires split-gpu or sequential gradient pairing.")
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if self.model.training and self.accelerator.process_index % 2 == 1:
            loss = -loss
        return (loss, outputs) if return_outputs else loss

    def _paired_forget_objective(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        loss: torch.Tensor,
        outputs: Any,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        return -loss


class KLTrainer(PairedGradientTrainer):
    def __init__(self, **kwargs) -> None:
        if kwargs.get("gradient_pairing") not in {"split-gpu", "sequential"}:
            raise ValueError("KL requires split-gpu or sequential gradient pairing.")
        super().__init__(**kwargs)
        self._last_sequential_retain_kl: torch.Tensor | None = None
        self._last_sequential_forget_kl: torch.Tensor | None = None

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if self.model.training:
            with self.accelerator.no_sync(model):
                kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch)
            loss = kl if self.accelerator.process_index % 2 == 0 else -loss
            with torch.no_grad():
                kl_pair = self.gather_alternate(kl)
                self._metrics["train"]["retain_KL"].append(kl_pair[0].mean().item())
                self._metrics["train"]["forget_KL"].append(kl_pair[1].mean().item())
        else:
            with torch.no_grad():
                kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch)
            self._metrics["eval"]["KL"].append(self.accelerator.gather_for_metrics(kl).mean().item())
        return (loss, outputs) if return_outputs else loss

    def _paired_retain_objective(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        loss: torch.Tensor,
        outputs: Any,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch)
        self._last_sequential_retain_kl = kl.detach()
        return kl

    def _paired_forget_objective(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        loss: torch.Tensor,
        outputs: Any,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        with torch.no_grad():
            self._last_sequential_forget_kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch).detach()
        return -loss

    def _record_sequential_pair_metrics(
        self,
        retain_loss: torch.Tensor,
        forget_loss: torch.Tensor,
        retain_outputs: Any,
        forget_outputs: Any,
        retain_inputs: dict[str, Any],
        forget_inputs: dict[str, Any],
    ) -> None:
        super()._record_sequential_pair_metrics(
            retain_loss,
            forget_loss,
            retain_outputs,
            forget_outputs,
            retain_inputs,
            forget_inputs,
        )
        if self._last_sequential_retain_kl is not None:
            self._metrics["train"]["retain_KL"].append(self._last_sequential_retain_kl.mean().item())
        if self._last_sequential_forget_kl is not None:
            self._metrics["train"]["forget_KL"].append(self._last_sequential_forget_kl.mean().item())
        self._last_sequential_retain_kl = None
        self._last_sequential_forget_kl = None


class SCRUBTrainer(PairedGradientTrainer):
    def __init__(self, alpha: float = 1.0, gamma: float = 1.0, **kwargs) -> None:
        if kwargs.get("gradient_pairing") not in {"split-gpu", "sequential"}:
            raise ValueError("SCRUB requires split-gpu or sequential gradient pairing.")
        super().__init__(**kwargs)
        self._alpha = alpha
        self._gamma = gamma
        self._last_sequential_retain_kl: torch.Tensor | None = None
        self._last_sequential_forget_kl: torch.Tensor | None = None

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        if self.model.training:
            with self.accelerator.no_sync(model):
                kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch)
            loss = self._alpha * kl + self._gamma * loss if self.accelerator.process_index % 2 == 0 else -kl
            with torch.no_grad():
                kl_pair = self.gather_alternate(kl)
                self._metrics["train"]["retain_KL"].append(kl_pair[0].mean().item())
                self._metrics["train"]["forget_KL"].append(kl_pair[1].mean().item())
        else:
            with torch.no_grad():
                kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch)
            self._metrics["eval"]["KL"].append(self.accelerator.gather_for_metrics(kl).mean().item())
        return (loss, outputs) if return_outputs else loss

    def _paired_retain_objective(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        loss: torch.Tensor,
        outputs: Any,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch)
        self._last_sequential_retain_kl = kl.detach()
        return self._alpha * kl + self._gamma * loss

    def _paired_forget_objective(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        loss: torch.Tensor,
        outputs: Any,
        num_items_in_batch: torch.Tensor | None,
    ) -> torch.Tensor:
        kl = self._compute_KL(model, inputs, outputs.logits, num_items_in_batch)
        self._last_sequential_forget_kl = kl.detach()
        return -kl

    def _record_sequential_pair_metrics(
        self,
        retain_loss: torch.Tensor,
        forget_loss: torch.Tensor,
        retain_outputs: Any,
        forget_outputs: Any,
        retain_inputs: dict[str, Any],
        forget_inputs: dict[str, Any],
    ) -> None:
        super()._record_sequential_pair_metrics(
            retain_loss,
            forget_loss,
            retain_outputs,
            forget_outputs,
            retain_inputs,
            forget_inputs,
        )
        if self._last_sequential_retain_kl is not None:
            self._metrics["train"]["retain_KL"].append(self._last_sequential_retain_kl.mean().item())
        if self._last_sequential_forget_kl is not None:
            self._metrics["train"]["forget_KL"].append(self._last_sequential_forget_kl.mean().item())
        self._last_sequential_retain_kl = None
        self._last_sequential_forget_kl = None


class HAMUTrainer(PairedGradientTrainer):
    """Trainer for HAMU-Q/HAMU-U."""

    def my_log(self, *args: Any, **kwargs) -> None:
        optimizer = unwrap_gradient_optimizer(self.optimizer)
        if isinstance(optimizer, HAMUOptimizer) and "feasibility" in optimizer.logs:
            optimizer.logs["max_feasibility"].append(max(optimizer.logs["feasibility"]))
        super().my_log(*args, **kwargs)

    def compute_stats(self, mode: str) -> None:
        if mode != "train":
            return
        optimizer = unwrap_gradient_optimizer(self.optimizer)
        if not isinstance(optimizer, HAMUOptimizer):
            return
        stats = self.accelerator.gather_for_metrics(optimizer.stats).reshape(-1, optimizer.stats.size(0)).sum(dim=0)
        total = torch.clamp(stats.sum(), min=1)
        stats = (stats / total).cpu()
        self._metrics[mode]["branch_direct"] = [stats[0].item()]
        self._metrics[mode]["branch_rectified"] = [stats[1].item()]
        self._metrics[mode]["failed_feasibility"] = [stats[2].item()]
        self._metrics[mode]["nan_in_delta"] = [stats[3].item()]

class ThresholdStoppingCallback(TrainerCallback):
    """Stop when HAMU feasibility, stopping criterion, or loss thresholds are met."""

    def __init__(
        self,
        stop_on_failed_feasibility: bool = False,
        stop_on_stopping_criterion: bool = False,
        stop_on_loss_thresholds: bool = False,
        retain_loss_threshold: float = 0.4,
        forget_loss_threshold: float = 0.1,
        *args: Any,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.failed_feasibility_threshold = 0.0 if stop_on_failed_feasibility else 0.99
        self.stop_on_stopping_criterion = stop_on_stopping_criterion
        self.stop_on_loss_thresholds = stop_on_loss_thresholds
        self.retain_loss_threshold = retain_loss_threshold
        self.forget_loss_threshold = forget_loss_threshold

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict[str, Any]] = None, **kwargs) -> Any:
        if logs is None:
            return control
        failed_feasibility = logs.get("failed_feasibility")
        if failed_feasibility is not None and torch.tensor(failed_feasibility) > self.failed_feasibility_threshold:
            print(f"Stopping training due to high failed_feasibility rate: {failed_feasibility}")
            control.should_training_stop = True
        stopping_met = logs.get("stopping_met")
        if self.stop_on_stopping_criterion and stopping_met is not None and torch.tensor(stopping_met) > 0.0:
            print("Stopping training because the HAMU stopping criterion was met.")
            control.should_training_stop = True
        if self.stop_on_loss_thresholds:
            retain_loss = max(
                (value for value in (logs.get("retain_loss"), logs.get("eval_retain_loss")) if value is not None),
                default=None,
            )
            if retain_loss is not None and torch.tensor(retain_loss) > self.retain_loss_threshold:
                print(f"Stopping training due to high retain_loss: {retain_loss}")
                control.should_training_stop = True
                return control
            forget_loss = min(
                (value for value in (logs.get("forget_loss"), logs.get("eval_forget_loss")) if value is not None),
                default=None,
            )
            if forget_loss is not None and torch.tensor(forget_loss) < self.forget_loss_threshold:
                print(f"Stopping training due to low forget_loss: {forget_loss}")
                control.should_training_stop = True
                return control
        return control
