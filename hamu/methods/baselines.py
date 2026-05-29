"""Gradient-transform baselines used in the HAMU experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from hamu.methods.core import gru_equivalent_gradient, pcgrad_equivalent_gradient
from hamu.methods.hamu import GradientTransformOptimizer


BASELINE_METHODS = ("ft", "ga", "gdiff", "kl", "scrub", "gru", "pcgrad")


@dataclass(frozen=True)
class BaselineMethod:
    """Public metadata for a baseline unlearning method."""

    name: str
    display_name: str
    requires_paired_gradients: bool
    supports_single_gpu: bool
    description: str


BASELINE_REGISTRY = {
    "ft": BaselineMethod(
        name="ft",
        display_name="Fine-Tune",
        requires_paired_gradients=False,
        supports_single_gpu=True,
        description="Fine-tune on the retain set.",
    ),
    "ga": BaselineMethod(
        name="ga",
        display_name="Gradient Ascent",
        requires_paired_gradients=False,
        supports_single_gpu=True,
        description="Maximize loss on the forget set.",
    ),
    "gdiff": BaselineMethod(
        name="gdiff",
        display_name="GDiff",
        requires_paired_gradients=True,
        supports_single_gpu=True,
        description="Use retain descent and forget ascent on paired retain/forget batches.",
    ),
    "kl": BaselineMethod(
        name="kl",
        display_name="KL",
        requires_paired_gradients=True,
        supports_single_gpu=True,
        description="Retain with KL to the original model and ascend forget loss.",
    ),
    "scrub": BaselineMethod(
        name="scrub",
        display_name="SCRUB",
        requires_paired_gradients=True,
        supports_single_gpu=True,
        description="SCRUB-style retain KL/CE objective and forget KL ascent.",
    ),
    "gru": BaselineMethod(
        name="gru",
        display_name="GRU",
        requires_paired_gradients=True,
        supports_single_gpu=True,
        description="Gradient rectified unlearning.",
    ),
    "pcgrad": BaselineMethod(
        name="pcgrad",
        display_name="PCGrad",
        requires_paired_gradients=True,
        supports_single_gpu=True,
        description="PCGrad-style projection of paired retain/forget gradients.",
    ),
}


def get_baseline_method(name: str) -> BaselineMethod:
    try:
        return BASELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline method: {name}") from exc


class GRUOptimizer(GradientTransformOptimizer):
    """Gradient rectified unlearning baseline."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        lr: float,
        gradient_pairing: str = "split-gpu",
    ) -> None:
        super().__init__(
            model,
            defaults={"lr": lr},
            optimizer=optimizer,
            gradient_pairing=gradient_pairing,
        )

    def get_grad(
        self,
        p: torch.nn.Parameter,
        eps: float = 1e-12,
        tau: Optional[float] = None,
        full_retain_norm: Optional[torch.Tensor] = None,
        full_dot_product: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
        return gru_equivalent_gradient(
            retain_grad,
            forget_grad,
            eps=eps,
            tau=tau,
            full_retain_norm=full_retain_norm,
            full_dot_product=full_dot_product,
        )

    def preprocess_params(self) -> dict[str, torch.Tensor]:
        retain_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        forget_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        dot_product = torch.tensor(0.0, device=self.accelerator.device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
                retain_norm_sq += torch.sum(retain_grad * retain_grad)
                forget_norm_sq += torch.sum(forget_grad * forget_grad)
                dot_product += torch.dot(retain_grad.flatten(), forget_grad.flatten())
        retain_norm = torch.sqrt(retain_norm_sq)
        forget_norm = torch.sqrt(forget_norm_sq)
        self.logs["full_retain_grad_norm"].append(retain_norm.to(device="cpu", non_blocking=True))
        self.logs["full_forget_grad_norm"].append(forget_norm.to(device="cpu", non_blocking=True))
        self.logs["full_gradient_dot_product"].append(dot_product.to(device="cpu", non_blocking=True))
        return {"full_retain_norm": retain_norm, "full_forget_norm": forget_norm, "full_dot_product": dot_product}


class PCGradOptimizer(GradientTransformOptimizer):
    """PCGrad-style paired-gradient baseline."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        lr: float,
        gradient_pairing: str = "split-gpu",
    ) -> None:
        super().__init__(
            model,
            defaults={"lr": lr},
            optimizer=optimizer,
            gradient_pairing=gradient_pairing,
        )

    def get_grad(
        self,
        p: torch.nn.Parameter,
        eps: float = 1e-12,
        tau: Optional[float] = None,
        full_retain_norm: Optional[torch.Tensor] = None,
        full_forget_norm: Optional[torch.Tensor] = None,
        full_dot_product: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
        return pcgrad_equivalent_gradient(
            retain_grad,
            forget_grad,
            eps=eps,
            tau=tau,
            full_retain_norm=full_retain_norm,
            full_forget_norm=full_forget_norm,
            full_dot_product=full_dot_product,
        )

    def preprocess_params(self) -> dict[str, torch.Tensor]:
        retain_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        forget_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
        dot_product = torch.tensor(0.0, device=self.accelerator.device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
                retain_norm_sq += torch.sum(retain_grad * retain_grad)
                forget_norm_sq += torch.sum(forget_grad * forget_grad)
                dot_product += torch.dot(retain_grad.flatten(), forget_grad.flatten())
        retain_norm = torch.sqrt(retain_norm_sq)
        forget_norm = torch.sqrt(forget_norm_sq)
        self.logs["full_retain_grad_norm"].append(retain_norm.to(device="cpu", non_blocking=True))
        self.logs["full_forget_grad_norm"].append(forget_norm.to(device="cpu", non_blocking=True))
        self.logs["full_gradient_dot_product"].append(dot_product.to(device="cpu", non_blocking=True))
        return {"full_retain_norm": retain_norm, "full_forget_norm": forget_norm, "full_dot_product": dot_product}
