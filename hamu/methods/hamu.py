"""HAMU gradient transforms.

This module contains the public implementation of HAMU-Q and HAMU-U.  The
optimizer computes an equivalent gradient from paired retain/forget gradients,
then delegates the actual parameter update to a standard PyTorch optimizer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Optional

import torch

from accelerate import Accelerator

from hamu.methods.core import HAMUUpdateResult, average_objective_gradients, compute_hamu_update


class GradientTransformOptimizer(torch.optim.Optimizer):
    """Base optimizer that rewrites gradients before delegating to an optimizer."""

    def __init__(
        self,
        model: torch.nn.Module,
        defaults: dict,
        optimizer: Optional[torch.optim.Optimizer] = None,
        gradient_pairing: str = "split-gpu",
    ) -> None:
        if isinstance(optimizer, torch.optim.LBFGS):
            raise TypeError("LBFGS is not supported for HAMU gradient transforms.")
        params = list(model.parameters())
        super().__init__(params, defaults)
        self.model = model
        self.gradient_pairing = gradient_pairing
        self.logs: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
        self._retain_grads: dict[int, torch.Tensor] = {}
        self.stop_reason: Optional[str] = None
        if optimizer is None:
            optimizer = torch.optim.SGD(self.param_groups, lr=defaults.get("lr", 1e-3), momentum=0.0)
        self.optimizer = optimizer

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Any:
        return self.optimizer.step(closure)

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        return self.optimizer.load_state_dict(state_dict)

    def add_param_group(self, param_group: dict[str, Any]) -> Any:
        if not hasattr(self, "optimizer"):
            return super().add_param_group(param_group)
        return self.optimizer.add_param_group(param_group)

    def set_accelerator(self, accelerator: Accelerator) -> None:
        self.accelerator = accelerator

    def store_retain_gradients(self) -> None:
        self._retain_grads = {
            id(p): p.grad.detach().clone()
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        }

    def clear_retain_gradients(self) -> None:
        self._retain_grads.clear()

    def preprocess_params(self) -> dict:
        return {}

    def _step(self) -> None:
        processed_params = self.preprocess_params()
        skip_gradient_transform = bool(processed_params.pop("skip_gradient_transform", False))
        if skip_gradient_transform:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad = None
            return
        for group in self.param_groups:
            values = {**group, **processed_params}
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.grad = self.get_grad(p, **values)
                if self.stop_reason == "failed_feasibility":
                    for clear_group in self.param_groups:
                        for clear_param in clear_group["params"]:
                            clear_param.grad = None
                    return

    def collate_retain_forget_gradients(self, p: torch.nn.Parameter) -> tuple[torch.Tensor, torch.Tensor]:
        if self.gradient_pairing == "sequential":
            retain_grad = self._retain_grads.get(id(p))
            if retain_grad is None:
                raise RuntimeError("Missing retained gradient for sequential paired-gradient step.")
            return retain_grad.contiguous(), p.grad.contiguous()

        local_grad = p.grad.contiguous()
        if self.accelerator.num_processes % 2 != 0:
            raise RuntimeError("split-gpu gradient pairing requires an even number of processes.")
        all_grads = self.accelerator.gather(local_grad)
        grads = all_grads.reshape(self.accelerator.num_processes // 2, 2, *local_grad.shape)
        grads = grads.mean(dim=0)
        return grads[0], grads[1]

    def get_grad(self, p: torch.nn.Parameter, **kwargs) -> torch.Tensor:
        if self.gradient_pairing == "sequential":
            retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
            return average_objective_gradients(retain_grad, forget_grad)
        local_grad = p.grad.contiguous()
        return self.accelerator.gather(local_grad).reshape(-1, *local_grad.shape).mean(dim=0)


class HAMUOptimizer(GradientTransformOptimizer):
    """Optimizer wrapper for HAMU-Q and HAMU-U."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        *,
        lr: Optional[float] = None,
        radius: Optional[float] = 1e-3,
        constraint: float = 0.0,
        variant: str = "hamu-q",
        stop_on_failed_feasibility: bool = False,
        stop_on_stopping_criterion: bool = False,
        full_grad: bool = False,
        distribute_constraint_softmax_dp: bool = False,
        distribute_constraint_grad_norm: bool = False,
        normalize_distributed_constraint: bool = True,
        gradient_pairing: str = "split-gpu",
    ) -> None:
        if constraint < 0:
            raise ValueError("HAMU constraint must be non-negative.")
        if variant not in {"hamu-q", "hamu-u"}:
            raise ValueError(f"Unsupported HAMU variant: {variant}")

        first_param = next(model.parameters())
        self.variant = variant
        self.use_lr_radius = lr is not None
        self.full_grad = full_grad
        self.stop_on_failed_feasibility = stop_on_failed_feasibility
        self.stop_on_stopping_criterion = stop_on_stopping_criterion
        self.distribute_constraint_softmax_dp = distribute_constraint_softmax_dp
        self.distribute_constraint_grad_norm = distribute_constraint_grad_norm
        self.normalize_distributed_constraint = normalize_distributed_constraint
        effective_lr = 1.0 if lr is None else lr
        super().__init__(
            model,
            defaults={"lr": effective_lr, "radius": radius, "constraint": constraint},
            optimizer=optimizer,
            gradient_pairing=gradient_pairing,
        )
        self._lr = torch.tensor(effective_lr, device=first_param.device)
        self._radius = torch.tensor(-1.0 if radius is None else radius, device=first_param.device)
        self._constraint = torch.tensor(constraint, device=first_param.device)

    def set_accelerator(self, accelerator: Accelerator) -> None:
        super().set_accelerator(accelerator)
        self.stats = torch.zeros(4, dtype=torch.long, device=accelerator.device)

    def _append_update_logs(self, result: HAMUUpdateResult) -> None:
        legacy_names = {
            "radius": "R",
            "constraint": "B",
            "forget_grad_norm": "gf_norm",
            "retain_grad_norm": "gr_norm",
            "orthogonal_retain_grad_norm": "gr_n_norm",
            "gradient_dot_product": "gdp",
        }
        for key, value in result.logs.items():
            self.logs[key].append(value.to(device="cpu", non_blocking=True))
            if key in legacy_names:
                self.logs[legacy_names[key]].append(value.to(device="cpu", non_blocking=True))

        if result.branch == "direct":
            self.stats[0] += 1
        elif result.branch in {"rectified", "colinear", "zero_forget_grad"}:
            self.stats[1] += 1
        if result.failed_feasibility:
            self.stats[2] += 1
        if result.nan_in_delta:
            self.stats[3] += 1

    def get_grad(
        self,
        p: torch.nn.Parameter,
        lr: torch.Tensor | float = torch.tensor(1.0),
        radius: torch.Tensor | float = torch.tensor(1.0),
        constraint: torch.Tensor | float = torch.tensor(0.0),
        retain_norm: Optional[torch.Tensor] = None,
        forget_norm: Optional[torch.Tensor] = None,
        dot_product: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
        local_constraint = constraint
        if not self.full_grad and (self.distribute_constraint_softmax_dp or self.distribute_constraint_grad_norm):
            effective_retain, effective_forget = self._effective_gradients(retain_grad, forget_grad)
            if self.distribute_constraint_softmax_dp:
                local_dot_product = torch.dot(effective_retain.flatten(), effective_forget.flatten())
                local_constraint = constraint * torch.exp(-local_dot_product)
            else:
                local_constraint = constraint * torch.linalg.norm(effective_retain) * torch.linalg.norm(effective_forget)
        result = compute_hamu_update(
            retain_grad,
            forget_grad,
            lr=lr,
            radius=radius,
            constraint=local_constraint,
            variant=self.variant,
            use_lr_radius=self.use_lr_radius,
            stop_on_infeasible=self.stop_on_failed_feasibility,
            full_retain_norm=retain_norm,
            full_forget_norm=forget_norm,
            full_dot_product=dot_product,
        )
        self._append_update_logs(result)
        if result.failed_feasibility and self.stop_on_failed_feasibility:
            self.stop_reason = "failed_feasibility"
        return -result.update / lr

    def _effective_gradients(self, retain_grad: torch.Tensor, forget_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.variant == "hamu-u":
            return -forget_grad, -retain_grad
        return retain_grad, forget_grad

    def _stopping_rhs_for_block(
        self,
        retain_grad: torch.Tensor,
        forget_grad: torch.Tensor,
        *,
        lr: torch.Tensor,
        radius: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        gr, gf = self._effective_gradients(retain_grad, forget_grad)
        gr_norm_sq = torch.sum(gr * gr)
        gf_norm_sq = torch.sum(gf * gf)
        dot_product = torch.dot(gr.flatten(), gf.flatten())
        gr_norm = torch.sqrt(gr_norm_sq)
        block_radius = lr * gr_norm if self.use_lr_radius else radius.to(device=gr.device)
        # Appendix stopping term for this layer/block.
        kappa2_numerator = torch.sqrt(torch.clamp(gr_norm_sq * gf_norm_sq - dot_product * dot_product, min=0.0))
        return block_radius * kappa2_numerator / torch.clamp(gr_norm, min=eps)

    def _append_stopping_logs(self, rhs: torch.Tensor, constraint: torch.Tensor) -> bool:
        rhs = rhs.detach()
        constraint = constraint.to(device=rhs.device).detach()
        margin = constraint - rhs
        stopping_met = bool((margin > 0).item())
        self.logs["aggregate_stopping_rhs"].append(rhs.to(device="cpu", non_blocking=True))
        self.logs["stopping_constraint"].append(constraint.to(device="cpu", non_blocking=True))
        self.logs["stopping_margin"].append(margin.to(device="cpu", non_blocking=True))
        self.logs["stopping_met"].append(torch.tensor(float(stopping_met), device=rhs.device).to(device="cpu", non_blocking=True))
        return stopping_met

    def preprocess_params(self) -> dict[str, torch.Tensor | None | bool]:
        constraint = self._constraint
        stopping_constraint = self._constraint
        retain_norm = None
        forget_norm = None
        dot_product = None
        stopping_rhs = torch.tensor(0.0, device=self.accelerator.device)

        if self.full_grad:
            retain_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
            forget_norm_sq = torch.tensor(0.0, device=self.accelerator.device)
            dot_product = torch.tensor(0.0, device=self.accelerator.device)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
                    effective_retain, effective_forget = self._effective_gradients(retain_grad, forget_grad)
                    retain_norm_sq += torch.sum(effective_retain * effective_retain)
                    forget_norm_sq += torch.sum(effective_forget * effective_forget)
                    dot_product += torch.dot(effective_retain.flatten(), effective_forget.flatten())
            retain_norm = torch.sqrt(retain_norm_sq)
            forget_norm = torch.sqrt(forget_norm_sq)
            full_radius = self._lr * retain_norm if self.use_lr_radius else self._radius
            stopping_rhs = full_radius * torch.sqrt(
                torch.clamp(retain_norm_sq * forget_norm_sq - dot_product * dot_product, min=0.0)
            ) / torch.clamp(retain_norm, min=1e-12)
            self.logs["full_retain_grad_norm"].append(retain_norm.to(device="cpu", non_blocking=True))
            self.logs["full_forget_grad_norm"].append(forget_norm.to(device="cpu", non_blocking=True))
            self.logs["full_gradient_dot_product"].append(dot_product.to(device="cpu", non_blocking=True))
        else:
            n_tensors = 0
            grad_norm_product = torch.tensor(0.0, device=self.accelerator.device)
            exp_neg_dp = torch.tensor(0.0, device=self.accelerator.device)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    n_tensors += 1
                    retain_grad, forget_grad = self.collate_retain_forget_gradients(p)
                    stopping_rhs += self._stopping_rhs_for_block(
                        retain_grad,
                        forget_grad,
                        lr=self._lr,
                        radius=self._radius,
                    )
                    effective_retain, effective_forget = self._effective_gradients(retain_grad, forget_grad)
                    if self.distribute_constraint_softmax_dp:
                        exp_neg_dp += torch.exp(-torch.dot(effective_retain.flatten(), effective_forget.flatten()))
                    elif self.distribute_constraint_grad_norm:
                        grad_norm_product += torch.linalg.norm(effective_retain) * torch.linalg.norm(effective_forget)

            if self.distribute_constraint_softmax_dp:
                self.logs["constraint_exp_neg_dp"].append(exp_neg_dp.to(device="cpu", non_blocking=True))
                if self.normalize_distributed_constraint:
                    constraint = constraint / torch.clamp(exp_neg_dp, min=1e-12)
                else:
                    stopping_constraint = constraint * exp_neg_dp
            elif self.distribute_constraint_grad_norm:
                self.logs["constraint_grad_norm_product"].append(grad_norm_product.to(device="cpu", non_blocking=True))
                if self.normalize_distributed_constraint:
                    constraint = constraint / torch.clamp(grad_norm_product, min=1e-12)
                else:
                    stopping_constraint = constraint * grad_norm_product
            else:
                constraint = constraint / max(n_tensors, 1)

        stopping_met = self._append_stopping_logs(stopping_rhs, stopping_constraint)
        skip_gradient_transform = self.stop_on_stopping_criterion and stopping_met
        if skip_gradient_transform:
            self.stop_reason = "stopping_criterion"

        return {
            "lr": self._lr,
            "radius": self._radius,
            "constraint": constraint,
            "retain_norm": retain_norm,
            "forget_norm": forget_norm,
            "dot_product": dot_product,
            "skip_gradient_transform": skip_gradient_transform,
        }
