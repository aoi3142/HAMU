"""Core HAMU and baseline gradient formulas.

This file intentionally keeps the method-level math compact.  Trainer code is
responsible for producing the relevant gradients; the functions here only
describe how paired retain/forget gradients are converted into an equivalent
optimizer gradient or HAMU weight update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class HAMUUpdateResult:
    """Result of a single HAMU layer/tensor update."""

    update: torch.Tensor
    branch: str
    failed_feasibility: bool
    nan_in_delta: bool
    logs: dict[str, torch.Tensor]


@dataclass(frozen=True)
class PairedLossSpec:
    """Human-readable paired retain/forget objectives for baseline methods."""

    retain_objective: str
    forget_objective: str


PAIRED_LOSS_SPECS = {
    "gdiff": PairedLossSpec("retain_loss", "-forget_loss"),
    "kl": PairedLossSpec("retain_kl_to_full_model", "-forget_loss"),
    "scrub": PairedLossSpec("alpha * retain_kl_to_full_model + gamma * retain_loss", "-forget_kl_to_full_model"),
}


def _as_tensor(value: torch.Tensor | float | int, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.tensor(value, device=device, dtype=torch.float32)


def average_objective_gradients(retain_objective_grad: torch.Tensor, forget_objective_grad: torch.Tensor) -> torch.Tensor:
    """Average two already-signed objective gradients.

    GDiff, KL, and SCRUB create one objective on retain batches and one objective
    on forget batches.  Split-GPU mode averages those objective gradients across
    processes; sequential mode should produce the same equivalent gradient.
    """

    return (retain_objective_grad + forget_objective_grad) / 2.0


def gdiff_equivalent_gradient(retain_grad: torch.Tensor, forget_grad: torch.Tensor) -> torch.Tensor:
    """Gradient difference baseline using raw retain/forget loss gradients."""

    return average_objective_gradients(retain_grad, -forget_grad)


def kl_equivalent_gradient(retain_kl_grad: torch.Tensor, forget_grad: torch.Tensor) -> torch.Tensor:
    """KL baseline using retain KL gradient and raw forget loss gradient."""

    return average_objective_gradients(retain_kl_grad, -forget_grad)


def scrub_equivalent_gradient(
    retain_grad: torch.Tensor,
    retain_kl_grad: torch.Tensor,
    forget_kl_grad: torch.Tensor,
    *,
    alpha: float = 0.001,
    gamma: float = 0.99,
) -> torch.Tensor:
    """SCRUB baseline using raw component gradients."""

    retain_objective_grad = alpha * retain_kl_grad + gamma * retain_grad
    return average_objective_gradients(retain_objective_grad, -forget_kl_grad)


def gru_equivalent_gradient(
    retain_grad: torch.Tensor,
    forget_grad: torch.Tensor,
    *,
    eps: float = 1e-12,
    tau: Optional[float] = None,
    full_retain_norm: Optional[torch.Tensor] = None,
    full_dot_product: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gradient Rectified Unlearning equivalent optimizer gradient."""

    dot_product = (
        full_dot_product
        if full_dot_product is not None
        else torch.dot(retain_grad.flatten(), forget_grad.flatten())
    )
    retain_norm_sq = (
        full_retain_norm * full_retain_norm
        if full_retain_norm is not None
        else torch.sum(retain_grad * retain_grad)
    )
    if dot_product < 0:
        update = forget_grad - dot_product / torch.clamp(retain_norm_sq, min=eps) * retain_grad
    else:
        update = forget_grad
    if tau is not None:
        update_norm = torch.linalg.norm(update)
        if update_norm > tau:
            update = update / update_norm * tau
    return -update


def pcgrad_equivalent_gradient(
    retain_grad: torch.Tensor,
    forget_grad: torch.Tensor,
    *,
    eps: float = 1e-12,
    tau: Optional[float] = None,
    full_retain_norm: Optional[torch.Tensor] = None,
    full_forget_norm: Optional[torch.Tensor] = None,
    full_dot_product: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PCGrad-style unlearning equivalent optimizer gradient."""

    dot_product = (
        full_dot_product
        if full_dot_product is not None
        else torch.dot(retain_grad.flatten(), forget_grad.flatten())
    )
    retain_norm_sq = (
        full_retain_norm * full_retain_norm
        if full_retain_norm is not None
        else torch.sum(retain_grad * retain_grad)
    )
    forget_norm_sq = (
        full_forget_norm * full_forget_norm
        if full_forget_norm is not None
        else torch.sum(forget_grad * forget_grad)
    )
    if dot_product < 0:
        update = (
            (-forget_grad + dot_product / torch.clamp(retain_norm_sq, min=eps) * retain_grad)
            + (retain_grad + dot_product / torch.clamp(forget_norm_sq, min=eps) * -forget_grad)
        ) / 2.0
    else:
        update = (retain_grad - forget_grad) / 2.0
    if tau is not None:
        update_norm = torch.linalg.norm(update)
        if update_norm > tau:
            update = update / update_norm * tau
    return update


def compute_hamu_update(
    retain_grad: torch.Tensor,
    forget_grad: torch.Tensor,
    *,
    lr: torch.Tensor | float,
    radius: torch.Tensor | float,
    constraint: torch.Tensor | float,
    variant: str = "hamu-q",
    use_lr_radius: bool = False,
    eps: float = 1e-12,
    stop_on_infeasible: bool = False,
    full_retain_norm: Optional[torch.Tensor] = None,
    full_forget_norm: Optional[torch.Tensor] = None,
    full_dot_product: Optional[torch.Tensor] = None,
) -> HAMUUpdateResult:
    """Compute the HAMU weight update for one tensor."""

    if variant not in {"hamu-q", "hamu-u"}:
        raise ValueError(f"Unsupported HAMU variant: {variant}")

    gr = retain_grad
    gf = forget_grad
    if variant == "hamu-u":
        gr, gf = -gf, -gr

    device = gr.device
    lr = _as_tensor(lr, device)
    radius = _as_tensor(radius, device)
    constraint = _as_tensor(constraint, device)

    if full_retain_norm is None:
        gr_norm_sq = torch.sum(gr * gr)
        gr_norm = torch.sqrt(gr_norm_sq)
    else:
        gr_norm = full_retain_norm.to(device=device)
        gr_norm_sq = gr_norm * gr_norm

    if full_forget_norm is None:
        gf_norm_sq = torch.sum(gf * gf)
        gf_norm = torch.sqrt(gf_norm_sq)
    else:
        gf_norm = full_forget_norm.to(device=device)
        gf_norm_sq = gf_norm * gf_norm

    if full_dot_product is None:
        dot_product = torch.dot(gr.flatten(), gf.flatten())
    else:
        dot_product = full_dot_product.to(device=device)

    if use_lr_radius:
        radius = lr * gr_norm

    safe_gr_norm = torch.clamp(gr_norm, min=eps)
    safe_gf_norm_sq = torch.clamp(gf_norm_sq, min=eps)
    safe_gf_norm = torch.sqrt(safe_gf_norm_sq)
    threshold = -constraint * safe_gr_norm / torch.clamp(radius, min=eps)
    discriminant = radius * radius - constraint * constraint / safe_gf_norm_sq
    gr_orthogonal = gr - (dot_product / safe_gf_norm_sq) * gf
    gr_orthogonal_norm = torch.linalg.norm(gr_orthogonal)
    constraint_over_gf_norm = constraint / safe_gf_norm

    denom = torch.clamp(gr_norm * gf_norm, min=eps)
    angle = torch.rad2deg(torch.acos(torch.clamp(dot_product / denom, -1.0, 1.0)))
    logs = {
        "radius": radius.detach(),
        "constraint": constraint.detach(),
        "forget_grad_norm": gf_norm.detach(),
        "retain_grad_norm": gr_norm.detach(),
        "orthogonal_retain_grad_norm": gr_orthogonal_norm.detach(),
        "gradient_dot_product": dot_product.detach(),
        "angle": angle.detach(),
        "feasibility": (constraint_over_gf_norm / torch.clamp(radius, min=eps)).detach(),
        "branch_threshold": threshold.detach(),
        "stopping_criterion": torch.sqrt(torch.clamp(gf_norm_sq * gr_norm_sq - dot_product * dot_product, min=0.0)).detach(),
        "stopping_threshold": (constraint / torch.clamp(lr, min=eps)).detach(),
    }

    failed_feasibility = False
    nan_in_delta = False

    if bool((gf_norm_sq < eps).item()):
        if bool((constraint <= eps).item()):
            update = -(radius / torch.clamp(gr_norm, min=eps)) * gr
            branch = "direct"
        elif stop_on_infeasible:
            update = torch.full_like(gr, float("nan"))
            branch = "infeasible"
            failed_feasibility = True
            nan_in_delta = True
        else:
            update = torch.zeros_like(gr)
            branch = "zero_forget_grad"
            failed_feasibility = True
        return HAMUUpdateResult(update, branch, failed_feasibility, nan_in_delta, logs)

    ascent_update = (constraint / safe_gf_norm_sq) * gf
    if bool((discriminant < 0).item()):
        failed_feasibility = True
        if stop_on_infeasible:
            update = torch.full_like(gr, float("nan"))
            nan_in_delta = True
        else:
            update = ascent_update
        return HAMUUpdateResult(update, "infeasible", failed_feasibility, nan_in_delta, logs)

    if bool((dot_product <= threshold).item()):
        update = -(radius / torch.clamp(gr_norm, min=eps)) * gr
        return HAMUUpdateResult(update, "direct", failed_feasibility, nan_in_delta, logs)

    if bool((gr_orthogonal_norm < eps).item()):
        return HAMUUpdateResult(ascent_update, "colinear", failed_feasibility, nan_in_delta, logs)

    orthogonal_scale = radius if bool((constraint <= eps).item()) else torch.sqrt(discriminant)
    update = ascent_update - orthogonal_scale * gr_orthogonal / torch.clamp(gr_orthogonal_norm, min=eps)
    return HAMUUpdateResult(update, "rectified", failed_feasibility, nan_in_delta, logs)
