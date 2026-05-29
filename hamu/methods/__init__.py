"""Unlearning methods and gradient transformations."""

from hamu.methods.core import (
    PAIRED_LOSS_SPECS,
    PairedLossSpec,
    average_objective_gradients,
    gdiff_equivalent_gradient,
    gru_equivalent_gradient,
    kl_equivalent_gradient,
    pcgrad_equivalent_gradient,
    scrub_equivalent_gradient,
)
from hamu.methods.hamu import HAMUOptimizer, HAMUUpdateResult, compute_hamu_update
from hamu.methods.baselines import (
    BASELINE_METHODS,
    BASELINE_REGISTRY,
    GRUOptimizer,
    PCGradOptimizer,
    BaselineMethod,
    get_baseline_method,
)

__all__ = [
    "BASELINE_METHODS",
    "BASELINE_REGISTRY",
    "BaselineMethod",
    "HAMUOptimizer",
    "HAMUUpdateResult",
    "PAIRED_LOSS_SPECS",
    "PairedLossSpec",
    "average_objective_gradients",
    "compute_hamu_update",
    "gdiff_equivalent_gradient",
    "get_baseline_method",
    "gru_equivalent_gradient",
    "GRUOptimizer",
    "kl_equivalent_gradient",
    "pcgrad_equivalent_gradient",
    "PCGradOptimizer",
    "scrub_equivalent_gradient",
]
