"""Mango SHAP - Model interpretability using SHAP values."""

from .explainer import SHAPExplainer
from .const import (
    TREE_EXPLAINER_MODELS,
    KERNEL_EXPLAINER_MODELS,
    DEEP_EXPLAINER_MODELS,
    LINEAR_EXPLAINER_MODELS,
)

__all__ = [
    "SHAPExplainer",
    "TREE_EXPLAINER_MODELS",
    "KERNEL_EXPLAINER_MODELS",
    "DEEP_EXPLAINER_MODELS",
    "LINEAR_EXPLAINER_MODELS",
]
