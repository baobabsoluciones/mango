"""
Mango SHAP - Comprehensive SHAP analysis for machine learning model interpretability.

This package provides a unified interface for generating SHAP (SHapley Additive exPlanations)
explanations for various types of machine learning models. It supports tree-based models,
neural networks, linear models, and kernel-based models with automatic model type detection.

Key Features:
    - Automatic model type detection (tree, deep, linear, kernel)
    - Support for multiple problem types (regression, binary/multiclass classification)
    - Metadata handling and filtering
    - Comprehensive analysis workflows
    - Export capabilities for results and visualizations
    - Integration with popular ML frameworks (scikit-learn, XGBoost, LightGBM, etc.)

Main Classes:
    - SHAPExplainer: Main class for SHAP analysis and model interpretability

Example:
    >>> from mango_shap import SHAPExplainer
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Train a model
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Create explainer and generate explanations
    >>> explainer = SHAPExplainer(
    ...     model=model,
    ...     data=X_train,
    ...     problem_type="binary_classification"
    ... )
    >>> shap_values = explainer.explain(X_test)
"""

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
