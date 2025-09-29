from typing import Any, Union, Optional

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class TreeExplainer:
    """
    SHAP explainer for tree-based machine learning models.

    Provides efficient and exact SHAP explanations for tree-based models including
    XGBoost, LightGBM, CatBoost, and scikit-learn tree models. Tree explainers
    leverage the tree structure to compute SHAP values exactly and efficiently.

    :param model: Trained tree-based model (RandomForest, XGBoost, LightGBM, etc.)
    :type model: Any
    :param background_data: Background dataset for SHAP calculations
    :type background_data: Union[np.ndarray, pd.DataFrame]

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> explainer = TreeExplainer(model, background_data)
        >>> shap_values = explainer.shap_values(test_data)
    """

    def __init__(
        self, model: Any, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Initialize the tree explainer with model and background data.

        Sets up the SHAP tree explainer using the provided model and background data.
        The background data is used to compute expected values and for efficient
        SHAP value calculations.

        :param model: Trained tree-based model
        :type model: Any
        :param background_data: Background dataset for SHAP calculations
        :type background_data: Union[np.ndarray, pd.DataFrame]
        :return: None
        :rtype: None

        Example:
            >>> explainer = TreeExplainer(trained_model, background_data)
        """
        self.logger = get_configured_logger()
        self.model = model
        self.background_data = background_data

        # Create SHAP TreeExplainer
        self.explainer = shap.TreeExplainer(model, background_data)

        self.logger.info("Tree explainer initialized")

    def shap_values(
        self, data: Union[np.ndarray, pd.DataFrame], max_evals: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for the given input data.

        Computes exact SHAP values for tree-based models using the efficient
        tree structure. The method leverages the tree explainer's ability to
        compute exact values without approximation.

        :param data: Input data to generate SHAP explanations for
        :type data: Union[np.ndarray, pd.DataFrame]
        :param max_evals: Not used for tree explainer (kept for API compatibility)
        :type max_evals: Optional[int]
        :return: SHAP values array with shape (n_samples, n_features) for regression
                 or (n_samples, n_features, n_classes) for classification
        :rtype: np.ndarray

        Example:
            >>> shap_values = explainer.shap_values(test_data)
            >>> print(f"SHAP values shape: {shap_values.shape}")
        """
        self.logger.info("Calculating SHAP values using TreeExplainer")

        shap_values = self.explainer.shap_values(data)

        # Handle multi-class case
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        self.logger.info(f"Generated SHAP values with shape: {shap_values.shape}")
        return shap_values

    def __call__(
        self, data: Union[np.ndarray, pd.DataFrame], max_evals: Optional[int] = None
    ) -> np.ndarray:
        """
        Make the explainer callable for direct SHAP value computation.

        Allows the explainer to be used as a callable object, providing a convenient
        interface for generating SHAP values. This method delegates to the shap_values
        method for actual computation.

        :param data: Input data to generate SHAP explanations for
        :type data: Union[np.ndarray, pd.DataFrame]
        :param max_evals: Not used for tree explainer (kept for API compatibility)
        :type max_evals: Optional[int]
        :return: SHAP values array with shape (n_samples, n_features) for regression
                 or (n_samples, n_features, n_classes) for classification
        :rtype: np.ndarray

        Example:
            >>> explainer = TreeExplainer(model, background_data)
            >>> # Direct call syntax
            >>> shap_values = explainer(test_data)
            >>> print(f"SHAP values shape: {shap_values.shape}")
        """
        return self.shap_values(data, max_evals)
