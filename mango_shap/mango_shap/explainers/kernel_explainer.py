from typing import Any, Union, Optional

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class KernelExplainer:
    """
    Model-agnostic SHAP explainer using kernel method.

    Provides SHAP explanations for any model that has a predict method using
    the kernel-based approach. This explainer is model-agnostic and uses
    sampling-based approximation, making it more computationally intensive
    but universally applicable to any model type.

    :param model: Trained model with predict method (any model type)
    :type model: Any
    :param background_data: Background dataset for SHAP calculations
    :type background_data: Union[np.ndarray, pd.DataFrame]

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> explainer = KernelExplainer(model, background_data)
        >>> shap_values = explainer.shap_values(test_data)
    """

    def __init__(
        self, model: Any, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Initialize the kernel explainer with model and background data.

        Sets up the SHAP kernel explainer using the provided model's predict method
        and background data. The kernel explainer uses sampling-based approximation
        to compute SHAP values for any model type.

        :param model: Trained model with predict method
        :type model: Any
        :param background_data: Background dataset for SHAP calculations
        :type background_data: Union[np.ndarray, pd.DataFrame]
        :return: None
        :rtype: None

        Example:
            >>> explainer = KernelExplainer(trained_model, background_data)
        """
        self.logger = get_configured_logger()
        self.model = model
        self.background_data = background_data

        # Create SHAP KernelExplainer
        self.explainer = shap.KernelExplainer(model.predict, background_data)

        self.logger.info("Kernel explainer initialized")

    def shap_values(
        self, data: Union[np.ndarray, pd.DataFrame], max_evals: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for the given input data using kernel method.

        Computes SHAP values using sampling-based approximation through the kernel
        method. The number of evaluations can be controlled to balance accuracy
        and computational cost. More evaluations provide better accuracy but
        increase computation time.

        :param data: Input data to generate SHAP explanations for
        :type data: Union[np.ndarray, pd.DataFrame]
        :param max_evals: Maximum number of evaluations for approximation (optional)
        :type max_evals: Optional[int]
        :return: SHAP values array with shape (n_samples, n_features) for regression
                 or (n_samples, n_features, n_classes) for classification
        :rtype: np.ndarray

        Example:
            >>> shap_values = explainer.shap_values(test_data, max_evals=200)
            >>> print(f"SHAP values shape: {shap_values.shape}")
        """
        self.logger.info("Calculating SHAP values using KernelExplainer")

        if max_evals is None:
            max_evals = min(100, 2 * data.shape[1] + 1)

        shap_values = self.explainer.shap_values(data, nsamples=max_evals)

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
        method for actual computation using kernel-based approximation.

        :param data: Input data to generate SHAP explanations for
        :type data: Union[np.ndarray, pd.DataFrame]
        :param max_evals: Maximum number of evaluations for approximation (optional)
        :type max_evals: Optional[int]
        :return: SHAP values array with shape (n_samples, n_features) for regression
                 or (n_samples, n_features, n_classes) for classification
        :rtype: np.ndarray

        Example:
            >>> explainer = KernelExplainer(model, background_data)
            >>> # Direct call syntax
            >>> shap_values = explainer(test_data, max_evals=200)
            >>> print(f"SHAP values shape: {shap_values.shape}")
        """
        return self.shap_values(data, max_evals)
