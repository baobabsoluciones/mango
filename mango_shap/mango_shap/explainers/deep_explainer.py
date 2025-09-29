from typing import Any, Union, Optional

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class DeepExplainer:
    """
    SHAP explainer for deep learning models using gradient-based methods.

    Provides SHAP explanations for deep learning models including TensorFlow/Keras
    and PyTorch models. Deep explainers use gradient-based methods to efficiently
    compute SHAP values for neural networks, making them much faster than
    model-agnostic approaches for deep learning models.

    :param model: Trained deep learning model (TensorFlow/Keras or PyTorch)
    :type model: Any
    :param background_data: Background dataset for SHAP calculations
    :type background_data: Union[np.ndarray, pd.DataFrame]

    Example:
        >>> import tensorflow as tf
        >>> explainer = DeepExplainer(model, background_data)
        >>> shap_values = explainer.shap_values(test_data)
    """

    def __init__(
        self, model: Any, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Initialize the deep explainer with model and background data.

        Sets up the SHAP deep explainer using the provided deep learning model
        and background data. The deep explainer uses gradient-based methods to
        efficiently compute SHAP values for neural networks.

        :param model: Trained deep learning model (TensorFlow/Keras or PyTorch)
        :type model: Any
        :param background_data: Background dataset for SHAP calculations
        :type background_data: Union[np.ndarray, pd.DataFrame]
        :return: None
        :rtype: None

        Example:
            >>> import tensorflow as tf
            >>> explainer = DeepExplainer(trained_model, background_data)
        """
        self.logger = get_configured_logger()
        self.model = model
        self.background_data = background_data

        # Create SHAP DeepExplainer
        self.explainer = shap.DeepExplainer(model, background_data)

        self.logger.info("Deep explainer initialized")

    def shap_values(
        self, data: Union[np.ndarray, pd.DataFrame], max_evals: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for the given input data using gradient-based methods.

        Computes SHAP values using gradient-based approximation specifically
        optimized for deep learning models. The deep explainer leverages the
        model's gradients to efficiently compute SHAP values without requiring
        model evaluations for each feature combination.

        :param data: Input data to generate SHAP explanations for
        :type data: Union[np.ndarray, pd.DataFrame]
        :param max_evals: Not used for deep explainer (kept for API compatibility)
        :type max_evals: Optional[int]
        :return: SHAP values array with shape (n_samples, n_features) for regression
                 or (n_samples, n_features, n_classes) for classification
        :rtype: np.ndarray

        Example:
            >>> shap_values = explainer.shap_values(test_data)
            >>> print(f"SHAP values shape: {shap_values.shape}")
        """
        self.logger.info("Calculating SHAP values using DeepExplainer")

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
        method for actual computation using gradient-based approximation.

        :param data: Input data to generate SHAP explanations for
        :type data: Union[np.ndarray, pd.DataFrame]
        :param max_evals: Not used for deep explainer (kept for API compatibility)
        :type max_evals: Optional[int]
        :return: SHAP values array with shape (n_samples, n_features) for regression
                 or (n_samples, n_features, n_classes) for classification
        :rtype: np.ndarray

        Example:
            >>> explainer = DeepExplainer(model, background_data)
            >>> # Direct call syntax
            >>> shap_values = explainer(test_data)
            >>> print(f"SHAP values shape: {shap_values.shape}")
        """
        return self.shap_values(data, max_evals)
