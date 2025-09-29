"""Model-agnostic SHAP explainer using kernel method."""

from typing import Any, Union, Optional

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class KernelExplainer:
    """
    Model-agnostic SHAP explainer using kernel method.

    Can be used with any model that has a predict method.
    """

    def __init__(
        self, model: Any, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Initialize the kernel explainer.

        :param model: Trained model with predict method
        :param background_data: Background data for SHAP calculations
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
        Calculate SHAP values for the given data.

        :param data: Data to explain
        :param max_evals: Maximum number of evaluations
        :return: SHAP values
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
        Make the explainer callable.

        :param data: Data to explain
        :param max_evals: Maximum number of evaluations
        :return: SHAP values
        """
        return self.shap_values(data, max_evals)
