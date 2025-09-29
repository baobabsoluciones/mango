"""Deep learning model SHAP explainer."""

import numpy as np
import pandas as pd
from typing import Any, Union
import shap
from ..logging.logger import get_logger


class DeepExplainer:
    """
    SHAP explainer for deep learning models.

    Supports TensorFlow/Keras and PyTorch models.
    """

    def __init__(
        self, model: Any, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Initialize the deep explainer.

        :param model: Trained deep learning model
        :param background_data: Background data for SHAP calculations
        """
        self.logger = get_logger(__name__)
        self.model = model
        self.background_data = background_data

        # Create SHAP DeepExplainer
        self.explainer = shap.DeepExplainer(model, background_data)

        self.logger.info("Deep explainer initialized")

    def shap_values(
        self, data: Union[np.ndarray, pd.DataFrame], max_evals: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for the given data.

        :param data: Data to explain
        :param max_evals: Not used for deep explainer (kept for compatibility)
        :return: SHAP values
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
        Make the explainer callable.

        :param data: Data to explain
        :param max_evals: Not used for deep explainer
        :return: SHAP values
        """
        return self.shap_values(data, max_evals)
