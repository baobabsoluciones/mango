"""Linear model SHAP explainer."""

import numpy as np
import pandas as pd
from typing import Any, Union
import shap
from ..logging.logger import get_logger


class LinearExplainer:
    """
    SHAP explainer for linear models.

    Supports scikit-learn linear models and generalized linear models.
    """

    def __init__(
        self, model: Any, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Initialize the linear explainer.

        :param model: Trained linear model
        :param background_data: Background data for SHAP calculations
        """
        self.logger = get_logger(__name__)
        self.model = model
        self.background_data = background_data

        # Create SHAP LinearExplainer
        self.explainer = shap.LinearExplainer(model, background_data)

        self.logger.info("Linear explainer initialized")

    def shap_values(
        self, data: Union[np.ndarray, pd.DataFrame], max_evals: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for the given data.

        :param data: Data to explain
        :param max_evals: Not used for linear explainer (kept for compatibility)
        :return: SHAP values
        """
        self.logger.info("Calculating SHAP values using LinearExplainer")

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
        :param max_evals: Not used for linear explainer
        :return: SHAP values
        """
        return self.shap_values(data, max_evals)
