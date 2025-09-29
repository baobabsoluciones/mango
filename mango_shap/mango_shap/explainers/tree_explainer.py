"""Tree-based model SHAP explainer."""

from typing import Any, Union, Optional

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class TreeExplainer:
    """
    SHAP explainer for tree-based models.

    Supports XGBoost, LightGBM, CatBoost, and scikit-learn tree models.
    """

    def __init__(
        self, model: Any, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Initialize the tree explainer.

        :param model: Trained tree-based model
        :param background_data: Background data for SHAP calculations
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
        Calculate SHAP values for the given data.

        :param data: Data to explain
        :param max_evals: Not used for tree explainer (kept for compatibility)
        :return: SHAP values
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
        Make the explainer callable.

        :param data: Data to explain
        :param max_evals: Not used for tree explainer
        :return: SHAP values
        """
        return self.shap_values(data, max_evals)
