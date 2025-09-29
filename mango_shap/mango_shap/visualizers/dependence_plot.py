"""Dependence plot visualization for SHAP values."""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class DependencePlot:
    """
    Create dependence plots for SHAP values.

    Shows how a feature's value affects its SHAP value.
    """

    def __init__(self) -> None:
        """Initialize the dependence plot generator."""
        self.logger = get_configured_logger()
        self.logger.info("DependencePlot initialized")

    def plot(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_idx: int,
        interaction_feature: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a dependence plot for a specific feature.

        :param shap_values: SHAP values to plot
        :param data: Data used to generate SHAP values
        :param feature_idx: Index of the feature to plot
        :param interaction_feature: Index of interaction feature
        :param feature_names: Names of features
        :param show: Whether to display the plot
        :param save_path: Path to save the plot
        """
        self.logger.info(f"Creating dependence plot for feature {feature_idx}")

        # Handle multi-class case
        if len(shap_values.shape) > 2:
            shap_values = shap_values[0]

        # Create the plot
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            data,
            interaction_index=interaction_feature,
            feature_names=feature_names,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Dependence plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
