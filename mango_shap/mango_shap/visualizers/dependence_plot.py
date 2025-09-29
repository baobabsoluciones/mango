from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class DependencePlot:
    """
    Create dependence plots for SHAP values.

    Generates dependence plots that visualize the relationship between feature
    values and their corresponding SHAP values. Supports interaction effects
    and provides options for display and saving.

    Example:
        >>> plotter = DependencePlot()
        >>> plotter.plot(shap_values, data, feature_idx=0, show=True)
    """

    def __init__(self) -> None:
        """
        Initialize the dependence plot generator.

        Sets up logging and prepares the plotter for creating dependence plots.
        No parameters are required as the plotter is stateless.

        :return: None
        :rtype: None
        """
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

        Generates a scatter plot showing the relationship between a feature's
        values and its corresponding SHAP values. Optionally includes interaction
        effects with another feature through color coding.

        :param shap_values: SHAP values array to plot
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :param feature_idx: Index of the feature to analyze
        :type feature_idx: int
        :param interaction_feature: Optional index of feature for interaction effects
        :type interaction_feature: Optional[int]
        :param feature_names: Optional list of feature names for labels
        :type feature_names: Optional[List[str]]
        :param show: Whether to display the plot immediately
        :type show: bool
        :param save_path: Optional path to save the plot as image file
        :type save_path: Optional[str]
        :return: None
        :rtype: None

        Example:
            >>> plotter.plot(shap_values, data, feature_idx=0, interaction_feature=1)
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
