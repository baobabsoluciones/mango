from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class SummaryPlot:
    """
    Create summary plots for SHAP values.

    Generates summary plots that display feature importance and the
    distribution of SHAP values across all features, providing a
    comprehensive overview of model behavior and feature contributions.

    Example:
        >>> plotter = SummaryPlot()
        >>> plotter.plot(shap_values, data, max_display=10, show=True)
    """

    def __init__(self) -> None:
        """
        Initialize the summary plot generator.

        Sets up logging and prepares the plotter for creating summary plots.
        No parameters are required as the plotter is stateless.

        :return: None
        :rtype: None
        """
        self.logger = get_configured_logger()

    def plot(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        max_display: int = 20,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a summary plot of SHAP values.

        Generates a summary plot that shows feature importance and the
        distribution of SHAP values across all features. Features are
        ordered by importance and color-coded by their impact on predictions.

        :param shap_values: SHAP values array to plot
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :param feature_names: Optional list of feature names for labels
        :type feature_names: Optional[List[str]]
        :param max_display: Maximum number of features to display in the plot
        :type max_display: int
        :param show: Whether to display the plot immediately
        :type show: bool
        :param save_path: Optional path to save the plot as image file
        :type save_path: Optional[str]
        :return: None
        :rtype: None

        Example:
            >>> plotter.plot(shap_values, data, max_display=10, show=True)
        """
        self.logger.info("Creating summary plot")

        # Handle multi-class case
        if len(shap_values.shape) > 2:
            shap_values = shap_values[0]

        # Create the plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            data,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Summary plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
