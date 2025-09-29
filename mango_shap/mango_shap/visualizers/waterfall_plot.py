from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class WaterfallPlot:
    """
    Create waterfall plots for SHAP values.

    Generates waterfall plots that show the step-by-step contribution of
    each feature to a specific prediction, starting from the baseline
    and building up to the final prediction value.

    Example:
        >>> plotter = WaterfallPlot()
        >>> plotter.plot(shap_values, data, instance_idx=0, show=True)
    """

    def __init__(self) -> None:
        """
        Initialize the waterfall plot generator.

        Sets up logging and prepares the plotter for creating waterfall plots.
        No parameters are required as the plotter is stateless.

        :return: None
        :rtype: None
        """
        self.logger = get_configured_logger()

    def plot(
        self,
        shap_values: np.ndarray,
        data: Union[np.ndarray, pd.DataFrame],
        instance_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        show: bool = True,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create a waterfall plot for a specific instance.

        Generates a waterfall plot that shows the step-by-step contribution
        of each feature to the prediction for a specific instance, starting
        from the baseline and building up to the final prediction value.

        :param shap_values: SHAP values array to plot
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :param instance_idx: Index of the instance to visualize
        :type instance_idx: int
        :param feature_names: Optional list of feature names for labels
        :type feature_names: Optional[List[str]]
        :param show: Whether to display the plot immediately
        :type show: bool
        :param save_path: Optional path to save the plot as image file
        :type save_path: Optional[str]
        :return: None
        :rtype: None

        Example:
            >>> plotter.plot(shap_values, data, instance_idx=0, show=True)
        """
        self.logger.info(f"Creating waterfall plot for instance {instance_idx}")

        # Handle multi-class case
        if len(shap_values.shape) > 2:
            shap_values = shap_values[0]  # Use first class for multi-class

        # Get SHAP values and data for the specific instance
        instance_shap = shap_values[instance_idx]
        instance_data = (
            data[instance_idx]
            if hasattr(data, "__getitem__")
            else data.iloc[instance_idx]
        )

        # Create the plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=instance_shap, data=instance_data, feature_names=feature_names
            ),
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Waterfall plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
