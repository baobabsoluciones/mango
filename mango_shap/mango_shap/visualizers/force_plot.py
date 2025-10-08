from typing import List, Optional, Union

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class ForcePlot:
    """
    Create force plots for SHAP values.

    Generates force plots that visualize the contribution of each feature
    to a specific prediction instance, showing how features push the
    prediction higher or lower from the baseline value.

    Example:
        >>> plotter = ForcePlot()
        >>> plotter.plot(shap_values, data, instance_idx=0, show=True)
    """

    def __init__(self) -> None:
        """
        Initialize the force plot generator.

        Sets up logging and prepares the plotter for creating force plots.
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
        Create a force plot for a specific instance.

        Generates a force plot that shows how each feature contributes to
        the prediction for a specific instance, with features pushing the
        prediction higher (positive) or lower (negative) from the baseline.

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
        self.logger.info(f"Creating force plot for instance {instance_idx}")

        # Handle multi-class case
        if len(shap_values.shape) > 2:
            shap_values = shap_values[0]

        # Get SHAP values and data for the specific instance
        instance_shap = shap_values[instance_idx]
        if isinstance(data, pd.DataFrame):
            instance_data = data.iloc[instance_idx]
        else:
            instance_data = data[instance_idx]

        # Create the plot
        force_plot = shap.force_plot(
            base_value=0.0,
            shap_values=instance_shap,
            features=instance_data,
            feature_names=feature_names,
            matplotlib=False,
        )

        if save_path:
            # Save the plot to HTML file
            # The AdditiveForceVisualizer object doesn't have save methods
            # We'll save the HTML representation manually
            html_content = force_plot.html()
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"Force plot saved to {save_path}")

        if show:
            force_plot.show()
        else:
            # Close the plot if it has a close method
            if hasattr(force_plot, "close"):
                force_plot.close()
