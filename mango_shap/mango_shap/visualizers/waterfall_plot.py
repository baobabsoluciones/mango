"""Waterfall plot visualization for SHAP values."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import shap
from ..logging.logger import get_logger


class WaterfallPlot:
    """
    Create waterfall plots for SHAP values.

    Shows how each feature contributes to the final prediction.
    """

    def __init__(self) -> None:
        """Initialize the waterfall plot generator."""
        self.logger = get_logger(__name__)

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

        :param shap_values: SHAP values to plot
        :param data: Data used to generate SHAP values
        :param instance_idx: Index of the instance to plot
        :param feature_names: Names of features
        :param show: Whether to display the plot
        :param save_path: Path to save the plot
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
