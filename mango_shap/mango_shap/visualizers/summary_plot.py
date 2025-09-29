"""Summary plot visualization for SHAP values."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import shap
from ..logging.logger import get_logger


class SummaryPlot:
    """
    Create summary plots for SHAP values.

    Shows feature importance and impact on model output.
    """

    def __init__(self) -> None:
        """Initialize the summary plot generator."""
        self.logger = get_logger(__name__)

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

        :param shap_values: SHAP values to plot
        :param data: Data used to generate SHAP values
        :param feature_names: Names of features
        :param max_display: Maximum number of features to display
        :param show: Whether to display the plot
        :param save_path: Path to save the plot
        """
        self.logger.info("Creating summary plot")

        # Handle multi-class case
        if len(shap_values.shape) > 2:
            shap_values = shap_values[0]  # Use first class for multi-class

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
