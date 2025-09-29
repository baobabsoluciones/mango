"""Force plot visualization for SHAP values."""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import shap
from mango_shap.logging import get_configured_logger


class ForcePlot:
    """
    Create force plots for SHAP values.

    Shows the force of each feature on the prediction.
    """

    def __init__(self) -> None:
        """Initialize the force plot generator."""
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

        :param shap_values: SHAP values to plot
        :param data: Data used to generate SHAP values
        :param instance_idx: Index of the instance to plot
        :param feature_names: Names of features
        :param show: Whether to display the plot
        :param save_path: Path to save the plot
        """
        self.logger.info(f"Creating force plot for instance {instance_idx}")

        # Handle multi-class case
        if len(shap_values.shape) > 2:
            shap_values = shap_values[0]

        # Get SHAP values and data for the specific instance
        instance_shap = shap_values[instance_idx]
        instance_data = (
            data[instance_idx]
            if hasattr(data, "__getitem__")
            else data.iloc[instance_idx]
        )

        # Create the plot
        force_plot = shap.force_plot(
            shap_values=instance_shap,
            features=instance_data,
            feature_names=feature_names,
            matplotlib=True,
        )

        if save_path:
            force_plot.savefig(save_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Force plot saved to {save_path}")

        if show:
            force_plot.show()
        else:
            force_plot.close()
