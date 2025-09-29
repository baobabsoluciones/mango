"""Input validation utilities for SHAP analysis."""

from typing import Union, Any

import numpy as np
import pandas as pd
from mango_shap.logging import get_configured_logger


class InputValidator:
    """
    Utility class for validating inputs for SHAP analysis.

    Ensures data quality and compatibility with SHAP explainers.
    """

    def __init__(self) -> None:
        """Initialize the input validator."""
        self.logger = get_configured_logger()
        self.logger.info("InputValidator initialized")

    def validate_model(self, model: Any) -> None:
        """
        Validate that the model is compatible with SHAP.

        :param model: Model to validate
        :raises ValueError: If model is not compatible
        """
        self.logger.info("Validating model compatibility")

        if model is None:
            raise ValueError("Model cannot be None")

        # Check if model has required methods
        required_methods = ["predict"]
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model must have {method} method")

        # Test model prediction
        try:
            test_data = np.random.random((1, 10))
            model.predict(test_data)
        except Exception as e:
            raise ValueError(f"Model prediction failed: {str(e)}")

        self.logger.info("Model validation passed")

    def validate_data(
        self, data: Union[np.ndarray, pd.DataFrame], name: str = "data"
    ) -> None:
        """
        Validate data for SHAP analysis.

        :param data: Data to validate
        :param name: Name of the data for error messages
        :raises ValueError: If data is invalid
        """
        self.logger.info(f"Validating {name}")

        if data is None:
            raise ValueError(f"{name} cannot be None")

        if isinstance(data, (list, tuple)):
            data = np.array(data)

        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            raise ValueError(f"{name} must be numpy array or pandas DataFrame")

        if hasattr(data, "shape"):
            if len(data.shape) != 2:
                raise ValueError(f"{name} must be 2-dimensional")

            if data.shape[0] == 0:
                raise ValueError(f"{name} cannot be empty")

            if data.shape[1] == 0:
                raise ValueError(f"{name} must have at least one feature")

        # Check for infinite values
        if isinstance(data, np.ndarray):
            if np.isinf(data).any():
                raise ValueError(f"{name} contains infinite values")
        elif isinstance(data, pd.DataFrame):
            if np.isinf(data.select_dtypes(include=[np.number])).any().any():
                raise ValueError(f"{name} contains infinite values")

        self.logger.info(f"{name} validation passed")

    def validate_background_data(
        self, background_data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Validate background data for SHAP analysis.

        :param background_data: Background data to validate
        :raises ValueError: If background data is invalid
        """
        self.validate_data(background_data, "background_data")

        # Additional validation for background data
        if hasattr(background_data, "shape"):
            if background_data.shape[0] < 2:
                raise ValueError("Background data must have at least 2 samples")

    @staticmethod
    def validate_feature_names(feature_names: list, data_shape: tuple) -> None:
        """
        Validate feature names.

        :param feature_names: Feature names to validate
        :param data_shape: Shape of the data
        :raises ValueError: If feature names are invalid
        """
        if feature_names is None:
            return

        if not isinstance(feature_names, list):
            raise ValueError("Feature names must be a list")

        if len(feature_names) != data_shape[1]:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) "
                f"must match number of features ({data_shape[1]})"
            )

        # Check for duplicate names
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Feature names must be unique")

    def validate_shap_values(
        self, shap_values: np.ndarray, data: Union[np.ndarray, pd.DataFrame]
    ) -> None:
        """
        Validate SHAP values.

        :param shap_values: SHAP values to validate
        :param data: Data used to generate SHAP values
        :raises ValueError: If SHAP values are invalid
        """
        self.logger.info("Validating SHAP values")

        if shap_values is None:
            raise ValueError("SHAP values cannot be None")

        if not isinstance(shap_values, np.ndarray):
            raise ValueError("SHAP values must be numpy array")

        if len(shap_values.shape) < 2:
            raise ValueError("SHAP values must be at least 2-dimensional")

        # Check shape compatibility
        if shap_values.shape[0] != data.shape[0]:
            raise ValueError(
                f"SHAP values first dimension ({shap_values.shape[0]}) "
                f"must match data first dimension ({data.shape[0]})"
            )

        if shap_values.shape[-1] != data.shape[1]:
            raise ValueError(
                f"SHAP values last dimension ({shap_values.shape[-1]}) "
                f"must match data second dimension ({data.shape[1]})"
            )

        # Check for infinite values
        if np.isinf(shap_values).any():
            raise ValueError("SHAP values contain infinite values")

        self.logger.info("SHAP values validation passed")
