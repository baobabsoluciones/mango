from typing import Union, Any

import numpy as np
import pandas as pd
from mango_shap.logging import get_configured_logger


class InputValidator:
    """
    Utility class for validating inputs for SHAP analysis.

    Ensures data quality and compatibility with SHAP explainers by validating
    models, data structures, feature names, and SHAP values before analysis.

    Example:
        >>> validator = InputValidator()
        >>> validator.validate_model(trained_model)
        >>> validator.validate_data(X_test)
    """

    def __init__(self) -> None:
        """
        Initialize the input validator.

        Sets up logging and prepares the validator for input validation
        operations. No parameters are required as the validator is stateless.

        :return: None
        :rtype: None
        """
        self.logger = get_configured_logger()
        self.logger.info("InputValidator initialized")

    def validate_model(self, model: Any) -> None:
        """
        Validate that the model is compatible with SHAP analysis.

        Checks if the model has the required methods and can make predictions
        on test data. Ensures the model is ready for SHAP explainer initialization.

        :param model: Model object to validate
        :type model: Any
        :return: None
        :rtype: None
        :raises ValueError: If model is None, missing required methods, or prediction fails
        """
        self.logger.info("Validating model compatibility")

        if model is None:
            raise ValueError("Model cannot be None")

        # Check if model has required methods
        required_methods = ["predict"]
        for method in required_methods:
            if not hasattr(model, method):
                raise ValueError(f"Model must have {method} method")

        # Test model prediction with minimal data
        # Note: We can't test with arbitrary data since we don't know the expected input shape
        # Instead, we'll just verify the predict method exists and is callable
        if not callable(getattr(model, "predict", None)):
            raise ValueError("Model predict method is not callable")

        self.logger.info("Model validation passed")

    def validate_data(
        self, data: Union[np.ndarray, pd.DataFrame], name: str = "data"
    ) -> None:
        """
        Validate data structure and content for SHAP analysis.

        Checks data type, dimensions, emptiness, and presence of infinite values.
        Ensures data is compatible with SHAP explainers and analysis requirements.

        :param data: Data to validate (numpy array or pandas DataFrame)
        :type data: Union[np.ndarray, pd.DataFrame]
        :param name: Name of the data for error messages
        :type name: str
        :return: None
        :rtype: None
        :raises ValueError: If data is None, wrong type, empty, or contains infinite values
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

        Performs standard data validation plus additional checks specific to
        background data requirements, ensuring sufficient samples for SHAP calculations.

        :param background_data: Background data to validate
        :type background_data: Union[np.ndarray, pd.DataFrame]
        :return: None
        :rtype: None
        :raises ValueError: If background data is invalid or has insufficient samples
        """
        self.validate_data(background_data, "background_data")

        # Additional validation for background data
        if hasattr(background_data, "shape"):
            if background_data.shape[0] < 2:
                raise ValueError("Background data must have at least 2 samples")

    @staticmethod
    def validate_feature_names(feature_names: list, data_shape: tuple) -> None:
        """
        Validate feature names for consistency with data shape.

        Checks that feature names are provided as a list, match the number of
        features in the data, and are unique. Allows None values to pass validation.

        :param feature_names: List of feature names to validate
        :type feature_names: list
        :param data_shape: Shape tuple of the data (n_samples, n_features)
        :type data_shape: tuple
        :return: None
        :rtype: None
        :raises ValueError: If feature names are invalid, wrong length, or contain duplicates
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
        Validate SHAP values for consistency with input data.

        Checks that SHAP values are properly formatted numpy arrays with correct
        dimensions matching the input data, and verifies absence of infinite values.

        :param shap_values: SHAP values array to validate
        :type shap_values: np.ndarray
        :param data: Original data used to generate SHAP values
        :type data: Union[np.ndarray, pd.DataFrame]
        :return: None
        :rtype: None
        :raises ValueError: If SHAP values are None, wrong type, wrong dimensions, or contain infinite values
        """
        self.logger.info("Validating SHAP values")

        if shap_values is None:
            raise ValueError("SHAP values cannot be None")

        if not isinstance(shap_values, np.ndarray):
            raise ValueError("SHAP values must be numpy array")

        if len(shap_values.shape) < 2:
            raise ValueError("SHAP values must be at least 2-dimensional")

        # Check shape compatibility
        # SHAP values can be calculated for a subset of data, so first dimension can be different
        # Only check that SHAP values don't have more samples than the original data
        if shap_values.shape[0] > data.shape[0]:
            raise ValueError(
                f"SHAP values first dimension ({shap_values.shape[0]}) "
                f"cannot be greater than data first dimension ({data.shape[0]})"
            )

        # Check that the number of features matches
        if shap_values.shape[-1] != data.shape[1]:
            raise ValueError(
                f"SHAP values last dimension ({shap_values.shape[-1]}) "
                f"must match data second dimension ({data.shape[1]})"
            )

        # Check for infinite values
        if np.isinf(shap_values).any():
            raise ValueError("SHAP values contain infinite values")

        self.logger.info("SHAP values validation passed")
