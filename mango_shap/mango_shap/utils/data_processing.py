"""Data processing utilities for SHAP analysis."""

import numpy as np
import pandas as pd
from typing import Union, Optional
from ..logging.logger import get_logger


class DataProcessor:
    """
    Utility class for processing data for SHAP analysis.

    Handles data validation, conversion, and preprocessing.
    """

    def __init__(self) -> None:
        """Initialize the data processor."""
        self.logger = get_logger(__name__)

    def process_data(
        self, data: Union[np.ndarray, pd.DataFrame], handle_missing: str = "drop"
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Process data for SHAP analysis.

        :param data: Input data
        :param handle_missing: How to handle missing values ('drop', 'fill', 'error')
        :return: Processed data
        """
        self.logger.info("Processing data for SHAP analysis")

        if isinstance(data, pd.DataFrame):
            processed_data = self._process_dataframe(data, handle_missing)
        elif isinstance(data, np.ndarray):
            processed_data = self._process_array(data, handle_missing)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        self.logger.info(f"Data processed successfully. Shape: {processed_data.shape}")
        return processed_data

    def _process_dataframe(
        self, data: pd.DataFrame, handle_missing: str
    ) -> pd.DataFrame:
        """
        Process pandas DataFrame.

        :param data: Input DataFrame
        :param handle_missing: How to handle missing values
        :return: Processed DataFrame
        """
        processed_data = data.copy()

        # Handle missing values
        if processed_data.isnull().any().any():
            if handle_missing == "drop":
                processed_data = processed_data.dropna()
                self.logger.info("Dropped rows with missing values")
            elif handle_missing == "fill":
                processed_data = processed_data.fillna(processed_data.mean())
                self.logger.info("Filled missing values with mean")
            elif handle_missing == "error":
                raise ValueError("Data contains missing values")

        # Convert categorical variables to numeric
        processed_data = self._encode_categorical(processed_data)

        return processed_data

    def _process_array(self, data: np.ndarray, handle_missing: str) -> np.ndarray:
        """
        Process numpy array.

        :param data: Input array
        :param handle_missing: How to handle missing values
        :return: Processed array
        """
        processed_data = data.copy()

        # Handle missing values
        if np.isnan(processed_data).any():
            if handle_missing == "drop":
                # For arrays, we can't easily drop rows, so we'll fill
                processed_data = np.nan_to_num(
                    processed_data, nan=np.nanmean(processed_data)
                )
                self.logger.info("Filled missing values with mean")
            elif handle_missing == "fill":
                processed_data = np.nan_to_num(
                    processed_data, nan=np.nanmean(processed_data)
                )
                self.logger.info("Filled missing values with mean")
            elif handle_missing == "error":
                raise ValueError("Data contains missing values")

        return processed_data

    def _encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables to numeric.

        :param data: Input DataFrame
        :return: DataFrame with encoded categorical variables
        """
        processed_data = data.copy()

        # Get categorical columns
        categorical_columns = processed_data.select_dtypes(
            include=["object", "category"]
        ).columns

        if len(categorical_columns) > 0:
            self.logger.info(f"Encoding {len(categorical_columns)} categorical columns")

            for col in categorical_columns:
                if processed_data[col].dtype == "category":
                    processed_data[col] = processed_data[col].cat.codes
                else:
                    processed_data[col] = pd.Categorical(processed_data[col]).codes

        return processed_data

    def get_feature_names(
        self, data: Union[np.ndarray, pd.DataFrame], custom_names: Optional[list] = None
    ) -> list:
        """
        Get feature names for the data.

        :param data: Input data
        :param custom_names: Custom feature names
        :return: List of feature names
        """
        if custom_names is not None:
            return custom_names

        if isinstance(data, pd.DataFrame):
            return list(data.columns)
        else:
            return [f"feature_{i}" for i in range(data.shape[1])]
