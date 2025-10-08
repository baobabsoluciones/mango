from typing import Union, Optional

import numpy as np
import pandas as pd

from mango_shap.logging import get_configured_logger


class DataProcessor:
    """
    Utility class for processing data for SHAP analysis.

    Handles data validation, conversion, and preprocessing to ensure
    compatibility with SHAP explainers. Supports both pandas DataFrames
    and numpy arrays with automatic handling of missing values and
    categorical encoding.

    Example:
        >>> processor = DataProcessor()
        >>> processed_data = processor.process_data(raw_data, handle_missing='fill')
        >>> feature_names = DataProcessor.get_feature_names(processed_data)
    """

    def __init__(self) -> None:
        """
        Initialize the data processor.

        Sets up logging and prepares the processor for data preprocessing
        operations. No parameters are required as the processor is stateless.

        :return: None
        :rtype: None
        """
        self.logger = get_configured_logger()
        self.logger.info("DataProcessor initialized")

    def process_data(
        self, data: Union[np.ndarray, pd.DataFrame], handle_missing: str = "drop"
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Process data for SHAP analysis with missing value handling and encoding.

        Preprocesses input data to ensure compatibility with SHAP explainers.
        Handles missing values according to the specified strategy and converts
        categorical variables to numeric format.

        :param data: Input data to process (DataFrame or numpy array)
        :type data: Union[np.ndarray, pd.DataFrame]
        :param handle_missing: Strategy for handling missing values
        :type handle_missing: str
        :return: Processed data ready for SHAP analysis
        :rtype: Union[np.ndarray, pd.DataFrame]

        Example:
            >>> processor = DataProcessor()
            >>> processed_data = processor.process_data(raw_data, handle_missing='fill')
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
        Process pandas DataFrame for SHAP analysis.

        Handles missing values and encodes categorical variables in pandas
        DataFrames. Creates a copy of the input data to avoid modifying
        the original dataset.

        :param data: Input pandas DataFrame to process
        :type data: pd.DataFrame
        :param handle_missing: Strategy for handling missing values
        :type handle_missing: str
        :return: Processed DataFrame with numeric data types
        :rtype: pd.DataFrame
        """
        processed_data = data.copy()

        # Convert categorical variables to numeric first
        processed_data = self._encode_categorical(processed_data)

        # Handle missing values after encoding
        if processed_data.isnull().any().any():
            if handle_missing == "drop":
                processed_data = processed_data.dropna()
                self.logger.info("Dropped rows with missing values")
            elif handle_missing == "fill":
                # Fill numeric columns with mean, categorical with mode
                for col in processed_data.columns:
                    if processed_data[col].dtype in ["int64", "float64"]:
                        processed_data[col] = processed_data[col].fillna(
                            processed_data[col].mean()
                        )
                    else:
                        processed_data[col] = processed_data[col].fillna(
                            processed_data[col].mode()[0]
                            if not processed_data[col].mode().empty
                            else 0
                        )
                self.logger.info("Filled missing values with mean/mode")
            elif handle_missing == "error":
                raise ValueError("Data contains missing values")

        return processed_data

    def _process_array(self, data: np.ndarray, handle_missing: str) -> np.ndarray:
        """
        Process numpy array for SHAP analysis.

        Handles missing values in numpy arrays by filling them with mean values
        or raising an error based on the specified strategy. Creates a copy
        of the input array to avoid modifying the original data.

        :param data: Input numpy array to process
        :type data: np.ndarray
        :param handle_missing: Strategy for handling missing values
        :type handle_missing: str
        :return: Processed array with no missing values
        :rtype: np.ndarray
        """
        processed_data = data.copy()

        # Handle missing values
        if np.isnan(processed_data).any():
            if handle_missing == "drop":
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
        Encode categorical variables to numeric format.

        Converts object and category data types to numeric codes for
        compatibility with SHAP explainers. Handles both pandas categorical
        columns and object columns by converting them to numeric codes.

        :param data: Input DataFrame with potential categorical columns
        :type data: pd.DataFrame
        :return: DataFrame with all categorical variables encoded as numeric
        :rtype: pd.DataFrame
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

    @staticmethod
    def get_feature_names(
        data: Union[np.ndarray, pd.DataFrame], custom_names: Optional[list] = None
    ) -> list:
        """
        Get feature names for the input data.

        Extracts feature names from the data structure or uses custom names
        if provided. For DataFrames, uses column names; for arrays, generates
        generic feature names based on the number of features.

        :param data: Input data (DataFrame or numpy array)
        :type data: Union[np.ndarray, pd.DataFrame]
        :param custom_names: Optional custom feature names to use
        :type custom_names: Optional[list]
        :return: List of feature names
        :rtype: list

        Example:
            >>> feature_names = DataProcessor.get_feature_names(data)
            >>> custom_names = DataProcessor.get_feature_names(data, ['feat1', 'feat2'])
        """
        if custom_names is not None:
            return custom_names

        if isinstance(data, pd.DataFrame):
            return list(data.columns)
        else:
            return [f"feature_{i}" for i in range(data.shape[1])]
