from typing import Any, List, Optional, Tuple, Union

import numpy as np


def time_series_split(
    data: np.ndarray, train_size: float, val_size: float, test_size: float
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Splits data into training, validation, and test sets according to the specified percentages.

    :param data: Array-like data to split
    :type data: np.ndarray
    :param train_size: Proportion of the dataset to include in the training set (0-1)
    :type train_size: float
    :param val_size: Proportion of the dataset to include in the validation set (0-1)
    :type val_size: float
    :param test_size: Proportion of the dataset to include in the test set (0-1)
    :type test_size: float
    :return: The training, validation, and test sets as numpy arrays
    :rtype: Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
    :raises ValueError: If train_size, val_size, or test_size are None or their sum is not 1.0
    """
    if train_size is None or val_size is None or test_size is None:
        raise ValueError(
            "train_size, val_size, and test_size must be specified and not None."
        )

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            "The sum of train_size, val_size, and test_size must be 1.0, "
            f"but got {train_size + val_size + test_size}."
        )

    # Original implementation for sequential split
    n = len(data)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train_set = data[:train_end]
    val_set = data[train_end:val_end] if val_size > 0 else None
    test_set = data[val_end:] if test_size > 0 else None

    return train_set, val_set, test_set


def convert_data_to_numpy(
    data: Any,
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], List[str]]:
    """
    Convert data to numpy array format.

    Handles pandas and polars DataFrames, converting them to numpy arrays.
    If data is a tuple, converts each element in the tuple.

    :param data: Input data that can be pandas DataFrame, polars DataFrame,
        numpy array, or tuple of these types
    :type data: Any
    :return: Data converted to numpy array(s) and feature names if available
    :rtype: Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], List[str]]
    :raises ValueError: If data type is not supported
    """
    try:
        import pandas as pd

        has_pandas = True
    except ImportError:
        has_pandas = False

    try:
        import polars as pl

        has_polars = True
    except ImportError:
        has_polars = False

    if data is None:
        return data, []

    feature_names = []

    if has_pandas and hasattr(data, "columns"):
        feature_names = data.columns.tolist()
    elif has_polars and hasattr(data, "columns"):
        feature_names = data.columns
    elif isinstance(data, tuple) and len(data) > 0:
        # For tuple, try to get column names from first element
        first_item = data[0]
        if has_pandas and hasattr(first_item, "columns"):
            feature_names = first_item.columns.tolist()
        elif has_polars and hasattr(first_item, "columns"):
            feature_names = first_item.columns

    if isinstance(data, tuple):
        converted_data = tuple(
            convert_single_data_to_numpy(item, has_pandas, has_polars) for item in data
        )
        return converted_data, feature_names
    else:
        converted_data = convert_single_data_to_numpy(data, has_pandas, has_polars)
        return converted_data, feature_names


def convert_single_data_to_numpy(
    data_item: Any, has_pandas: bool, has_polars: bool
) -> np.ndarray:
    """
    Convert a single data item to numpy array.

    :param data_item: Single data item to convert
    :type data_item: Any
    :param has_pandas: Whether pandas is available
    :type has_pandas: bool
    :param has_polars: Whether polars is available
    :type has_polars: bool
    :return: Data converted to numpy array
    :rtype: np.ndarray
    :raises ValueError: If data type is not supported
    """
    if has_pandas and hasattr(data_item, "to_numpy"):
        return data_item.to_numpy()
    elif has_polars and hasattr(data_item, "to_numpy"):
        return data_item.to_numpy()
    elif isinstance(data_item, np.ndarray):
        return data_item
    else:
        raise ValueError(
            f"Unsupported data type: {type(data_item)}. "
            f"Data must be a numpy array, pandas DataFrame, or polars DataFrame."
        )
