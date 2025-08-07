from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


def _to_numpy(data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]) -> np.ndarray:
    """
    Convert input data to numpy array.

    :param data: Input data
    :type data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :return: Numpy array
    :rtype: np.ndarray
    :raises ValueError: If input is not a valid type
    """
    if isinstance(data, pd.DataFrame):
        return data.values
    if isinstance(data, pl.DataFrame):
        return data.to_numpy()
    if not isinstance(data, np.ndarray):
        raise ValueError(
            "Input data must be a numpy array, pandas DataFrame, or polars DataFrame"
        )
    return data


def _create_sequences(data: np.ndarray, context_window: int) -> np.ndarray:
    """
    Create sequences from data using sliding window.

    :param data: Input data
    :type data: np.ndarray
    :param context_window: Length of each time window
    :type context_window: int
    :return: Array of sequences
    :rtype: np.ndarray
    """
    return np.array(
        [data[t - context_window : t] for t in range(context_window, len(data) + 1, 1)]
    )


def time_series_to_sequence(
    data: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    context_window: int,
    val_data: Optional[Union[np.ndarray, pd.DataFrame, pl.DataFrame]] = None,
    test_data: Optional[Union[np.ndarray, pd.DataFrame, pl.DataFrame]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert time series data into sequences of fixed length for use with RNN-based models.

    This function can handle both single dataset and multiple dataset cases:
    1. Single dataset: Converts a single time series into sequences
    2. Multiple datasets: Converts train, validation, and test datasets into sequences,
       ensuring continuity between splits by prepending the last context_window - 1 rows
       of the previous split to the next one.

    :param data: Time series data (training data in case of multiple datasets)
    :type data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :param context_window: Length of each time window (sequence)
    :type context_window: int
    :param val_data: Validation dataset (optional)
    :type val_data: Optional[Union[np.ndarray, pd.DataFrame, pl.DataFrame]]
    :param test_data: Test dataset (optional)
    :type test_data: Optional[Union[np.ndarray, pd.DataFrame, pl.DataFrame]]
    :return: Either:
        - Single array of shape (n_sequences, context_window, n_features) for single dataset
        - Tuple of three arrays for train, validation, and test datasets
    :rtype: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    :raises ValueError: If inputs are not of valid types, or if context_window exceeds dataset length
    """
    if context_window <= 0:
        raise ValueError("context_window must be greater than 0")

    # Handle single dataset case
    if val_data is None and test_data is None:
        data = _to_numpy(data)
        if len(data) <= context_window:
            raise ValueError("Data length must be greater than context_window")
        return _create_sequences(data, context_window)

    # Handle multiple datasets case
    elif val_data is None or test_data is None:
        raise ValueError(
            "For multiple datasets, both val_data and test_data must be provided"
        )

    # Convert all inputs to numpy arrays
    data = _to_numpy(data)
    val_data = _to_numpy(val_data)
    test_data = _to_numpy(test_data)

    # Validate lengths
    for dataset, name in [
        (data, "train"),
        (val_data, "validation"),
        (test_data, "test"),
    ]:
        if len(dataset) <= context_window:
            raise ValueError(
                f"{name.capitalize()} data length must be greater than context_window"
            )

    # Create sequences for each dataset
    sequences_train = _create_sequences(data, context_window)

    # Create sequences for validation data with prepended train data
    val_data_seq = np.concatenate([data[-context_window + 1 :], val_data])
    sequences_val = _create_sequences(val_data_seq, context_window)

    # Create sequences for test data with prepended validation data
    test_data_seq = np.concatenate([val_data[-context_window + 1 :], test_data])
    sequences_test = _create_sequences(test_data_seq, context_window)

    return sequences_train, sequences_val, sequences_test
