from typing import Union

import numpy as np
import pandas as pd
import polars as pl


def time_series_to_sequence(
    data: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    context_window: int,
) -> np.ndarray:
    """
    Convert a time series to a sequence of context_window length to be used with recurrent neural networks.

    If we have a univariate time series of shape (n_samples, 1) and we pass a context_window of 10,
    we will get a sequence of shape (n_samples - 10, 10, 1).

    If we have a multivariate time series of shape (n_samples, 10) and we pass a context_window of 10,
    we will get a sequence of shape (n_samples - 10, 10, 10).

    :param data: time series data
    :type data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :param context_window: length of the sequence
    :type context_window: int
    :return: reshaped data
    :rtype: np.ndarray
    """
    # Convert input to numpy array based on type
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif isinstance(data, pl.DataFrame):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        raise ValueError(
            "Input data must be a numpy array, pandas DataFrame, or polars DataFrame"
        )

    if context_window <= 0:
        raise ValueError("context_window must be greater than 0")

    if len(data) <= context_window:
        raise ValueError("Data length must be greater than context_window")

    sequences = np.array(
        [data[t - context_window : t] for t in range(context_window, len(data) + 1, 1)]
    )
    return sequences


def time_series_to_sequence_v2(
    train_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    val_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    test_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    context_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts train, validation, and test time series data into sequences of fixed length for use with RNN-based models.

    This function slices each dataset into overlapping sequences of length context_window, and ensures continuity
    between splits by prepending the last context_window - 1 rows of the previous split to the next one.

    For example, the last context window from train_data is prepended to val_data to form a continuous sequence,
    and similarly for val_data and test_data.

    :param train_data: Training portion of the time series data
    :type train_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :param val_data: Validation portion of the time series data
    :type val_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :param test_data: Test portion of the time series data
    :type test_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :param context_window: Length of each time window (sequence)
    :type context_window: int
    :return: Arrays of shape (n_sequences, context_window, n_features) for train, validation, and test
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
    :raises ValueError: If inputs are not of valid types, or if context_window exceeds any dataset length
    """
    # Convert input to numpy array based on type
    if isinstance(train_data, pd.DataFrame):
        train_data = train_data.values
    elif isinstance(train_data, pl.DataFrame):
        train_data = train_data.to_numpy()
    elif not isinstance(train_data, np.ndarray):
        raise ValueError(
            "Input data must be a numpy array, pandas DataFrame, or polars DataFrame"
        )
    if isinstance(val_data, pd.DataFrame):
        val_data = val_data.values
    elif isinstance(val_data, pl.DataFrame):
        val_data = val_data.to_numpy()
    elif not isinstance(val_data, np.ndarray):
        raise ValueError(
            "Input data must be a numpy array, pandas DataFrame, or polars DataFrame"
        )
    if isinstance(test_data, pd.DataFrame):
        test_data = test_data.values
    elif isinstance(test_data, pl.DataFrame):
        test_data = test_data.to_numpy()
    elif not isinstance(test_data, np.ndarray):
        raise ValueError(
            "Input data must be a numpy array, pandas DataFrame, or polars DataFrame"
        )

    if context_window <= 0:
        raise ValueError("context_window must be greater than 0")

    if len(train_data) <= context_window:
        raise ValueError("Data length must be greater than context_window")
    if len(val_data) <= context_window:
        raise ValueError("Data length must be greater than context_window")
    if len(test_data) <= context_window:
        raise ValueError("Data length must be greater than context_window")

    # Define sequences of context window length for train data
    sequences_train = np.array(
        [
            train_data[t - context_window : t]
            for t in range(context_window, len(train_data) + 1, 1)
        ]
    )

    # Define sequences of context window length for validation data
    # In this case last context window from train data is prepended to val data
    val_data_train = train_data[
        len(train_data) - context_window + 1 : len(train_data) + 1
    ]
    val_data_seq = np.concatenate([val_data_train, val_data])
    sequences_val = np.array(
        [
            val_data_seq[t - context_window : t]
            for t in range(context_window, len(val_data_seq) + 1, 1)
        ]
    )

    # Define sequences of context window length for test data
    # In this case last context window from val data is prepended to test data
    test_data_val = val_data[len(val_data) - context_window + 1 : len(val_data) + 1]
    test_data_seq = np.concatenate([test_data_val, test_data])
    sequences_test = np.array(
        [
            test_data_seq[t - context_window : t]
            for t in range(context_window, len(test_data_seq) + 1, 1)
        ]
    )

    return sequences_train, sequences_val, sequences_test
