from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


def _to_numpy(data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]) -> np.ndarray:
    """
    Convert input data to numpy array format.

    Internal helper function that converts various data formats (numpy arrays,
    pandas DataFrames, or polars DataFrames) to numpy arrays for consistent
    processing in sequence generation functions.

    :param data: Input data to convert to numpy array
    :type data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :return: Data converted to numpy array
    :rtype: np.ndarray
    :raises ValueError: If input is not a valid supported type

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> array = _to_numpy(df)
        >>> print(f"Array shape: {array.shape}")
        Array shape: (3, 2)

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
    Create sequences from data using sliding window approach.

    Internal helper function that generates overlapping sequences from time series
    data using a sliding window of specified length. Each sequence contains
    consecutive time steps for RNN-based model training.

    :param data: Input time series data (samples x features)
    :type data: np.ndarray
    :param context_window: Length of each time window/sequence
    :type context_window: int
    :return: Array of sequences with shape (n_sequences, context_window, n_features)
    :rtype: np.ndarray

    Example:
        >>> import numpy as np
        >>> data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        >>> sequences = _create_sequences(data, 3)
        >>> print(f"Sequences shape: {sequences.shape}")
        Sequences shape: (3, 3, 2)

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
    Convert time series data into sequences of fixed length for RNN-based models.

    Transforms time series data into overlapping sequences suitable for training
    recurrent neural networks. Handles both single dataset and multiple dataset
    scenarios with proper temporal continuity between train/validation/test splits.

    This function can handle two main cases:
    1. Single dataset: Converts a single time series into sequences
    2. Multiple datasets: Converts train, validation, and test datasets into sequences,
    ensuring continuity between splits by prepending the last context_window - 1 rows
    of the previous split to the next one.

    :param data: Time series data (training data in case of multiple datasets)
    :type data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
    :param context_window: Length of each time window/sequence
    :type context_window: int
    :param val_data: Validation dataset (optional, required for multiple datasets)
    :type val_data: Optional[Union[np.ndarray, pd.DataFrame, pl.DataFrame]]
    :param test_data: Test dataset (optional, required for multiple datasets)
    :type test_data: Optional[Union[np.ndarray, pd.DataFrame, pl.DataFrame]]
    :return: Either:
        - Single array of shape (n_sequences, context_window, n_features) for single dataset
        - Tuple of three arrays (train_sequences, val_sequences, test_sequences) for multiple datasets
    :rtype: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    :raises ValueError: If inputs are not of valid types, context_window is invalid, or dataset lengths are insufficient

    Example:
        >>> import numpy as np
        >>> # Single dataset case
        >>> data = np.random.randn(100, 5)  # 100 time steps, 5 features
        >>> sequences = time_series_to_sequence(data, 10)
        >>> print(f"Single dataset sequences shape: {sequences.shape}")
        Single dataset sequences shape: (91, 10, 5)

        >>> # Multiple datasets case
        >>> train = np.random.randn(70, 5)
        >>> val = np.random.randn(20, 5)
        >>> test = np.random.randn(10, 5)
        >>> train_seq, val_seq, test_seq = time_series_to_sequence(train, 10, val, test)
        >>> print(f"Train: {train_seq.shape}, Val: {val_seq.shape}, Test: {test_seq.shape}")
        Train: (61, 10, 5), Val: (20, 10, 5), Test: (10, 10, 5)

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
