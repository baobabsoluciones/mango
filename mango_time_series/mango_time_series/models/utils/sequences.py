from typing import Union

import numpy as np
import pandas as pd
import polars as pl


def time_series_to_sequence(
    data: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    context_window: int,
    id_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame] = None,
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
    :param id_data: id data
    :type id_data: Union[np.ndarray, pd.DataFrame, pl.DataFrame]
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

    if id_data is not None:
        if isinstance(id_data, pd.DataFrame):
            id_data = id_data.values
        elif isinstance(id_data, pl.DataFrame):
            id_data = id_data.to_numpy()
        elif not isinstance(id_data, np.ndarray):
            raise ValueError(
                "id_data must be a numpy array, pandas DataFrame, or polars DataFrame"
            )

        # extract the unique ids from id_data
        unique_ids = np.unique(id_data)
        sequences = []
        for unique_id in unique_ids:
            # get the index of the unique id
            id_index = np.where(id_data == unique_id)[0]
            data_i = data[id_index]
            sequences_i = np.array(
                [
                    data_i[t - context_window : t]
                    for t in range(context_window, len(data_i) + 1, 1)
                ]
            )
            sequences.extend(sequences_i)
        sequences = np.array(sequences)

    else:
        sequences = np.array(
            [
                data[t - context_window : t]
                for t in range(context_window, len(data) + 1, 1)
            ]
        )
    return sequences
