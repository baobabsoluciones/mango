from typing import Optional, Tuple

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
