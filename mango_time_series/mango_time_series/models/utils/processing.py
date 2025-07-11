from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from mango.logging import get_configured_logger

logger = get_configured_logger()


def abs_mean(
    data: Union[np.ndarray, pd.Series, pl.Series],
) -> Union[np.floating, float]:
    """
    Compute absolute mean of data.

    :param data: data item to convert
    :type data: Union[np.ndarray, pd.Series, pl.Series]
    :rtype: Union[np.floating, float]
    :raises ValueError: If data type is not supported
    """
    if isinstance(data, (pd.Series, pl.Series)):
        return data.abs().mean()
    elif isinstance(data, np.ndarray):
        return np.abs(data).mean()
    else:
        raise TypeError(
            f"data type is {type(data)} but it "
            "must be a np.ndarray, pd.Series, or pl.Series."
        )


def reintroduce_nans(self, df: pd.DataFrame, id: str) -> pd.DataFrame:
    """
    When preparing the datasets NaNs are removed. This function adds
    the NaNs back to the data based on self._nan_coordinates.

    :param df: Data to reintroduce NaNs back into
    :type df: pd.DataFrame
    :param id: Identifier of dataset ("global" is only one dataset)
    :type id: str
    :return: Data with reintroduced NaNs
    :rtype: pd.DataFrame
    :raises ValueError: If "id" not in self._nan_coordinates
    """
    if id not in self._nan_coordinates:
        raise ValueError(f"{id} not found in _nan_coordinates.")

    df_nans = df.copy()

    # Need to shift one column to right if "data_split" in columns
    col_offset = 1 if df.columns[0] == "data_split" else 0

    for row, col in self._nan_coordinates[id]:
        adj_row = row - self._time_step_to_check[0]
        adj_col = col + col_offset

        # Reintroduce NaNs for valid indices in df
        if 0 <= adj_row < len(df):
            df_nans.iloc[adj_row, adj_col] = np.nan

    return df_nans


def id_pivot(df: pd.DataFrame, id: str) -> pd.DataFrame:
    """
    Select subset of data based on id and then pivot on that data
    in order to produced desired format with time_step as rows
    and features as columns.

    :param df: Data
    :type df: pd.DataFrame
    :param id: Identifier of dataset ("global" is only one dataset)
    :type id: str
    :return: Data subset in desired format
    :rtype: pd.DataFrame
    :raises ValueError: If df does not have required columns
    """
    required_cols = {"id", "feature", "time_step", "value", "data_split"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Select id
    df_id = df[df.id == id].copy()
    if df_id.empty:
        raise ValueError(f"No data found for id = {id}")

    # Save feature order since pd.pivot sorts columns automatically
    first_ts = df_id["time_step"].iloc[0]
    df_id_feat = df_id[df_id.time_step == first_ts]
    feature_order = df_id_feat["feature"].tolist()
    feature_column_no_duplicates = df_id["feature"].drop_duplicates().tolist()
    if feature_order != feature_column_no_duplicates:
        raise ValueError(
            f"feature in first time_step ({feature_order}) "
            f"does not match feature column ({feature_column_no_duplicates})"
        )

    # Pivot to have time_step as rows, features as columns
    df_id_pivot = pd.pivot(
        df_id, columns="feature", index=["time_step", "data_split"], values="value"
    )

    # Add data_split as a column
    df_id_pivot = df_id_pivot.reset_index(level=["data_split"])

    # Reorder columns to match original order
    df_id_pivot = df_id_pivot[["data_split"] + feature_order]

    return df_id_pivot


def save_csv(
    data: pd.DataFrame,
    save_path: str,
    filename: str,
    save_index: bool = False,
    decimals: int = 4,
    compression: str = "infer",
    logger_msg: str = "standard",
) -> None:
    """
    Save a DataFrame as a csv based on parameters provided.

    :param data: Data to save
    :type data: pd.DataFrame
    :param save_path: Path to save data
    :type save_path: str
    :param filename: Name to use to save file
    :type filename: str
    :param save_index: Whether to save the index of the DataFrame
    :type save_index: bool
    :param decimals: Number of decimal places for floating point numbers
    :type decimals: int
    :param compression: Type of compression used by pandas.DataFrame.to_csv
    :type compression: str
    :param logger_msg: Logger message to use
    :type logger_msg: str
    :type return: None
    """
    if not (filename.endswith(".csv") or filename.endswith(".csv.zip")):
        raise ValueError(f"Filename ({filename}) ending needs to be .csv or .csv.zip ")

    try:
        float_format = f"%.{decimals}f"
        data = data.round(decimals)

        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / filename
        data.to_csv(
            file_path,
            index=save_index,
            float_format=float_format,
            compression=compression,
        )

        if logger_msg == "standard":
            logger.info(f"{filename} saved to {path}")
        else:
            logger.info(logger_msg)

    except Exception as e:
        logger.error(f"Error saving csv: {str(e)}")
        raise


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
    if data is None:
        return data, []

    elif isinstance(data, tuple):
        arrays = []
        feature_names_list = []
        for data_i in data:
            array_i, feature_names_i = convert_data_to_numpy(data=data_i)
            arrays.append(array_i)
            feature_names_list.append(feature_names_i)

        if (
            feature_names_list[0] != feature_names_list[1]
            or feature_names_list[1] != feature_names_list[2]
        ):
            raise ValueError(f"Tuple has different feature names: {feature_names_list}")

        feature_names = feature_names_list[0]
        return tuple(arrays), feature_names

    else:
        array = _to_numpy(data=data)
        feature_names = _extract_feature_names(data=data)
        return array, feature_names


def _to_numpy(data: Any) -> np.ndarray:
    """
    Convert a single data item to numpy array.

    :param data: Single data item to convert
    :type data: np.ndarray or has attribute .to_numpy()
    :return: Data converted to numpy array
    :rtype: np.ndarray
    :raises ValueError: If data type is not supported
    """
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, "to_numpy"):
        return data.to_numpy()
    else:
        raise TypeError(
            f"Unsupported data type: {type(data)}. "
            f"Data must be a numpy array, pandas DataFrame, or polars DataFrame."
        )


def _extract_feature_names(data: Any) -> List[str]:
    """
    Extract feature names from data by looking at columns.

    :param data: Single data item to convert
    :type data: Any
    :return: Data columns as a list
    :rtype: List[str]
    """
    columns = getattr(data, "columns", None)
    if columns is not None:
        return list(columns)
    else:
        return []


def denormalize_data(
    data: np.ndarray,
    normalization_method: Optional[str],
    min_x: Optional[np.ndarray] = None,
    max_x: Optional[np.ndarray] = None,
    mean_: Optional[np.ndarray] = None,
    std_: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Denormalize data.

    :param data: Normalized data to denormalize
    :type data: np.ndarray
    :param normalization_method: Method used for normalization
    :type normalization_method: Optional[str]
    :param min_x: Minimum values for minmax normalization
    :type min_x: Optional[np.ndarray]
    :param max_x: Maximum values for minmax normalization
    :type max_x: Optional[np.ndarray]
    :param mean_: Mean values for zscore normalization
    :type mean_: Optional[np.ndarray]
    :param std_: Standard deviation values for zscore normalization
    :type std_: Optional[np.ndarray]
    :return: Denormalized data
    :rtype: np.ndarray
    :raises ValueError: If normalization method is invalid or parameters are missing
    """
    if normalization_method not in ["minmax", "zscore", None]:
        raise ValueError(
            f"Invalid normalization method: {normalization_method}. Must be 'minmax', 'zscore', or None."
        )

    if normalization_method == "minmax":
        if isinstance(min_x, np.ndarray) and isinstance(max_x, np.ndarray):
            return data * (max_x - min_x) + min_x
        else:
            raise TypeError("min_x and max_x need to be np.ndarrays.")

    elif normalization_method == "zscore":
        if isinstance(std_, np.ndarray) and isinstance(mean_, np.ndarray):
            return data * std_ + mean_
        else:
            raise TypeError("std_ and mean_ need to be np.ndarrays.")

    elif normalization_method is None:
        return data


def normalize_data_for_training(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    normalization_method: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Normalize training, validation and test data using the specified method.

    :param x_train: Training data to normalize
    :type x_train: np.ndarray
    :param x_val: Validation data to normalize
    :type x_val: np.ndarray
    :param x_test: Test data to normalize
    :type x_test: np.ndarray
    :param normalization_method: Method to use for normalization
    :type normalization_method: Optional[str]
    :return: Normalized training, validation and test data, and normalization values
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]
    :raises ValueError: If normalization method is invalid
    """
    if normalization_method not in ["minmax", "zscore", None]:
        raise ValueError(
            f"Invalid normalization method: {normalization_method}. Must be 'minmax', 'zscore', or None."
        )

    if normalization_method == "minmax":
        min_x = np.nanmin(x_train, axis=0)
        max_x = np.nanmax(x_train, axis=0)
        range_x = max_x - min_x

        # Safe divisor for constant columns
        safe_range_x = np.where(range_x == 0, 1.0, range_x)

        x_train = (x_train - min_x) / safe_range_x
        x_val = (x_val - min_x) / safe_range_x
        x_test = (x_test - min_x) / safe_range_x
        normalization_values = {"min_x": min_x, "max_x": max_x}

    elif normalization_method == "zscore":
        mean_ = np.nanmean(x_train, axis=0)
        std_ = np.nanstd(x_train, axis=0)

        # Safe divisor for constant columns
        safe_std = np.where(std_ == 0, 1.0, std_)

        x_train = (x_train - mean_) / safe_std
        x_val = (x_val - mean_) / safe_std
        x_test = (x_test - mean_) / safe_std
        normalization_values = {"mean_": mean_, "std_": std_}

    elif normalization_method is None:
        normalization_values = {}

    return x_train, x_val, x_test, normalization_values


def normalize_data(
    data: np.ndarray,
    normalization_method: Optional[str],
    normalization_values: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if normalization_method not in ["minmax", "zscore", None]:
        raise ValueError(
            f"Invalid normalization method: {normalization_method}. Must be 'minmax', 'zscore', or None."
        )

    if normalization_method == "minmax":
        if normalization_values == {}:
            x_min = np.nanmin(data, axis=0)
            x_max = np.nanmax(data, axis=0)
        else:
            x_min = normalization_values.get("x_min", None)
            x_max = normalization_values.get("x_max", None)
            if x_min is None or x_max is None:
                raise ValueError("normalization_values missing x_min and x_max.")

        x_range = x_max - x_min
        # Safe divisor for constant columns
        safe_x_range = np.where(x_range == 0, 1.0, x_range)
        data_normalized = (data - x_min) / safe_x_range
        new_normalization_values = {"x_min": x_min, "x_max": x_max}

    elif normalization_method == "zscore":
        if normalization_values == {}:
            x_mean = np.nanmean(data, axis=0)
            x_std = np.nanstd(data, axis=0)
        else:
            x_mean = normalization_values.get("x_mean", None)
            x_std = normalization_values.get("x_std", None)
            if x_mean is None or x_std is None:
                raise ValueError("normalization_values missing x_mean and x_std.")

        # Safe divisor for constant columns
        safe_x_std = np.where(x_std == 0, 1.0, x_std)
        data_normalized = (data - x_mean) / safe_x_std
        new_normalization_values = {"x_mean": x_mean, "x_std": x_std}

    elif normalization_method is None:
        data_normalized = data
        new_normalization_values = {}

    return data_normalized, new_normalization_values


def normalize_data_for_prediction(
    data: np.ndarray,
    normalization_method: Optional[str],
    min_x: Optional[np.ndarray] = None,
    max_x: Optional[np.ndarray] = None,
    mean_: Optional[np.ndarray] = None,
    std_: Optional[np.ndarray] = None,
    feature_to_check: Optional[Union[int, List[int]]] = None,
    feature_to_check_filter: bool = False,
) -> np.ndarray:
    """
    Normalize new data using stored normalization parameters.
    If parameters are not available, computes them from input data.

    :param data: New data to normalize
    :type data: np.ndarray
    :param normalization_method: Method to use for normalization
    :type normalization_method: str
    :param min_x: Minimum values for minmax normalization
    :type min_x: Optional[np.ndarray]
    :param max_x: Maximum values for minmax normalization
    :type max_x: Optional[np.ndarray]
    :param mean_: Mean values for zscore normalization
    :type mean_: Optional[np.ndarray]
    :param std_: Standard deviation values for zscore normalization
    :type std_: Optional[np.ndarray]
    :param feature_to_check: Feature indices to check
    :type feature_to_check: Optional[Union[int, List[int]]]
    :param feature_to_check_filter: Whether to filter features for checking
    :type feature_to_check_filter: bool
    :return: Normalized data
    :rtype: np.ndarray
    :raises ValueError: If normalization method is invalid
    """
    if normalization_method not in ["minmax", "zscore", None]:
        raise ValueError(
            f"Invalid normalization method: {normalization_method}. Must be 'minmax', 'zscore', or None."
        )

    if normalization_method == "minmax":
        if min_x is None or max_x is None:
            min_x = np.nanmin(data, axis=0)
            max_x = np.nanmax(data, axis=0)
            range_x = max_x - min_x

            # Safe divisor for constant columns
            safe_range_x = np.where(range_x == 0, 1.0, range_x)

            return (data - min_x) / safe_range_x
        else:
            if feature_to_check_filter and feature_to_check is not None:
                range_x = max_x[feature_to_check] - min_x[feature_to_check]
                return (data - min_x[feature_to_check]) / range_x
            else:
                range_x = max_x - min_x
                return (data - min_x) / range_x

    elif normalization_method == "zscore":
        if mean_ is None or std_ is None:
            mean_ = np.nanmean(data, axis=0)
            std_ = np.nanstd(data, axis=0)
            return (data - mean_) / std_
        else:
            if feature_to_check_filter and feature_to_check is not None:
                return (data - mean_[feature_to_check]) / std_[feature_to_check]
            else:
                return (data - mean_) / std_


def apply_padding(
    data: np.ndarray,
    reconstructed: np.ndarray,
    context_window: int,
    time_step_to_check: Union[int, List[int]],
) -> np.ndarray:
    """
    Apply padding dynamically based on time_step_to_check and context_window.

    This method handles the padding of reconstructed data to match the original data shape,
    taking into account the context window and the specific time step being predicted.

    :param data: Original dataset with shape (num_samples, num_features)
    :type data: np.ndarray
    :param reconstructed: Predicted values with shape (num_samples - context_window, num_features)
    :type reconstructed: np.ndarray
    :param context_window: Size of the context window used for prediction
    :type context_window: int
    :param time_step_to_check: Time step to predict within the window
    :type time_step_to_check: Union[int, List[int]]
    :return: Padded reconstructed dataset with shape matching the original data
    :rtype: np.ndarray
    :raises ValueError: If time_step_to_check is not within valid range
    """
    num_samples, num_features = data.shape
    padded_reconstructed = np.full((num_samples, num_features), np.nan)

    # Determine the offset based on time_step_to_check
    if isinstance(time_step_to_check, list):
        time_step_to_check = time_step_to_check[0]

    if time_step_to_check < 0 or time_step_to_check > context_window - 1:
        raise ValueError(
            f"time_step_to_check must be between 0 and {context_window - 1}, "
            f"but got {time_step_to_check}"
        )

    if time_step_to_check == 0:
        padded_reconstructed[: num_samples - (context_window - 1)] = reconstructed
    elif time_step_to_check == context_window - 1:
        padded_reconstructed[context_window - 1 :] = reconstructed
    else:
        before = time_step_to_check
        after = context_window - 1 - time_step_to_check
        padded_reconstructed[before : num_samples - after] = reconstructed

    return padded_reconstructed


def handle_id_columns(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    id_columns: Union[str, int, List[str], List[int], None],
    features_name: Optional[List[str]],
    context_window: Optional[int],
) -> Tuple[
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    Optional[np.ndarray],
    Dict[str, np.ndarray],
    List[int],
]:
    """
    Handle id_columns processing for data grouping.

    :param data: Data to process, can be single array or tuple of arrays
    :type data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
    :param id_columns: Column(s) to process the data by groups
    :type id_columns: Union[str, int, List[str], List[int], None]
    :param features_name: List of feature names
    :type features_name: Optional[List[str]]
    :param context_window: Context window for the model
    :type context_window: Optional[int]
    :return: Tuple containing:
        - Processed data (with ID columns removed)
        - ID mapping array
        - Dictionary with grouped data by ID
        - List of column indices
    :rtype: Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], Optional[np.ndarray], Dict[str, np.ndarray], List[int]]
    :raises ValueError: If id_columns format is invalid or if minimum samples per ID is less than context_window
    """
    if id_columns is None:
        return data, None, {}, []

    id_columns = [id_columns] if isinstance(id_columns, (str, int)) else id_columns

    if all(isinstance(i, str) for i in id_columns):
        id_column_indices = [
            i for i, value in enumerate(features_name) if value in id_columns
        ]
    elif all(isinstance(i, int) for i in id_columns):
        id_column_indices = id_columns
    else:
        raise ValueError("id_columns must be a list of strings or integers")

    if isinstance(data, tuple):
        id_data = tuple(
            np.array([f"__".join(map(str, row)) for row in d[:, id_column_indices]])
            for d in data
        )
        data = tuple(
            np.delete(d, id_column_indices, axis=1).astype(np.float64) for d in data
        )
    else:
        id_data = np.array(
            [f"__".join(map(str, row)) for row in data[:, id_column_indices]]
        )
        data = np.delete(data, id_column_indices, axis=1).astype(np.float64)

    if isinstance(id_data, tuple):
        unique_ids = np.unique(id_data[0])
        id_data_dict = {
            unique_id: (
                data[0][id_data[0] == unique_id],
                data[1][id_data[1] == unique_id],
                data[2][id_data[2] == unique_id],
            )
            for unique_id in unique_ids
        }
        min_samples_all_ids = min(
            [
                np.min(np.unique(id_data_item, return_counts=True)[1])
                for id_data_item in id_data
            ]
        )
    else:
        unique_ids = np.unique(id_data)
        id_data_dict = {uid: data[id_data == uid] for uid in unique_ids}
        min_samples_all_ids = np.min(np.unique(id_data, return_counts=True)[1])

    if min_samples_all_ids < context_window:
        raise ValueError(
            f"The minimum number of samples of all IDs is {min_samples_all_ids}, "
            f"but the context_window is {context_window}. "
            "Reduce the context_window or ensure each ID has enough data."
        )

    return data, id_data, id_data_dict, id_column_indices
