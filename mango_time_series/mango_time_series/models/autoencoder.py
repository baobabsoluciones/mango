import os
import pickle
from typing import Union, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.src.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from mango.logging import get_configured_logger
from mango.processing.data_imputer import DataImputer

from mango_time_series.models.modules import encoder, decoder
from mango_time_series.models.utils.plots import (
    plot_actual_and_reconstructed,
    plot_loss_history,
    plot_reconstruction_iterations,
)
from mango_time_series.models.utils.sequences import time_series_to_sequence

logger = get_configured_logger()


class AutoEncoder:
    """
    Autoencoder model

    This Autoencoder model can be highly configurable but is already set up so
    that quick training and profiling can be done.
    """

    def __init__(self):
        """
        Initialize the Autoencoder model
        """

        self.root_dir = os.path.abspath(os.getcwd())
        self.save_path = None
        self.model = None
        self.context_window = None
        self.time_step_to_check = None
        self.normalization_method = None
        self.normalization_values = {}
        self.imputer = None

    @classmethod
    def load_from_pickle(cls, path: str):
        """
        Load an AutoEncoder model from a pickle file.

        :param path: Path to the pickle file
        :type path: str
        :return: AutoEncoder model
        :rtype: AutoEncoder
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")

        try:
            # 1. Load the model and parameters from the pickle file
            with open(path, "rb") as f:
                saved_data = pickle.load(f)

            if "model" not in saved_data or "params" not in saved_data:
                raise ValueError("Invalid pickle file format: missing required keys.")

            model = saved_data["model"]
            params = saved_data["params"]

            # 2. Create an instance of AutoEncoder
            instance = cls()

            # 3. Assign loaded parameters
            instance.model = model
            instance.context_window = params.get("context_window")
            instance.time_step_to_check = params.get("time_step_to_check")
            instance.normalization_method = params.get("normalization_method")
            instance.features_name = params.get("features_name", None)
            instance.feature_to_check = params.get("feature_to_check", 0)

            # 4. Cargar los valores de normalización (global y por ID)
            instance.normalization_values = params.get("normalization_values", {})

            logger.info(f"Model successfully loaded from {path}")

            return instance

        except Exception as e:
            raise RuntimeError(f"Error loading the AutoEncoder model: {e}")

    @staticmethod
    def create_folder_structure(folder_structure: List[str]):
        """
        Create a folder structure if it does not exist.

        :param folder_structure: List of folders to create
        :type folder_structure: List[str]
        :return: None
        """
        for path in folder_structure:
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _time_series_split(
        data, train_size, val_size, test_size
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data into training, validation, and test sets according to the specified percentages.

        :param data: array-like data to split
        :type data: np.ndarray
        :param train_size: float
            Proportion of the dataset to include in the training set (0-1).
        :param val_size: float
            Proportion of the dataset to include in the validation set (0-1).
        :param test_size: float
            Proportion of the dataset to include in the test set (0-1).
        :return: Tuple[np.ndarray, np.ndarray, np.ndarray]
            The training, validation, and test sets as numpy arrays.
        """

        if train_size is None or val_size is None or test_size is None:
            raise ValueError(
                "train_size, val_size, and test_size must be specified and not None."
            )

        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError(
                f"The sum of train_size, val_size, and test_size must be 1.0, but got {train_size + val_size + test_size}."
            )

        # Original implementation for sequential split
        n = len(data)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)

        train_set = data[:train_end]
        val_set = data[train_end:val_end] if val_size > 0 else None
        test_set = data[val_end:] if test_size > 0 else None

        return train_set, val_set, test_set

    @staticmethod
    def _convert_data_to_numpy(data):
        """
        Convert data to numpy array format.

        Handles pandas and polars DataFrames, converting them to numpy arrays.
        If data is a tuple, converts each element in the tuple.

        :param data: Input data that can be pandas DataFrame, polars DataFrame,
            numpy array, or tuple of these types
        :type data: Any
        :return: Data converted to numpy array(s) and feature names if available
        :rtype: Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]], List[str]]
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
                AutoEncoder._convert_single_data_to_numpy(item, has_pandas, has_polars)
                for item in data
            )
            return converted_data, feature_names
        else:
            converted_data = AutoEncoder._convert_single_data_to_numpy(
                data, has_pandas, has_polars
            )
            return converted_data, feature_names

    @staticmethod
    def _convert_single_data_to_numpy(data_item, has_pandas, has_polars):
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

    def _normalize_data_for_training(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        id_iter: Optional[Union[str, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize training, validation and test data using the specified method.
        Computes and stores normalization parameters during training.

        :param x_train: Training data to normalize
        :type x_train: np.ndarray
        :param x_val: Validation data to normalize
        :type x_val: np.ndarray
        :param x_test: Test data to normalize
        :type x_test: np.ndarray
        :param id_iter: ID of the iteration for group-specific normalization, defaults to None
        :type id_iter: Optional[Union[str, int]]
        :return: Tuple containing normalized training, validation and test data
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]

        :note: The method stores normalization parameters either globally or per ID
               in the instance's normalization_values dictionary.
        """
        normalization_values = {}

        if self.normalization_method == "minmax":
            min_x = np.nanmin(x_train, axis=0)
            max_x = np.nanmax(x_train, axis=0)
            range_x = max_x - min_x
            x_train = (x_train - min_x) / range_x
            x_val = (x_val - min_x) / range_x
            x_test = (x_test - min_x) / range_x
            normalization_values = {"min_x": min_x, "max_x": max_x}
        elif self.normalization_method == "zscore":
            mean_ = np.nanmean(x_train, axis=0)
            std_ = np.nanstd(x_train, axis=0)
            x_train = (x_train - mean_) / std_
            x_val = (x_val - mean_) / std_
            x_test = (x_test - mean_) / std_
            normalization_values = {"mean_": mean_, "std_": std_}

        # Initialize normalization_values attribute if it doesn't exist
        if not hasattr(self, "normalization_values"):
            self.normalization_values = {}

        # Store normalization values by ID or globally
        if id_iter is not None:
            self.normalization_values[f"{id_iter}"] = normalization_values
        else:
            self.normalization_values["global"] = normalization_values

        return x_train, x_val, x_test

    def _normalize_data_for_prediction(
        self, data: np.ndarray, feature_to_check_filter: bool = False
    ) -> np.ndarray:
        """
        Normalize new data using stored normalization parameters.
        If parameters are not available, computes them from input data.

        :param data: New data to normalize
        :type data: np.ndarray
        :return: Normalized data
        :rtype: np.ndarray
        """
        if self.normalization_method == "minmax":
            if self.min_x is None or self.max_x is None:
                min_x = np.nanmin(data, axis=0)
                max_x = np.nanmax(data, axis=0)
                range_x = max_x - min_x
                self.min_x = min_x
                self.max_x = max_x
                return (data - min_x) / range_x
            else:
                if feature_to_check_filter:
                    range_x = (
                        self.max_x[self.feature_to_check]
                        - self.min_x[self.feature_to_check]
                    )
                    return (data - self.min_x[self.feature_to_check]) / range_x
                else:
                    range_x = self.max_x - self.min_x
                    return (data - self.min_x) / range_x

        elif self.normalization_method == "zscore":
            if self.mean_ is None or self.std_ is None:
                mean_ = np.nanmean(data, axis=0)
                std_ = np.nanstd(data, axis=0)
                self.mean_ = mean_
                self.std_ = std_
                return (data - mean_) / std_
            else:
                if feature_to_check_filter:
                    return (data - self.mean_[self.feature_to_check]) / self.std_[
                        self.feature_to_check
                    ]
                else:
                    return (data - self.mean_) / self.std_

    def _normalize_data(
        self,
        x_train: np.ndarray = None,
        x_val: np.ndarray = None,
        x_test: np.ndarray = None,
        data: np.ndarray = None,
        id_iter: Optional[Union[str, int]] = None,
        feature_to_check_filter: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Normalize data using the specified method.
        Can be used for training (x_train, x_val, x_test) or prediction (data).

        :param x_train: Training data for training mode
        :type x_train: np.ndarray, optional
        :param x_val: Validation data for training mode
        :type x_val: np.ndarray, optional
        :param x_test: Test data for training mode
        :type x_test: np.ndarray, optional
        :param data: New data to normalize for prediction mode
        :type data: np.ndarray, optional
        :param id_iter: ID of the iteration for group-specific normalization
        :type id_iter: Optional[Union[str, int]], optional
        :return: Normalized data in training or prediction mode
        :rtype: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        :raises ValueError: If neither training nor prediction data is provided
        """
        # Training mode
        if x_train is not None and x_val is not None and x_test is not None:
            return self._normalize_data_for_training(
                x_train, x_val, x_test, id_iter=id_iter
            )

        # Prediction mode
        elif data is not None:
            return self._normalize_data_for_prediction(
                data, feature_to_check_filter=feature_to_check_filter
            )

        else:
            raise ValueError(
                "Provide either (x_train, x_val, x_test) for training or `data` for prediction."
            )

    @staticmethod
    def _denormalize_data(
        data, normalization_method: str, min_x=None, max_x=None, mean_=None, std_=None
    ):
        """
        Denormalize data using stored normalization parameters.
        Assumes `_normalize_data` was used during training to store min_x/max_x or mean_/std_.

        :param data: Normalized data to denormalize
        :return: Denormalized data
        """
        if normalization_method not in ["minmax", "zscore"]:
            raise ValueError(
                "Invalid normalization method. Choose 'minmax' or 'zscore'."
            )

        if min_x is None and mean_ is None:
            raise ValueError(
                "No normalization parameters found. Ensure the model was trained with normalization."
            )

        if normalization_method == "minmax":
            return data * (max_x - min_x) + min_x
        elif normalization_method == "zscore":
            return data * std_ + mean_

    def prepare_datasets(
        self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        context_window: int,
        normalize: bool,
        id_iter: Optional[Union[str, int]] = None,
    ):
        """
        Prepare the datasets for the model training and testing.
        :param data: data to train the model. It can be a single numpy array
            with the whole dataset from which a train, validation and test split
            is created, or a tuple with three numpy arrays, one for
            the train, one for the validation and one for the test.
        :type data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        :param context_window: context window for the model
        :type context_window: int
        :param normalize: whether to normalize the data or not
        :type normalize: bool
        :param id_iter: id of the iteration
        :type id_iter: Optional[Union[str, int]]
        :return: True if the datasets are prepared successfully
        """
        # we need to set up two functions to prepare the datasets. One when data is a
        # single numpy array and one when data is a tuple with three numpy arrays.
        if isinstance(data, np.ndarray):
            x_train, x_val, x_test = self._time_series_split(
                data,
                self.train_size,
                self.val_size,
                self.test_size,
            )
            data = tuple([x_train, x_val, x_test])
        else:
            if not isinstance(data, tuple) or len(data) != 3:
                raise ValueError(
                    "Data must be a numpy array or a tuple with three numpy arrays"
                )

        return self._prepare_dataset(data, context_window, normalize, id_iter=id_iter)

    def _prepare_dataset(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        context_window: int,
        normalize: bool,
        id_iter: Optional[Union[str, int]] = None,
    ):
        """
        Prepare the dataset for the model training and testing when the data is a tuple with three numpy arrays.

        :param data: tuple with three numpy arrays for the train, validation and test datasets
        :type data: Tuple[np.ndarray, np.ndarray, np.ndarray]
        :param context_window: context window for the model
        :type context_window: int
        :param normalize: whether to normalize the data or not
        :type normalize: bool
        :param id_iter: id of the iteration
        :type id_iter: Optional[Union[str, int]]
        :return: True if the dataset is prepared successfully
        """
        x_train, x_val, x_test = data

        if self.use_mask:
            if self.custom_mask is None:
                mask_train = np.where(np.isnan(np.copy(x_train)), 0, 1)
                mask_val = np.where(np.isnan(np.copy(x_val)), 0, 1)
                mask_test = np.where(np.isnan(np.copy(x_test)), 0, 1)
            else:
                if isinstance(self.custom_mask, tuple):
                    mask_train, mask_val, mask_test = self.custom_mask
                else:
                    mask_train, mask_val, mask_test = self._time_series_split(
                        (
                            self.id_data_dict_mask[id_iter]
                            if id_iter is not None
                            else self.custom_mask
                        ),
                        self.train_size,
                        self.val_size,
                        self.test_size,
                    )

            seq_mask_train = time_series_to_sequence(mask_train, context_window)
            seq_mask_val = time_series_to_sequence(mask_val, context_window)
            seq_mask_test = time_series_to_sequence(mask_test, context_window)

        if normalize:
            x_train, x_val, x_test = self._normalize_data(
                x_train, x_val, x_test, id_iter=id_iter
            )

        if self.use_mask and self.imputer is not None:
            import pandas as pd

            x_train = self.imputer.apply_imputation(pd.DataFrame(x_train)).to_numpy()
            x_val = self.imputer.apply_imputation(pd.DataFrame(x_val)).to_numpy()
            x_test = self.imputer.apply_imputation(pd.DataFrame(x_test)).to_numpy()
        else:
            x_train = np.nan_to_num(x_train)
            x_val = np.nan_to_num(x_val)
            x_test = np.nan_to_num(x_test)

        seq_x_train = time_series_to_sequence(x_train, context_window)
        seq_x_val = time_series_to_sequence(x_val, context_window)
        seq_x_test = time_series_to_sequence(x_test, context_window)

        if id_iter is not None:
            self.data[id_iter] = (x_train, x_val, x_test)
            self.x_train[id_iter] = seq_x_train
            self.x_val[id_iter] = seq_x_val
            self.x_test[id_iter] = seq_x_test
            if self.use_mask:
                self.mask_train[id_iter] = seq_mask_train
                self.mask_val[id_iter] = seq_mask_val
                self.mask_test[id_iter] = seq_mask_test
        else:
            self.data = (seq_x_train, seq_x_val, seq_x_test)
            self.x_train = seq_x_train
            self.x_val = seq_x_val
            self.x_test = seq_x_test
            if self.use_mask:
                self.mask_train = seq_mask_train
                self.mask_val = seq_mask_val
                self.mask_test = seq_mask_test

        return True

    def masked_weighted_mse(self, y_true, y_pred, mask=None):
        """
        Compute Mean Squared Error (MSE) with optional masking and feature weights.

        :param y_true: Ground truth values (batch_size, seq_length, num_features)
        :param y_pred: Predicted values (batch_size, seq_length, num_features)
        :param mask: Optional binary mask (batch_size, seq_length, num_features), 1 for observed values, 0 for missing values
        :return: Masked and weighted MSE loss
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply mask if provided
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # Select the same time steps and features from the mask as we're using from the data
            # First select the time steps
            mask_selected = tf.gather(mask, self.time_step_to_check, axis=1)
            # Then select the features
            mask_selected = tf.gather(mask_selected, self.feature_to_check, axis=2)

            # Apply the mask to both true and predicted values
            y_true = tf.where(mask_selected > 0, y_true, tf.zeros_like(y_true))
            y_pred = tf.where(mask_selected > 0, y_pred, tf.zeros_like(y_pred))

        squared_error = tf.square(y_true - y_pred)

        # Apply feature-specific weights if provided
        if self.feature_weights is not None:
            feature_weights = tf.convert_to_tensor(
                self.feature_weights, dtype=tf.float32
            )
            squared_error = squared_error * feature_weights

        # Compute mean only over observed values if mask is provided
        if mask is not None:
            # Use the selected mask dimensions
            loss = tf.reduce_sum(squared_error) / (
                tf.reduce_sum(mask_selected + tf.keras.backend.epsilon())
            )
        else:
            loss = tf.reduce_mean(squared_error)

        return loss

    @staticmethod
    def _get_optimizer(optimizer_name: str):
        """
        Returns the optimizer based on the given name.
        """
        optimizers = {
            "adam": Adam(),
            "sgd": SGD(),
            "rmsprop": RMSprop(),
            "adagrad": Adagrad(),
            "adadelta": Adadelta(),
            "adamax": Adamax(),
            "nadam": Nadam(),
        }

        if optimizer_name.lower() not in optimizers:
            raise ValueError(
                f"Invalid optimizer '{optimizer_name}'. Choose from {list(optimizers.keys())}."
            )

        return optimizers[optimizer_name.lower()]

    def _handle_id_columns(self, data, id_columns):
        """
        Handle id_columns.

        :param data: Data to train the model.
        :type data: Any
        :param id_columns: Column(s) to process the data by groups.
        :type id_columns: Union[str, int, List[str], List[int], None]

        :return: Processed data, ID mapping, and a dictionary with grouped data by ID.
        :rtype: Tuple[np.ndarray, np.ndarray, dict]
        """
        self.id_columns_indices = []

        if id_columns is None:
            return data, None, {}

        id_columns = [id_columns] if isinstance(id_columns, (str, int)) else id_columns

        if all(isinstance(i, str) for i in id_columns):
            id_column_indices = [
                i for i, value in enumerate(self.features_name) if value in id_columns
            ]
        elif all(isinstance(i, int) for i in id_columns):
            id_column_indices = id_columns
        else:
            raise ValueError("id_columns must be a list of strings or integers")

        self.id_columns_indices = id_column_indices

        if isinstance(data, tuple):
            id_data = tuple(
                np.array(["__".join(map(str, row)) for row in d[:, id_column_indices]])
                for d in data
            )
            data = tuple(
                np.delete(d, id_column_indices, axis=1).astype(np.float64) for d in data
            )
        else:
            id_data = np.array(
                ["__".join(map(str, row)) for row in data[:, id_column_indices]]
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
                    np.min(np.unique(id_data, return_counts=True)[1])
                    for id_data in id_data
                ]
            )
        else:
            unique_ids = np.unique(id_data)
            id_data_dict = {uid: data[id_data == uid] for uid in unique_ids}
            min_samples_all_ids = np.min(np.unique(id_data, return_counts=True)[1])

        if min_samples_all_ids < self.context_window:
            raise ValueError(
                f"The minimum number of samples of all IDs is {min_samples_all_ids}, "
                f"but the context_window is {self.context_window}. "
                "Reduce the context_window or ensure each ID has enough data."
            )

        return data, id_data, id_data_dict

    def build_model(
        self,
        form: str = "lstm",
        data: Any = None,
        context_window: int = None,
        time_step_to_check: Union[int, List[int]] = 0,
        feature_to_check: Union[int, List[int]] = 0,
        hidden_dim: Union[int, List[int]] = None,
        bidirectional_encoder: bool = False,
        bidirectional_decoder: bool = False,
        activation_encoder: str = None,
        activation_decoder: str = None,
        normalize: bool = False,
        normalization_method: str = "minmax",
        optimizer: str = "adam",
        batch_size: int = 32,
        save_path: str = None,
        verbose: bool = False,
        feature_names: Optional[List[str]] = None,
        feature_weights: Optional[List[float]] = None,
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None,
        use_mask: bool = False,
        custom_mask: Any = None,
        imputer: Optional[DataImputer] = None,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        id_columns: Union[str, int, List[str], List[int], None] = None,
    ):
        """
        Build the Autoencoder model.

        :param form: type of encoder, one of "dense", "rnn", "gru" or "lstm".
          Currently, these types of cells are both used on the encoder and
          decoder. In the future each part could have a different structure?
        :type form: str
        :param data: data to train the model. It can be:
          - A single numpy array, pandas DataFrame, or polars DataFrame
            from which train, validation and test splits are created
          - A tuple with three numpy arrays, pandas DataFrames, or polars DataFrames
            for train, validation, and test sets respectively
        :type data: Any
        :param context_window: context window for the model. This is used
          to transform the tabular data into a sequence of data
          (from 2D tensor to 3D tensor)
        :type context_window: int
        :param time_step_to_check: time steps to check for the autoencoder.
          Currently only int value is supported and it should be the index
          of the context window to check. In the future this could be a list of
          indices to check. For taking only the last timestep of the context
          window this should be set to -1.
        :type time_step_to_check: Union[int, List[int]]
        :param hidden_dim: number of hidden dimensions in the internal layers.
          It can be a single integer (same for all layers) or a list of
          dimensions for each layer.
        :type hidden_dim: Union[int, List[int]]
        :param bidirectional_encoder: whether to use bidirectional LSTM in the
            encoder part of the model.
        :type bidirectional_encoder: bool
        :param bidirectional_decoder: whether to use bidirectional LSTM in the
            decoder part of the model.
        :type bidirectional_decoder: bool
        :param activation_encoder: activation function for the encoder layers.
        :type activation_encoder: str
        :param activation_decoder: activation function for the decoder layers.
        :type activation_decoder: str
        :param normalize: whether to normalize the data or not.
        :type normalize: bool
        :param normalization_method: method to normalize the data. It can be
            "minmax" or "zscore".
        :type normalization_method: str
        :param batch_size: batch size for the model
        :type batch_size: int
        :param save_path: folder path to save the model checkpoints
        :type save_path: str
        :param verbose: whether to log model summary and model training.
        :type verbose: bool
        :param feature_names: optional list of feature names to use for the model.
            If provided, these names will be used instead of automatically extracted ones.
        :type feature_names: Optional[List[str]]
        :param feature_weights: optional list of feature weights to use for the model.
            If provided, these weights will be used to scale the loss for each feature.
        :type feature_weights: Optional[List[float]]
        :param shuffle: whether to shuffle the training dataset
        :type shuffle: bool
        :param use_mask: whether to use a mask for missing values
        :type use_mask: bool
        :param custom_mask: optional custom mask to use for missing values
        :type custom_mask: Optional[np.array]
        :param imputer: optional imputer to use for missing values
        :type imputer: Optional[DataImputer]
        :param train_size: proportion of the dataset to include in the training set
        :type train_size: float
        :param val_size: proportion of the dataset to include in the validation set
        :type val_size: float
        :param test_size: proportion of the dataset to include in the test set
        :type test_size: float
        :param id_columns: optional column(s) to process the data by groups.
            If provided, the data will be grouped by this column and processed separately.
            Can be a column name (str), a column index (int), or a list of either.
        :type id_columns: Union[str, int, List[str], List[int], None]
        :raises NotImplementedError: If the model type is 'dense'
        """
        if form == "dense":
            raise NotImplementedError("Dense model type is not yet implemented")

        self.save_path = (
            save_path if save_path else os.path.join(self.root_dir, "autoencoder")
        )

        self.create_folder_structure(
            [
                os.path.join(self.save_path, "models"),
                os.path.join(self.save_path, "plots"),
            ]
        )
        self.context_window = context_window

        if isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        elif isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim]
        else:
            raise ValueError("hidden_dim must be a list of integers or an integer")

        num_layers = len(self.hidden_dim)

        # Convert data to numpy arrays if it's a pandas or polars DataFrame
        # and extract feature names if available
        data, extracted_feature_names = self._convert_data_to_numpy(data)
        # Store feature names or generate default names
        if feature_names:
            # Use user-provided feature names
            self.features_name = feature_names
        elif extracted_feature_names and len(extracted_feature_names) > 0:
            # Use extracted feature names from DataFrame
            self.features_name = extracted_feature_names
        else:
            # If data is provided, create generic feature names based on the number of features
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                num_features = data.shape[1]
                self.features_name = [f"feature_{i}" for i in range(num_features)]
            elif (
                isinstance(data, tuple)
                and all(isinstance(d, np.ndarray) for d in data)
                and data[0].ndim >= 2
            ):
                num_features = data[0].shape[1]
                self.features_name = [f"feature_{i}" for i in range(num_features)]
            else:
                self.features_name = []

        # Store feature weights if provided
        self.use_mask = use_mask
        self.custom_mask = custom_mask
        if self.use_mask and self.custom_mask is not None:
            self.custom_mask, _ = self._convert_data_to_numpy(self.custom_mask)
            mask, self.id_data_mask, self.id_data_dict_mask = self._handle_id_columns(
                self.custom_mask, id_columns
            )

        data, self.id_data, self.id_data_dict = self._handle_id_columns(
            data, id_columns
        )

        if self.use_mask and self.custom_mask is not None and self.id_data is not None:
            if isinstance(self.id_data, tuple) and isinstance(self.id_data_mask, tuple):
                if any((id_d != id_m).all() for id_d, id_m in zip(self.id_data, self.id_data_mask)):
                    raise ValueError("The mask must have the same IDs as the data.")
            else:
                if (self.id_data_mask != self.id_data).all():
                    raise ValueError("The mask must have the same IDs as the data.")

        # Now we check if data is a single numpy array or a tuple with three numpy arrays
        if isinstance(data, tuple):
            if len(data) != 3:
                raise ValueError("Data must be a tuple with three numpy arrays")
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise ValueError(
                "Data must be a numpy array or a tuple with three numpy arrays"
            )

        if not self.use_mask:
            if isinstance(data, tuple):
                if any(np.isnan(d).any() for d in data):
                    raise ValueError(
                        "Data contains NaNs in one or more splits (train, val, test), "
                        "but use_mask is False. Please preprocess data to remove or impute NaNs."
                    )
            else:
                if np.isnan(data).any():
                    raise ValueError(
                        "Data contains NaNs, but use_mask is False. Please preprocess data to remove or impute NaNs."
                    )

        bidirectional_allowed = {"lstm", "gru", "rnn"}

        if form not in bidirectional_allowed:
            if bidirectional_encoder and bidirectional_decoder:
                raise ValueError(
                    f"Bidirectional is not supported for encoder and decoder type '{form}'."
                )
            elif bidirectional_encoder:
                raise ValueError(
                    f"Bidirectional is not supported for encoder type '{form}'."
                )
            elif bidirectional_decoder:
                raise ValueError(
                    f"Bidirectional is not supported for decoder type '{form}'."
                )

        if normalization_method not in ["minmax", "zscore"]:
            raise ValueError(
                "Invalid normalization method. Choose 'minmax' or 'zscore'."
            )

        self.normalization_method = normalization_method

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        self.imputer = imputer
        if self.id_data is not None:
            self.data = {}
            self.x_train = {}
            self.x_val = {}
            self.x_test = {}
            self.mask_train = {}
            self.mask_val = {}
            self.mask_test = {}
            for id_iter in self.id_data_dict:
                self.prepare_datasets(
                    self.id_data_dict[id_iter],
                    context_window,
                    normalize,
                    id_iter=id_iter,
                )

            # Extract the length of the datasets for each id to use it on the reconstruction
            self.length_datasets = {}
            for id_iter in self.id_data_dict:
                self.length_datasets[id_iter] = {}
                self.length_datasets[id_iter]["train"] = len(self.x_train[id_iter])
                self.length_datasets[id_iter]["val"] = len(self.x_val[id_iter])
                self.length_datasets[id_iter]["test"] = len(self.x_test[id_iter])

            # Concat all the datasets
            self.x_train = np.concatenate(
                [self.x_train[id_iter] for id_iter in sorted(self.id_data_dict.keys())],
                axis=0,
            )
            self.x_val = np.concatenate(
                [self.x_val[id_iter] for id_iter in sorted(self.id_data_dict.keys())],
                axis=0,
            )
            self.x_test = np.concatenate(
                [self.x_test[id_iter] for id_iter in sorted(self.id_data_dict.keys())],
                axis=0,
            )
            if self.use_mask:
                self.mask_train = np.concatenate(
                    [
                        self.mask_train[id_iter]
                        for id_iter in sorted(self.id_data_dict.keys())
                    ],
                    axis=0,
                )
                self.mask_val = np.concatenate(
                    [
                        self.mask_val[id_iter]
                        for id_iter in sorted(self.id_data_dict.keys())
                    ],
                    axis=0,
                )
                self.mask_test = np.concatenate(
                    [
                        self.mask_test[id_iter]
                        for id_iter in sorted(self.id_data_dict.keys())
                    ],
                    axis=0,
                )
        else:
            self.prepare_datasets(data, context_window, normalize)
        self.normalize = normalize

        if isinstance(feature_to_check, int):
            feature_to_check = [feature_to_check]
        self.feature_to_check = feature_to_check

        self.input_features = self.x_train.shape[2]
        self.output_features = len(self.feature_to_check)

        self.shuffle = shuffle

        if self.shuffle:
            if shuffle_buffer_size is not None:
                if not isinstance(shuffle_buffer_size, int) or shuffle_buffer_size <= 0:
                    raise ValueError("shuffle_buffer_size must be a positive integer.")
                self.shuffle_buffer_size = shuffle_buffer_size
            else:
                self.shuffle_buffer_size = len(self.x_train)
        else:
            self.shuffle_buffer_size = None

        self.x_train_no_shuffle = np.copy(self.x_train)

        if self.use_mask:
            if self.custom_mask is not None:
                if isinstance(data, tuple) and (
                    not isinstance(self.custom_mask, tuple)
                    or len(self.custom_mask) != 3
                ):
                    raise ValueError(
                        "If data is a tuple, custom_mask must also be a tuple of the same length (train, val, test)."
                    )

                if not isinstance(data, tuple) and isinstance(self.custom_mask, tuple):
                    raise ValueError(
                        "If data is a single array, custom_mask cannot be a tuple."
                    )

                if isinstance(self.custom_mask, tuple):
                    if (
                        mask[0].shape != data[0].shape
                        or mask[1].shape != data[1].shape
                        or mask[2].shape != data[2].shape
                    ):
                        raise ValueError(
                            "Each element of custom_mask must have the same shape as its corresponding dataset "
                            "(mask_train with x_train, mask_val with x_val, mask_test with x_test)."
                        )
                else:
                    if mask.shape != data.shape:
                        raise ValueError(
                            "custom_mask must have the same shape as the original input data before transformation"
                        )

            # Check if masks and data have the same shape
            if (
                self.mask_train.shape != self.x_train.shape
                or self.mask_val.shape != self.x_val.shape
                or self.mask_test.shape != self.x_test.shape
            ):
                raise ValueError(
                    "Masks must have the same shape as the data after transformation."
                )

            train_dataset = tf.data.Dataset.from_tensor_slices(
                (self.x_train, self.mask_train)
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (self.x_val, self.mask_val)
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (self.x_test, self.mask_test)
            )

        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
            val_dataset = tf.data.Dataset.from_tensor_slices(self.x_val)
            test_dataset = tf.data.Dataset.from_tensor_slices(self.x_test)

        if self.shuffle:
            self.train_dataset = train_dataset.shuffle(
                buffer_size=self.shuffle_buffer_size
            )
        self.train_dataset = train_dataset.cache().batch(batch_size)
        self.val_dataset = val_dataset.cache().batch(batch_size)
        self.test_dataset = test_dataset.cache().batch(batch_size)

        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder

        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder

        model = Sequential(
            [
                encoder(
                    form=form,
                    context_window=context_window,
                    features=self.input_features,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    use_bidirectional=self.bidirectional_encoder,
                    activation=self.activation_encoder,
                    verbose=verbose,
                ),
                decoder(
                    form=form,
                    context_window=context_window,
                    features=self.output_features,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    use_bidirectional=self.bidirectional_decoder,
                    activation=self.activation_decoder,
                    verbose=verbose,
                ),
            ],
            name="autoencoder",
        )
        model.build()

        if verbose:
            logger.info(f"The model has the following structure: {model.summary()}")

        self.form = form
        self.model = model

        self.optimizer_name = optimizer
        self.model_optimizer = self._get_optimizer(optimizer)

        if isinstance(time_step_to_check, int):
            time_step_to_check = [time_step_to_check]
        self.time_step_to_check = time_step_to_check

        max_time_step = self.context_window - 1
        if any(t > max_time_step for t in self.time_step_to_check):
            raise ValueError(
                f"time_step_to_check contains invalid indices. Must be between 0 and {max_time_step}."
            )

        self.verbose = verbose
        self.feature_weights = feature_weights

    def train(
        self,
        epochs: int = 100,
        checkpoint: int = 10,
        use_early_stopping: bool = True,
        patience: int = 10,
    ):
        """
        Train the model using the train and validation datasets and save the best model.

        :param epochs: number of epochs to train the model
        :type epochs: int
        :param checkpoint: number of epochs to save a checkpoint
        :type checkpoint: int
        :param use_early_stopping: whether to use early stopping or not
        :type use_early_stopping: bool
        :param patience: number of epochs to wait before stopping the training
        :type patience: int

        :return: None
        """
        self.last_epoch = 0
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.train_loss_history = None
        self.val_loss_history = None
        self.use_early_stopping = use_early_stopping
        self.patience = patience

        @tf.function
        def train_step(x, mask=None):
            """
            Training step for the model.
            :param x: input data
            :param mask: optional binary mask for missing
            """
            with tf.GradientTape() as autoencoder_tape:
                x = tf.cast(x, tf.float32)

                hx = self.model.get_layer(f"{self.form}_encoder")(x)
                x_hat = self.model.get_layer(f"{self.form}_decoder")(hx)

                # Gather all required time steps
                x_real = tf.gather(x, self.time_step_to_check, axis=1)
                x_real = tf.gather(x_real, self.feature_to_check, axis=2)

                x_pred = tf.expand_dims(x_hat, axis=1)

                # Calculate mean loss across all selected points
                train_loss = self.masked_weighted_mse(x_real, x_pred, mask)

            autoencoder_gradient = autoencoder_tape.gradient(
                train_loss, self.model.trainable_variables
            )

            self.model_optimizer.apply_gradients(
                zip(autoencoder_gradient, self.model.trainable_variables)
            )

            return train_loss

        @tf.function
        def validation_step(x, mask=None):
            """
            Validation step for the model.
            :param x: input data
            :param mask: optional binary mask for missing
            """
            x = tf.cast(x, tf.float32)

            hx = self.model.get_layer(f"{self.form}_encoder")(x)
            x_hat = self.model.get_layer(f"{self.form}_decoder")(hx)

            # Gather all required time steps
            x_real = tf.gather(x, self.time_step_to_check, axis=1)
            x_real = tf.gather(x_real, self.feature_to_check, axis=2)

            x_pred = tf.expand_dims(x_hat, axis=1)
            # Calculate mean loss across all selected points
            val_loss = self.masked_weighted_mse(x_real, x_pred, mask)

            return val_loss

        # Lists to store loss history
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Training loop
            epoch_train_losses = []
            for batch in self.train_dataset:
                if self.use_mask:
                    data, mask = batch
                else:
                    data = batch
                    mask = None

                loss = train_step(x=data, mask=mask)
                epoch_train_losses.append(float(loss))

            # Calculate average training loss for the epoch
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_loss_history.append(avg_train_loss)

            # Validation loop
            epoch_val_losses = []
            for batch in self.val_dataset:
                if self.use_mask:
                    data, mask = batch
                else:
                    data = batch
                    mask = None

                val_loss = validation_step(x=data, mask=mask)
                epoch_val_losses.append(float(val_loss))

            # Calculate average validation loss for the epoch
            avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
            val_loss_history.append(avg_val_loss)

            self.last_epoch = epoch

            # Early stopping logic
            if self.use_early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    self.save(filename="best_model.pkl")
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} | Best Validation Loss: {best_val_loss:.6f}"
                    )
                    break

            if epoch % self.checkpoint == 0:
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"Training Loss: {avg_train_loss:.6f} | "
                        f"Validation Loss: {avg_val_loss:.6f}"
                    )

                self.save(filename=f"{epoch}.pkl")

                # Store the loss history in the model instance
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

        # Plot loss history
        plot_loss_history(
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            save_path=os.path.join(self.save_path, "plots"),
        )

        self.save(filename=f"{self.last_epoch}.pkl")

    def reconstruct(self):
        """
        Reconstruct the data using the trained model and plot the actual and reconstructed values.
        """
        # Calculate fitted values for each dataset
        # We use the original data for the training set to avoid shuffling in reconstruction step
        x_hat_train = self.model(self.x_train_no_shuffle)
        x_hat_val = self.model(self.x_val)
        x_hat_test = self.model(self.x_test)

        # Convert to numpy arrays
        x_hat_train = x_hat_train.numpy()
        x_hat_val = x_hat_val.numpy()
        x_hat_test = x_hat_test.numpy()

        # Get the original data for comparison
        x_train_converted = np.copy(
            self.x_train[:, self.time_step_to_check, self.feature_to_check]
        )
        x_val_converted = np.copy(
            self.x_val[:, self.time_step_to_check, self.feature_to_check]
        )
        x_test_converted = np.copy(
            self.x_test[:, self.time_step_to_check, self.feature_to_check]
        )

        # Handle denormalization if normalization was applied
        if self.normalize:
            # Use the global normalization values if ID-based normalization wasn't used
            if "global" in self.normalization_values:
                norm_values = self.normalization_values["global"]

                if self.normalization_method == "minmax":
                    scale_min = norm_values["min_x"][self.feature_to_check]
                    scale_max = norm_values["max_x"][self.feature_to_check]

                    # Denormalize predictions
                    x_hat_train = x_hat_train * (scale_max - scale_min) + scale_min
                    x_hat_val = x_hat_val * (scale_max - scale_min) + scale_min
                    x_hat_test = x_hat_test * (scale_max - scale_min) + scale_min

                    # Denormalize original data
                    x_train_converted = (
                        x_train_converted * (scale_max - scale_min) + scale_min
                    )
                    x_val_converted = (
                        x_val_converted * (scale_max - scale_min) + scale_min
                    )
                    x_test_converted = (
                        x_test_converted * (scale_max - scale_min) + scale_min
                    )

                elif self.normalization_method == "zscore":
                    scale_mean = norm_values["mean_"][self.feature_to_check]
                    scale_std = norm_values["std_"][self.feature_to_check]

                    # Denormalize predictions
                    x_hat_train = x_hat_train * scale_std + scale_mean
                    x_hat_val = x_hat_val * scale_std + scale_mean
                    x_hat_test = x_hat_test * scale_std + scale_mean

                    # Denormalize original data
                    x_train_converted = x_train_converted * scale_std + scale_mean
                    x_val_converted = x_val_converted * scale_std + scale_mean
                    x_test_converted = x_test_converted * scale_std + scale_mean

            # If we used ID-based normalization and need to reconstruct by ID
            elif self.id_data is not None and hasattr(self, "length_datasets"):
                logger.info("Performing ID-based denormalization")

                # Initialize arrays to store denormalized data
                denorm_x_hat_train = []
                denorm_x_hat_val = []
                denorm_x_hat_test = []
                denorm_x_train = []
                denorm_x_val = []
                denorm_x_test = []

                # Keep track of current positions in the datasets
                train_start_idx = 0
                val_start_idx = 0
                test_start_idx = 0

                # Process each ID separately
                for id_key in sorted(self.length_datasets.keys()):
                    # Get the normalization values for this ID
                    if id_key not in self.normalization_values:
                        logger.warning(
                            f"No normalization values found for {id_key}, skipping"
                        )
                        continue

                    norm_values = self.normalization_values[id_key]

                    # Get dataset lengths for this ID
                    train_length = self.length_datasets[id_key]["train"]
                    val_length = self.length_datasets[id_key]["val"]
                    test_length = self.length_datasets[id_key]["test"]

                    # Extract segments for this ID
                    train_end_idx = train_start_idx + train_length
                    val_end_idx = val_start_idx + val_length
                    test_end_idx = test_start_idx + test_length

                    id_x_hat_train = x_hat_train[train_start_idx:train_end_idx]
                    id_x_hat_val = x_hat_val[val_start_idx:val_end_idx]
                    id_x_hat_test = x_hat_test[test_start_idx:test_end_idx]

                    id_x_train = x_train_converted[train_start_idx:train_end_idx]
                    id_x_val = x_val_converted[val_start_idx:val_end_idx]
                    id_x_test = x_test_converted[test_start_idx:test_end_idx]

                    # Apply denormalization based on the normalization method
                    if self.normalization_method == "minmax":
                        scale_min = norm_values["min_x"][self.feature_to_check]
                        scale_max = norm_values["max_x"][self.feature_to_check]

                        # Denormalize predictions
                        id_x_hat_train = (
                            id_x_hat_train * (scale_max - scale_min) + scale_min
                        )
                        id_x_hat_val = (
                            id_x_hat_val * (scale_max - scale_min) + scale_min
                        )
                        id_x_hat_test = (
                            id_x_hat_test * (scale_max - scale_min) + scale_min
                        )

                        # Denormalize original data
                        id_x_train = id_x_train * (scale_max - scale_min) + scale_min
                        id_x_val = id_x_val * (scale_max - scale_min) + scale_min
                        id_x_test = id_x_test * (scale_max - scale_min) + scale_min

                    elif self.normalization_method == "zscore":
                        scale_mean = norm_values["mean_"][self.feature_to_check]
                        scale_std = norm_values["std_"][self.feature_to_check]

                        # Denormalize predictions
                        id_x_hat_train = id_x_hat_train * scale_std + scale_mean
                        id_x_hat_val = id_x_hat_val * scale_std + scale_mean
                        id_x_hat_test = id_x_hat_test * scale_std + scale_mean

                        # Denormalize original data
                        id_x_train = id_x_train * scale_std + scale_mean
                        id_x_val = id_x_val * scale_std + scale_mean
                        id_x_test = id_x_test * scale_std + scale_mean

                    # Store denormalized data
                    denorm_x_hat_train.append(id_x_hat_train)
                    denorm_x_hat_val.append(id_x_hat_val)
                    denorm_x_hat_test.append(id_x_hat_test)
                    denorm_x_train.append(id_x_train)
                    denorm_x_val.append(id_x_val)
                    denorm_x_test.append(id_x_test)

                    # Update indices for next iteration
                    train_start_idx += train_length
                    val_start_idx += val_length
                    test_start_idx += test_length

                # Concatenate all denormalized data
                if denorm_x_hat_train:
                    x_hat_train = np.concatenate(denorm_x_hat_train, axis=0)
                    x_hat_val = np.concatenate(denorm_x_hat_val, axis=0)
                    x_hat_test = np.concatenate(denorm_x_hat_test, axis=0)
                    x_train_converted = np.concatenate(denorm_x_train, axis=0)
                    x_val_converted = np.concatenate(denorm_x_val, axis=0)
                    x_test_converted = np.concatenate(denorm_x_test, axis=0)
                else:
                    logger.warning("No IDs were successfully denormalized")

        # Combine the datasets
        x_hat = np.concatenate((x_hat_train.T, x_hat_val.T, x_hat_test.T), axis=1)
        x_converted = np.concatenate(
            (x_train_converted.T, x_val_converted.T, x_test_converted.T), axis=1
        )

        # Get feature labels for the selected features, if we have feature names, extract only those that correspond to feature_to_check
        features_names_without_id = [
            feature
            for i, feature in enumerate(self.features_name)
            if i not in self.id_columns_indices
        ]
        feature_labels = (
            [features_names_without_id[i] for i in self.feature_to_check]
            if hasattr(self, "features_name")
            else None
        )

        # Get the split indices
        train_split = self.x_train.shape[0]
        val_split = train_split + self.x_val.shape[0]

        plot_actual_and_reconstructed(
            actual=x_converted,
            reconstructed=x_hat,
            save_path=os.path.join(self.save_path, "plots"),
            feature_labels=feature_labels,
            train_split=train_split,
            val_split=val_split,
            length_datasets=self.length_datasets if self.id_data_dict != {} else None,
        )

        return True

    def save(self, save_path: str = None, filename: str = "model.pkl"):
        """
        Save the model (Keras model + training parameters) into a single .pkl file.

        :param save_path: Path to save the model.
        :type save_path: str
        :param filename: Name of the file to save the model.
        :type filename: str
        """
        try:
            save_path = save_path or self.save_path
            os.makedirs(os.path.join(save_path, "models"), exist_ok=True)

            model_path = os.path.join(save_path, "models", filename)

            training_params = {
                "context_window": self.context_window,
                "time_step_to_check": self.time_step_to_check,
                "normalization_method": (
                    self.normalization_method if self.normalize else None
                ),
                "normalization_values": {},
                "features_name": self.features_name,
                "feature_to_check": self.feature_to_check,
            }

            if self.normalize:
                if hasattr(self, "normalization_values") and isinstance(
                    self.normalization_values, dict
                ):
                    training_params["normalization_values"] = self.normalization_values
                else:
                    training_params["normalization_values"]["global"] = {
                        "min_x": (
                            self.min_x.tolist() if self.min_x is not None else None
                        ),
                        "max_x": (
                            self.max_x.tolist() if self.max_x is not None else None
                        ),
                        "mean_": (
                            self.mean_.tolist() if self.mean_ is not None else None
                        ),
                        "std_": self.std_.tolist() if self.std_ is not None else None,
                    }

            with open(model_path, "wb") as f:
                pickle.dump({"model": self.model, "params": training_params}, f)

            logger.info(f"Model and parameters saved in: {model_path}")

        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise

    def build_and_train(
        self,
        data: Any = None,
        context_window: int = None,
        time_step_to_check: Union[int, List[int]] = 0,
        feature_to_check: Union[int, List[int]] = 0,
        form: str = "lstm",
        hidden_dim: Union[int, List[int]] = None,
        bidirectional_encoder: bool = False,
        bidirectional_decoder: bool = False,
        activation_encoder: str = None,
        activation_decoder: str = None,
        normalize: bool = False,
        normalization_method: str = "minmax",
        optimizer: str = "adam",
        batch_size: int = 32,
        save_path: str = None,
        verbose: bool = False,
        feature_names: Optional[List[str]] = None,
        feature_weights: Optional[List[float]] = None,
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None,
        use_mask: bool = False,
        custom_mask: Any = None,
        imputer: Optional[DataImputer] = None,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        id_columns: Union[str, int, List[str], List[int], None] = None,
        epochs: int = 100,
        checkpoint: int = 10,
        use_early_stopping: bool = True,
        patience: int = 10,
    ):
        """
        Build and train the Autoencoder model in a single step.

        This method combines the functionality of `build_model` and `train` methods,
        allowing for a more streamlined workflow.

        :param data: Data to train the model. It can be:
          - A single numpy array, pandas DataFrame, or polars DataFrame
            from which train, validation and test splits are created
          - A tuple with three numpy arrays, pandas DataFrames, or polars DataFrames
            for train, validation, and test sets respectively
        :type data: Any
        :param context_window: Context window for the model used to transform
            tabular data into sequence data (2D tensor to 3D tensor)
        :type context_window: int
        :param time_step_to_check: Time steps to check for the autoencoder
        :type time_step_to_check: Union[int, List[int]]
        :param feature_to_check: Features to check in the autoencoder
        :type feature_to_check: Union[int, List[int]]
        :param form: Type of encoder, one of "dense", "rnn", "gru" or "lstm"
        :type form: str
        :param hidden_dim: Number of hidden dimensions in the internal layers
        :type hidden_dim: Union[int, List[int]]
        :param bidirectional_encoder: Whether to use bidirectional LSTM in encoder
        :type bidirectional_encoder: bool
        :param bidirectional_decoder: Whether to use bidirectional LSTM in decoder
        :type bidirectional_decoder: bool
        :param activation_encoder: Activation function for the encoder layers
        :type activation_encoder: str
        :param activation_decoder: Activation function for the decoder layers
        :type activation_decoder: str
        :param normalize: Whether to normalize the data
        :type normalize: bool
        :param normalization_method: Method to normalize the data "minmax" or "zscore"
        :type normalization_method: str
        :param optimizer: Optimizer to use for training
        :type optimizer: str
        :param batch_size: Batch size for training
        :type batch_size: int
        :param save_path: Folder path to save model checkpoints
        :type save_path: str
        :param verbose: Whether to log model summary and training progress
        :type verbose: bool
        :param feature_names: List of feature names to use
        :type feature_names: Optional[List[str]]
        :param feature_weights: List of feature weights for loss scaling
        :type feature_weights: Optional[List[float]]
        :param shuffle: Whether to shuffle the training dataset
        :type shuffle: bool
        :param shuffle_buffer_size: Buffer size for shuffling
        :type shuffle_buffer_size: Optional[int]
        :param use_mask: Whether to use a mask for missing values
        :type use_mask: bool
        :param custom_mask: Custom mask to use for missing values
        :type custom_mask: Any
        :param imputer: Imputer to use for missing values
        :type imputer: Optional[DataImputer]
        :param train_size: Proportion of dataset for training
        :type train_size: float
        :param val_size: Proportion of dataset for validation
        :type val_size: float
        :param test_size: Proportion of dataset for testing
        :type test_size: float
        :param id_columns: Column(s) to process data by groups
        :type id_columns: Union[str, int, List[str], List[int], None]
        :param epochs: Number of epochs for training
        :type epochs: int
        :param checkpoint: Number of epochs between model checkpoints
        :type checkpoint: int
        :param use_early_stopping: Whether to use early stopping
        :type use_early_stopping: bool
        :param patience: Number of epochs to wait before early stopping
        :type patience: int
        :return: Self for method chaining
        :rtype: AutoEncoder
        """

        self.build_model(
            form=form,
            data=data,
            context_window=context_window,
            time_step_to_check=time_step_to_check,
            feature_to_check=feature_to_check,
            hidden_dim=hidden_dim,
            bidirectional_encoder=bidirectional_encoder,
            bidirectional_decoder=bidirectional_decoder,
            activation_encoder=activation_encoder,
            activation_decoder=activation_decoder,
            normalize=normalize,
            normalization_method=normalization_method,
            optimizer=optimizer,
            batch_size=batch_size,
            save_path=save_path,
            verbose=verbose,
            feature_names=feature_names,
            feature_weights=feature_weights,
            shuffle=shuffle,
            shuffle_buffer_size=shuffle_buffer_size,
            use_mask=use_mask,
            custom_mask=custom_mask,
            imputer=imputer,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            id_columns=id_columns,
        )

        self.train(
            epochs=epochs,
            checkpoint=checkpoint,
            use_early_stopping=use_early_stopping,
            patience=patience,
        )

        return self

    @staticmethod
    def _apply_padding(data, reconstructed, context_window, time_step_to_check):
        """
        Apply padding dynamically based on time_step_to_check and context_window.

        :param data: Original dataset shape (num_samples, num_features)
        :param reconstructed: Predicted values shape (num_samples - context_window, num_features)
        :param context_window: Context window size
        :param time_step_to_check: Time step to predict within the window
        :return: Padded reconstructed dataset
        """
        num_samples, num_features = data.shape
        padded_reconstructed = np.full((num_samples, num_features), np.nan)

        # Determine the offset based on time_step_to_check
        # TODO: Improve this logic to handle multiple time steps
        if isinstance(time_step_to_check, list):
            time_step_to_check = time_step_to_check[0]
        if time_step_to_check == 0:
            padded_reconstructed[: num_samples - (context_window - 1)] = reconstructed
        elif time_step_to_check == context_window - 1:
            padded_reconstructed[context_window - 1 :] = reconstructed
        else:
            before = time_step_to_check
            after = context_window - 1 - time_step_to_check
            padded_reconstructed[before : num_samples - after] = reconstructed

        return padded_reconstructed

    # def reconstruct_new_data_before(self, data, iterations: int = None):
    #     """
    #     Predict and reconstruct unknown data, iterating over NaN values to improve predictions.
    #     Uses stored `context_window`, normalization parameters, and the trained model.

    #     :param data: Input data (numpy array, pandas DataFrame, or polars DataFrame).
    #     :param iterations: Number of reconstruction iterations (None = no iteration).
    #     :param id_columns: Column(s) to process data by groups.
    #     :return: Reconstructed data.
    #     """

    #     if self.model is None:
    #         raise ValueError(
    #             "No model loaded. Use `load_from_pickle()` before calling `reconstruct_new_data()`."
    #         )

    #     normalization_used = self.normalization_method is not None
    #     data, feature_names = self._convert_data_to_numpy(data)

    #     data_original = np.copy(data)
    #     nan_positions = np.isnan(data)
    #     has_nans = np.any(nan_positions)
    #     reconstructed_iterations = {}

    #     # Case 1: No NaNs and no iterations (simple prediction)
    #     if not has_nans:
    #         if normalization_used:
    #             try:
    #                 data = self._normalize_data(data=data)
    #             except Exception as e:
    #                 raise ValueError(f"Error during normalization: {e}")

    #         data_seq = time_series_to_sequence(data, self.context_window)
    #         reconstructed_data = self.model.predict(data_seq)

    #         if normalization_used:
    #             reconstructed_data = self._denormalize_data(
    #                 reconstructed_data,
    #                 normalization_method=self.normalization_method,
    #                 min_x=self.min_x,
    #                 max_x=self.max_x,
    #                 mean_=self.mean_,
    #                 std_=self.std_,
    #             )

    #         padded_reconstructed = self._apply_padding(
    #             data,
    #             reconstructed_data,
    #             self.context_window,
    #             self.time_step_to_check,
    #         )

    #         reconstructed_df = (
    #             pd.DataFrame(padded_reconstructed, columns=feature_names)
    #             if feature_names
    #             else padded_reconstructed
    #         )

    #         plot_actual_and_reconstructed(
    #             actual=data_original.T,
    #             reconstructed=padded_reconstructed.T,
    #             save_path=os.path.join(self.save_path, "plots"),
    #             feature_labels=feature_names,
    #             train_split=None,
    #             val_split=None,
    #             length_datasets=None,
    #         )
    #         return reconstructed_df

    #     # Case 2: Dataset with Nans (iterative prediction)
    #     elif has_nans and iterations is not None and iterations > 0:
    #         reconstruction_records = []
    #         reconstructed_iterations[0] = np.copy(data)

    #         if normalization_used:
    #             try:
    #                 data = self._normalize_data(data=data)
    #             except Exception as e:
    #                 raise ValueError(f"Error during normalization: {e}")

    #         for iter_num in range(1, iterations):
    #             if self.imputer is not None:
    #                 data = self.imputer.apply_imputation(pd.DataFrame(data)).to_numpy()
    #             else:
    #                 data = np.nan_to_num(data, nan=0)

    #             data_seq = time_series_to_sequence(data, self.context_window)

    #             reconstructed_data = self.model.predict(data_seq)

    #             if normalization_used:
    #                 reconstructed_data = self._denormalize_data(
    #                     reconstructed_data,
    #                     normalization_method=self.normalization_method,
    #                     min_x=self.min_x,
    #                     max_x=self.max_x,
    #                     mean_=self.mean_,
    #                     std_=self.std_,
    #                 )

    #             padded_reconstructed = self._apply_padding(
    #                 data,
    #                 reconstructed_data,
    #                 self.context_window,
    #                 self.time_step_to_check,
    #             )
    #             reconstructed_iterations[iter_num] = np.copy(padded_reconstructed)

    #             for i, j in zip(*np.where(nan_positions)):
    #                 reconstruction_records.append(
    #                     {
    #                         "Column": j + 1,
    #                         "Timestep": i,
    #                         "Iteration": iter_num,
    #                         "Reconstructed value": padded_reconstructed[i, j],
    #                     }
    #                 )
    #                 if normalization_used:
    #                     data[i, j] = self._normalize_data(data=padded_reconstructed)[
    #                         i, j
    #                     ]
    #                 else:
    #                     data[i, j] = padded_reconstructed[i, j]

    #         # Last iteration
    #         if self.imputer is not None:
    #             data = self.imputer.apply_imputation(pd.DataFrame(data)).to_numpy()
    #         else:
    #             data = np.nan_to_num(data, nan=0)

    #         data_seq = time_series_to_sequence(data, self.context_window)
    #         reconstructed_data_final = self.model.predict(data_seq)

    #         if normalization_used:
    #             reconstructed_data_final = self._denormalize_data(
    #                 reconstructed_data_final,
    #                 normalization_method=self.normalization_method,
    #                 min_x=self.min_x,
    #                 max_x=self.max_x,
    #                 mean_=self.mean_,
    #                 std_=self.std_,
    #             )

    #         padded_reconstructed_final = self._apply_padding(
    #             data,
    #             reconstructed_data_final,
    #             self.context_window,
    #             self.time_step_to_check,
    #         )
    #         reconstructed_iterations[iterations] = np.copy(padded_reconstructed_final)

    #         for i in range(data.shape[0]):
    #             for j in range(data.shape[1]):
    #                 if nan_positions[i, j]:
    #                     reconstruction_records.append(
    #                         {
    #                             "Column": j + 1,
    #                             "Timestep": i,
    #                             "Iteration": iterations,
    #                             "Reconstructed value": padded_reconstructed_final[i, j],
    #                         }
    #                     )

    #         reconstructed_df = (
    #             pd.DataFrame(reconstructed_data_final, columns=feature_names)
    #             if feature_names
    #             else reconstructed_data_final
    #         )

    #         progress_df = pd.DataFrame(reconstruction_records)
    #         file_path = os.path.join(self.save_path, "reconstruction_progress.xlsx")
    #         progress_df.to_excel(file_path, index=False)

    #         plot_reconstruction_iterations(
    #             original_data=data_original.T,
    #             reconstructed_iterations={
    #                 k: v.T for k, v in reconstructed_iterations.items()
    #             },
    #             save_path=os.path.join(self.save_path, "plots"),
    #             feature_labels=feature_names,
    #         )

    #         return reconstructed_df

    def reconstruct_new_data(
        self,
        data,
        iterations: int = 1,
        id_columns: Union[str, int, List[str], List[int], None] = None,
    ):
        """
        Predict and reconstruct unknown data, iterating over NaN values to improve predictions.
        Uses stored `context_window`, normalization parameters, and the trained model.

        :param data: Input data (numpy array, pandas DataFrame, or polars DataFrame).
        :param iterations: Number of reconstruction iterations (None = no iteration).
        :param id_columns: Column(s) that define IDs to process reconstruction separately.
        :return: Dictionary with reconstructed data per ID (or "global" if no ID).
        """
        if self.model is None:
            raise ValueError(
                "No model loaded. Use `load_from_pickle()` before calling `reconstruct_new_data()`."
            )

        data, feature_names = self._convert_data_to_numpy(data)
        features_names_to_check = (
            [feature_names[i] for i in self.feature_to_check] if feature_names else None
        )

        # Handle ID columns
        if id_columns is not None:
            _, _, id_data_dict = self._handle_id_columns(data, id_columns)
        else:
            id_data_dict = {"global": data}

        reconstructed_results = {}
        if id_columns is not None:
            for id_iter, data_id in id_data_dict.items():
                nan_positions_id = np.isnan(data_id)
                has_nans_id = np.any(nan_positions_id)

                reconstructed_results[id_iter] = self._reconstruct_single_dataset(
                    data=data_id,
                    feature_names=features_names_to_check,
                    nan_positions=nan_positions_id[:, self.feature_to_check],
                    has_nans=has_nans_id,
                    iterations=iterations,
                    id_iter=id_iter,
                )
        else:
            nan_positions = np.isnan(data)
            has_nans = np.any(nan_positions)
            reconstructed_results["global"] = self._reconstruct_single_dataset(
                data=data,
                feature_names=features_names_to_check,
                nan_positions=nan_positions[:, self.feature_to_check],
                has_nans=has_nans,
                iterations=iterations,
                id_iter=None,
            )

        return reconstructed_results

    def _reconstruct_single_dataset(
        self,
        data,
        feature_names,
        nan_positions,
        has_nans,
        iterations: int = 1,
        id_iter: Optional[str] = None,
    ):
        """
        Reconstruct missing values for a single dataset (either global or for a specific ID).

        :param data: Subset of data to reconstruct (global dataset or per ID).
        :param feature_names: Feature labels.
        :param nan_positions: Boolean mask indicating NaN positions.
        :param has_nans: Boolean flag indicating if the dataset contains NaNs.
        :param iterations: Number of iterations for reconstruction.
        :param id_iter: ID of the subset being reconstructed (or None for global).
        :return: Reconstructed dataset.
        """

        data_original = np.copy(data)
        reconstructed_iterations = {}

        normalization_values = (
            self.normalization_values.get(f"{id_iter}")
            if id_iter
            else self.normalization_values.get("global")
        ) or {}
        self.min_x = normalization_values.get("min_x", None)
        self.max_x = normalization_values.get("max_x", None)
        self.mean_ = normalization_values.get("mean_", None)
        self.std_ = normalization_values.get("std_", None)

        # 1. If no Nand: simple prediction
        if not has_nans:
            if self.normalization_method:
                try:
                    data = self._normalize_data(data=data)
                except Exception as e:
                    raise ValueError(f"Error during normalization: {e}")
            data_seq = time_series_to_sequence(data, self.context_window)
            reconstructed_data = self.model.predict(data_seq)

            if self.normalization_method:
                reconstructed_data = self._denormalize_data(
                    reconstructed_data,
                    normalization_method=self.normalization_method,
                    min_x=(
                        self.min_x[self.feature_to_check]
                        if self.min_x is not None
                        else None
                    ),
                    max_x=(
                        self.max_x[self.feature_to_check]
                        if self.max_x is not None
                        else None
                    ),
                    mean_=(
                        self.mean_[self.feature_to_check]
                        if self.mean_ is not None
                        else None
                    ),
                    std_=(
                        self.std_[self.feature_to_check]
                        if self.std_ is not None
                        else None
                    ),
                )

            padded_reconstructed = self._apply_padding(
                data[:, self.feature_to_check],
                reconstructed_data,
                self.context_window,
                self.time_step_to_check,
            )

            reconstructed_df = pd.DataFrame(padded_reconstructed, columns=feature_names)

            plot_path = os.path.join(
                self.save_path if self.save_path else self.root_dir,
                "plots",
                f"{id_iter}_new_data.html" if id_iter else "global_new_data.html",
            )

            plot_actual_and_reconstructed(
                actual=data_original[:, self.feature_to_check].T,
                reconstructed=padded_reconstructed.T,
                save_path=plot_path,
                feature_labels=feature_names,
                train_split=None,
                val_split=None,
                length_datasets=None,
            )

            return reconstructed_df

        # If NaNs iterative mode
        reconstruction_records = []
        reconstructed_iterations[0] = np.copy(data[:, self.feature_to_check])

        if self.normalization_method:
            try:
                data = self._normalize_data(data=data, id_iter=id_iter)
            except Exception as e:
                raise ValueError(f"Error during normalization for ID {id_iter}: {e}")

        for iter_num in range(1, iterations):
            if self.imputer is not None:
                data = self.imputer.apply_imputation(pd.DataFrame(data)).to_numpy()
            else:
                data = np.nan_to_num(data, nan=0)

            data_seq = time_series_to_sequence(data, self.context_window)

            reconstructed_data = self.model.predict(data_seq)

            if self.normalization_method:
                reconstructed_data = self._denormalize_data(
                    reconstructed_data,
                    normalization_method=self.normalization_method,
                    min_x=(
                        self.min_x[self.feature_to_check]
                        if self.min_x is not None
                        else None
                    ),
                    max_x=(
                        self.max_x[self.feature_to_check]
                        if self.max_x is not None
                        else None
                    ),
                    mean_=(
                        self.mean_[self.feature_to_check]
                        if self.mean_ is not None
                        else None
                    ),
                    std_=(
                        self.std_[self.feature_to_check]
                        if self.std_ is not None
                        else None
                    ),
                )

            padded_reconstructed = self._apply_padding(
                data[:, self.feature_to_check],
                reconstructed_data,
                self.context_window,
                self.time_step_to_check,
            )
            reconstructed_iterations[iter_num] = np.copy(padded_reconstructed)

            for i, j in zip(*np.where(nan_positions)):
                reconstruction_records.append(
                    {
                        "ID": id_iter if id_iter else "global",
                        "Column": j + 1,
                        "Timestep": i,
                        "Iteration": iter_num,
                        "Reconstructed value": padded_reconstructed[i, j],
                    }
                )
                if self.normalization_method:
                    data[i, self.feature_to_check[j]] = self._normalize_data(
                        data=padded_reconstructed,
                        id_iter=id_iter,
                        feature_to_check_filter=True,
                    )[i, j]
                else:
                    data[i, self.feature_to_check[j]] = padded_reconstructed[i, j]

        # === 3. Última iteración: reconstrucción total ===
        if self.imputer is not None:
            data = self.imputer.apply_imputation(pd.DataFrame(data)).to_numpy()
        else:
            data = np.nan_to_num(data, nan=0)

        data_seq = time_series_to_sequence(data, self.context_window)
        reconstructed_data_final = self.model.predict(data_seq)

        if self.normalization_method:
            reconstructed_data_final = self._denormalize_data(
                reconstructed_data_final,
                normalization_method=self.normalization_method,
                min_x=(
                    self.min_x[self.feature_to_check]
                    if self.min_x is not None
                    else None
                ),
                max_x=(
                    self.max_x[self.feature_to_check]
                    if self.max_x is not None
                    else None
                ),
                mean_=(
                    self.mean_[self.feature_to_check]
                    if self.mean_ is not None
                    else None
                ),
                std_=(
                    self.std_[self.feature_to_check] if self.std_ is not None else None
                ),
            )

        padded_reconstructed_final = self._apply_padding(
            data[:, self.feature_to_check],
            reconstructed_data_final,
            self.context_window,
            self.time_step_to_check,
        )

        reconstructed_iterations[iterations] = np.copy(padded_reconstructed_final)

        for i, j in zip(*np.where(nan_positions)):
            reconstruction_records.append(
                {
                    "ID": id_iter if id_iter else "global",
                    "Column": j + 1,
                    "Timestep": i,
                    "Iteration": iterations,
                    "Reconstructed value": padded_reconstructed_final[i, j],
                }
            )

        reconstructed_df = pd.DataFrame(reconstructed_data_final, columns=feature_names)

        progress_df = pd.DataFrame(reconstruction_records)
        file_path = os.path.join(
            self.save_path if self.save_path else self.root_dir,
            "reconstruction_progress",
            f"{id_iter}_progress.xlsx" if id_iter else "global_progress.xlsx",
        )
        # Make dirs if they don't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        progress_df.to_excel(file_path, index=False)

        plot_reconstruction_iterations(
            original_data=data_original[:, self.feature_to_check].T,
            reconstructed_iterations={
                k: v.T for k, v in reconstructed_iterations.items()
            },
            save_path=os.path.join(
                self.save_path if self.save_path else self.root_dir, "plots"
            ),
            feature_labels=feature_names,
            id_iter=id_iter,
        )

        return reconstructed_df
