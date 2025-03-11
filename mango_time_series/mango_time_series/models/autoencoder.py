import logging
import os
from typing import Union, List, Tuple, Any, Optional

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from mango.logging import get_configured_logger
from mango.processing.data_imputer import DataImputer
from tensorflow.keras.models import load_model

from mango_time_series.models.modules import encoder, decoder
from mango_time_series.models.utils.plots import (
    plot_actual_and_reconstructed,
    plot_loss_history,
)
from mango_time_series.models.utils.sequences import time_series_to_sequence

logger = get_configured_logger()


class AutoEncoder:
    """
    Autoencoder model

    This Autoencoder model can be highly configurable but is already set up so
    that quick training and profiling can be done.
    """

    def __init__(
        self,
        form: str = "dense",
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
        epochs: int = 100,
        save_path: str = None,
        checkpoint: int = 10,
        use_early_stopping: bool = True,
        patience: int = 10,
        verbose: bool = False,
        feature_names: Optional[List[str]] = None,
        feature_weights: Optional[List[float]] = None,
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None,
        use_mask: bool = False,
        custom_mask: Optional[
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
        imputer: Optional[DataImputer] = None,
        train_size: Optional[float] = None,
        val_size: Optional[float] = None,
        test_size: Optional[float] = None,
        id_columns: Union[str, int, List[str], List[int], None] = None,
    ):
        """
        Initialize the Autoencoder model

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
        :param split_size: size of the split for the train (train + validation,
          validation always 10% of total data) and test datasets.
          Default value is 60% train, 10% validation and 30% test.
        :type split_size: float
        :param epochs: number of epochs to train the model
        :type epochs: int
        :param save_path: folder path to save the model checkpoints
        :type save_path: str
        :param checkpoint: number of epochs to save the model checkpoints.
        :type checkpoint: int
        :param patience: number of epochs to wait before early stopping
        :type patience: int
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
        :type train_size: Optional[float]
        :param val_size: proportion of the dataset to include in the validation set
        :type val_size: Optional[float]
        :param test_size: proportion of the dataset to include in the test set
        :type test_size: Optional[float]
        :param id_columns: optional column(s) to process the data by groups.
            If provided, the data will be grouped by this column and processed separately.
            Can be a column name (str), a column index (int), or a list of either.
        :type id_columns: Union[str, int, List[str], List[int], None]
        """

        root_dir = os.path.abspath(os.getcwd())
        self.save_path = (
            save_path if save_path else os.path.join(root_dir, "autoencoder")
        )

        self.create_folder_structure(
            [
                os.path.join(self.save_path, "models"),
                os.path.join(self.save_path, "plots"),
            ]
        )

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
        self.use_mask = use_mask
        if self.use_mask and custom_mask is not None:
            custom_mask, _ = self._convert_data_to_numpy(custom_mask)

        # Handle id_columns
        if id_columns is not None:
            if isinstance(id_columns, str) or isinstance(id_columns, int):
                id_columns = [id_columns]
            if isinstance(id_columns, list):
                if all(isinstance(i, str) for i in id_columns):
                    id_column_indices = [
                        i
                        for i, value in enumerate(extracted_feature_names)
                        if value in id_columns
                    ]
                elif all(isinstance(i, int) for i in id_columns):
                    id_column_indices = id_columns
                else:
                    raise ValueError("id_columns must be a list of strings or integers")
            else:
                raise ValueError(
                    "id_columns must be a string, integer, or a list of strings or integers"
                )
            if isinstance(data, tuple):
                self.id_data = tuple(d[:, id_column_indices] for d in data)
            else:
                self.id_data = data[:, id_column_indices]
            if isinstance(data, np.ndarray):
                data = np.delete(data, id_column_indices, axis=1)
            elif isinstance(data, tuple):
                data = tuple(np.delete(d, id_column_indices, axis=1) for d in data)

        else:
            self.id_data = None

        if self.id_data is not None:
            unique_ids, counts = np.unique(self.id_data, return_counts=True)
            min_samples_per_id = np.min(counts)

            if min_samples_per_id < context_window:
                raise ValueError(
                    f"The minimum number of samples per ID is {min_samples_per_id}, "
                    f"but the context_window is {context_window}. "
                    "Reduce the context_window or ensure each ID has enough data."
                )

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

        if isinstance(data, tuple):
            if not isinstance(custom_mask, tuple) or len(custom_mask) != 3:
                raise ValueError(
                    "If data is a tuple, custom_mask must also be a tuple of the same length."
                )
        else:
            if isinstance(custom_mask, tuple):
                raise ValueError(
                    "If data is a single array, custom_mask cannot be a tuple."
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

        contains_nans = np.isnan(data).any() if isinstance(data, np.ndarray) else False
        if contains_nans and not use_mask:
            raise ValueError(
                "Data contains NaNs, but use_mask is False. "
                "Please remove or impute NaNs before training."
            )

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # if this three sizes are none, we set the default values
        if self.train_size is None and self.val_size is None and self.test_size is None:
            self.train_size = 0.8
            self.val_size = 0.1
            self.test_size = 0.1

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

        self.x_train_original = np.copy(self.x_train)

        if self.use_mask:
            if custom_mask is not None:
                if isinstance(custom_mask, tuple):
                    if len(custom_mask) != 3:
                        raise ValueError(
                            "If custom_mask is a tuple, it must contain three arrays (train, val, test)."
                        )
                    mask_train, mask_val, mask_test = custom_mask
                    if (
                        mask_train.shape != self.x_train.shape
                        or mask_val.shape != self.x_val.shape
                        or mask_test.shape != self.x_test.shape
                    ):
                        raise ValueError(
                            "Each element of custom_mask must have the same shape as its corresponding dataset "
                            "(mask_train with x_train, mask_val with x_val, mask_test with x_test)."
                        )
                else:
                    if custom_mask.shape != data.shape:
                        raise ValueError(
                            "custom_mask must have the same shape as the original input data before transformation"
                        )
                    mask_train, mask_val, mask_test = self._time_series_split(
                        custom_mask, self.train_size, self.val_size, self.test_size
                    )
            else:
                mask_train = np.where(np.isnan(self.x_train), 0, 1)
                mask_val = np.where(np.isnan(self.x_val), 0, 1)
                mask_test = np.where(np.isnan(self.x_test), 0, 1)

            self.mask_train = time_series_to_sequence(mask_train, context_window)
            self.mask_val = time_series_to_sequence(mask_val, context_window)
            self.mask_test = time_series_to_sequence(mask_test, context_window)

            if imputer is not None:
                self.x_train = imputer.apply_imputation(self.x_train)
                self.x_val = imputer.apply_imputation(self.x_val)
                self.x_test = imputer.apply_imputation(self.x_test)
            else:
                self.x_train = np.nan_to_num(self.x_train)
                self.x_val = np.nan_to_num(self.x_val)
                self.x_test = np.nan_to_num(self.x_test)

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
            if (
                np.isnan(self.x_train).any()
                or np.isnan(self.x_val).any()
                or np.isnan(self.x_test).any()
            ):
                raise ValueError(
                    "Data contains NaNs, but use_mask is False. Please preprocess data to remove or impute NaNs."
                )

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

        self.context_window = context_window
        if isinstance(time_step_to_check, int):
            time_step_to_check = [time_step_to_check]
        self.time_step_to_check = time_step_to_check

        max_time_step = self.context_window - 1
        if any(t > max_time_step for t in self.time_step_to_check):
            raise ValueError(
                f"time_step_to_check contains invalid indices. Must be between 0 and {max_time_step}."
            )

        self.hidden_dim = hidden_dim

        self.last_epoch = 0
        self.epochs = epochs

        self.save_path = save_path
        self.checkpoint = checkpoint

        self.verbose = verbose

        self.train_loss_history = None
        self.val_loss_history = None

        self.use_early_stopping = use_early_stopping
        self.patience = patience
        self.feature_weights = feature_weights

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
        data, train_size=None, val_size=None, test_size=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits data into training, validation, and test sets according to the specified percentages.

        :param data: array-like data to split
        :type data: np.ndarray
        :param train_size: float, optional
            Proportion of the dataset to include in the training set (0-1).
        :param val_size: float, optional
            Proportion of the dataset to include in the validation set (0-1).
        :param test_size: float, optional
            Proportion of the dataset to include in the test set (0-1).
        :return: Tuple[np.ndarray, np.ndarray, np.ndarray]
            The training, validation, and test sets as numpy arrays.
        """

        if train_size is None and val_size is None and test_size is None:
            raise ValueError(
                "At least one of train_size, val_size, or test_size must be specified."
            )

        if train_size is None:
            train_size = 1.0 - (val_size or 0) - (test_size or 0)
        if val_size is None:
            val_size = 1.0 - train_size - (test_size or 0)
        if test_size is None:
            test_size = 1.0 - train_size - val_size

        total_size = train_size + val_size + test_size
        if not np.isclose(total_size, 1.0):
            raise ValueError(
                f"The sum of train_size, val_size, and test_size must be 1.0, but got {total_size}."
            )

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

    def _normalize_data(self, x_train: np.array, x_val: np.array, x_test: np.array):
        """
        Normalize the data using the specified method.
        :param x_train: training data
        :type x_train: np.array
        :param x_val: validation data
        :type x_val: np.array
        :param x_test: test data
        :type x_test: np.array
        :return: normalized data
        """
        # Normalize train for non nulls and apply the same transformation to val and test
        mask = ~np.isnan(x_train)
        if self.normalization_method == "minmax":
            self.min_x = np.min(x_train[mask], axis=0)
            self.max_x = np.max(x_train[mask], axis=0)
            range_x = self.max_x - self.min_x
            x_train = (x_train - self.min_x) / range_x
            x_val = (x_val - self.min_x) / range_x
            x_test = (x_test - self.min_x) / range_x
        elif self.normalization_method == "zscore":
            self.mean_ = np.mean(x_train[mask], axis=0)
            self.std_ = np.std(x_train[mask], axis=0)
            x_train = (x_train - self.mean_) / self.std_
            x_val = (x_val - self.mean_) / self.std_
            x_test = (x_test - self.mean_) / self.std_
        return x_train, x_val, x_test

    def prepare_datasets(
        self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        context_window: int,
        normalize: bool,
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
        :return: True if the datasets are prepared successfully
        """
        # we need to set up two functions to prepare the datasets. One when data is a
        # single numpy array and one when data is a tuple with three numpy arrays.
        if isinstance(data, np.ndarray):
            return self._prepare_numpy_dataset(data, context_window, normalize)
        elif (
            isinstance(data, tuple)
            and len(data) == 3
            and all(isinstance(i, np.ndarray) for i in data)
        ):
            return self._prepare_tuple_dataset(data, context_window, normalize)
        else:
            raise ValueError(
                "Data must be a numpy array or a tuple with three numpy arrays"
            )

    def _prepare_numpy_dataset(
        self, data: np.array, context_window: int, normalize: bool, split_size: float
    ):
        """
        Prepare the dataset for the model training and testing when the data is a single numpy array.
        :param data: numpy array with the data
        :type data: np.array
        :param context_window: context window for the model
        :type context_window: int
        :param normalize: whether to normalize the data or not
        :type normalize: bool
        :param split_size: size of the split for the train, validation and test datasets
        :type split_size: float
        :return: True if the dataset is prepared successfully
        """
        if self.id_data is not None:
            unique_ids = np.unique(self.id_data)
            train_indices, val_indices, test_indices = [], [], []

            for uid in unique_ids:
                id_idx = np.where(self.id_data == uid)[0]

                # ¿Necesario?
                # id_idx = np.sort(id_idx)

                train_cutoff = round(split_size * len(id_idx))
                val_cutoff = train_cutoff + round(0.1 * len(id_idx))

                train_indices.extend(id_idx[:train_cutoff])
                val_indices.extend(id_idx[train_cutoff:val_cutoff])
                test_indices.extend(id_idx[val_cutoff:])

            train_idx = np.array(train_indices)
            val_idx = np.array(val_indices)
            test_idx = np.array(test_indices)

            x_train, x_val, x_test = data[train_idx], data[val_idx], data[test_idx]
            id_train, id_val, id_test = (
                self.id_data[train_idx],
                self.id_data[val_idx],
                self.id_data[test_idx],
            )

        else:
            x_train, x_val, x_test = self._time_series_split(
                data, self.train_size, self.val_size, self.test_size
            )
        id_train = id_val = id_test = None
        if normalize:
            x_train, x_val, x_test = self._normalize_data(x_train, x_val, x_test)

        # We need to transform the data into a sequence of data.
        self.data = (x_train, x_val, x_test)
        self.x_train = time_series_to_sequence(
            x_train, context_window, id_data=id_train
        )
        self.x_val = time_series_to_sequence(x_val, context_window, id_data=id_val)
        self.x_test = time_series_to_sequence(x_test, context_window, id_data=id_test)

        if self.x_val.shape[0] == 0 or self.x_test.shape[0] == 0:
            raise ValueError(
                "Validation or Test sets are empty after sequence transformation. "
                "Consider reducing context_window or adjusting split_size."
            )

        # Update samples count
        self.samples = (
            self.x_train.shape[0] + self.x_val.shape[0] + self.x_test.shape[0]
        )
        return True

    def _prepare_tuple_dataset(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        context_window: int,
        normalize: bool,
    ):
        """
        Prepare the dataset for the model training and testing when the data is a tuple with three numpy arrays.
        :param data: tuple with three numpy arrays for the train, validation and test datasets
        :type data: Tuple[np.ndarray, np.ndarray, np.ndarray]
        :param context_window: context window for the model
        :type context_window: int
        :param normalize: whether to normalize the data or not
        :type normalize: bool
        :return: True if the dataset is prepared successfully
        """
        x_train, x_val, x_test = data

        if normalize:
            x_train, x_val, x_test = self._normalize_data(x_train, x_val, x_test)

        self.data = (x_train, x_val, x_test)

        self.x_train = time_series_to_sequence(x_train, context_window, id_data=self.id_data[0] if self.id_data is not None else None)
        self.x_val = time_series_to_sequence(x_val, context_window, id_data=self.id_data[1] if self.id_data is not None else None)
        self.x_test = time_series_to_sequence(x_test, context_window, id_data=self.id_data[2] if self.id_data is not None else None)

        self.samples = (
            self.x_train.shape[0] + self.x_val.shape[0] + self.x_test.shape[0]
        )

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
            y_true = tf.where(mask > 0, y_true, tf.zeros_like(y_true))
            y_pred = tf.where(mask > 0, y_pred, tf.zeros_like(y_pred))

        squared_error = tf.square(y_true - y_pred)

        # Apply feature-specific weights if provided
        if self.feature_weights is not None:
            feature_weights = tf.convert_to_tensor(
                self.feature_weights, dtype=tf.float32
            )
            squared_error = squared_error * feature_weights

        # Compute mean only over observed values if mask is provided
        if mask is not None:
            loss = tf.reduce_sum(squared_error) / tf.reduce_sum(
                mask + tf.keras.backend.epsilon()
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

    def train(self):
        """
        Train the model using the train and validation datasets and save the best model.
        """

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
                    self.save(filename="best_model.keras")
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

                self.save(filename=f"{epoch}.keras")

                # Store the loss history in the model instance
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

        # Plot loss history
        plot_loss_history(
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            save_path=os.path.join(self.save_path, "plots"),
        )

        self.save()

    def reconstruct(self):
        """
        Reconstruct the data using the trained model and plot the actual and reconstructed values.
        """
        # Calculate fitted values for each dataset
        # We use the original data for the training set to avoid shuffling in reconstruction step
        x_hat_train = self.model(self.x_train_original)
        x_hat_val = self.model(self.x_val)
        x_hat_test = self.model(self.x_test)

        # Convert to numpy arrays
        x_hat_train = x_hat_train.numpy()
        x_hat_val = x_hat_val.numpy()
        x_hat_test = x_hat_test.numpy()
        if self.normalize:
            if self.normalization_method == "minmax":
                scale_max = self.max_x[self.feature_to_check]
                scale_min = self.min_x[self.feature_to_check]
                x_hat_train = x_hat_train * (scale_max - scale_min) + scale_min
                x_hat_val = x_hat_val * (scale_max - scale_min) + scale_min
                x_hat_test = x_hat_test * (scale_max - scale_min) + scale_min
            elif self.normalization_method == "zscore":
                scale_mean = self.mean_[self.feature_to_check]
                scale_std = self.std_[self.feature_to_check]
                x_hat_train = x_hat_train * scale_std + scale_mean
                x_hat_val = x_hat_val * scale_std + scale_mean
                x_hat_test = x_hat_test * scale_std + scale_mean

        x_hat = np.concatenate((x_hat_train.T, x_hat_val.T, x_hat_test.T), axis=1)

        x_train_converted = np.copy(
            self.x_train[:, self.time_step_to_check, self.feature_to_check]
        )
        x_val_converted = np.copy(
            self.x_val[:, self.time_step_to_check, self.feature_to_check]
        )
        x_test_converted = np.copy(
            self.x_test[:, self.time_step_to_check, self.feature_to_check]
        )

        if self.normalization_method == "minmax":
            x_train_converted = x_train_converted * (scale_max - scale_min) + scale_min
            x_val_converted = x_val_converted * (scale_max - scale_min) + scale_min
            x_test_converted = x_test_converted * (scale_max - scale_min) + scale_min
        elif self.normalization_method == "zscore":
            x_train_converted = x_train_converted * scale_std + scale_mean
            x_val_converted = x_val_converted * scale_std + scale_mean
            x_test_converted = x_test_converted * scale_std + scale_mean

        x_converted = np.concatenate(
            (x_train_converted.T, x_val_converted.T, x_test_converted.T), axis=1
        )

        # Get feature labels for the selected features, if we have feature names, extract only those that correspond to feature_to_check
        feature_labels = (
            [self.features_name[i] for i in self.feature_to_check]
            if hasattr(self, "features_name")
            else None
        )

        plot_actual_and_reconstructed(
            actual=x_converted,
            reconstructed=x_hat,
            save_path=os.path.join(self.save_path, "plots"),
            feature_labels=feature_labels,
            ids=self.id_data,
        )

        return True

    def save(self, save_path: str = None, filename: str = None):
        """
        Save the model to the specified path.
        :param save_path: path to save the model
        :type save_path: str
        :param filename: name of the file to save the model
        :type filename: str
        """
        try:
            save_path = save_path or self.save_path
            filename = filename or f"{self.last_epoch}.keras"
            self.model.save(os.path.join(save_path, "models", filename))
        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            raise

    def load(self, model_path: str):
        """
        Load the model from the specified path.
        :param model_path: path to load the model
        :type model_path: str
        """
        self.model = load_model(model_path)
