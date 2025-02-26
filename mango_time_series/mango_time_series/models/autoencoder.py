import logging
import os
from typing import Union, List, Tuple, Any, Optional

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.keras.models import load_model

from mango_time_series.models.losses import mean_squared_error
from mango_time_series.models.modules import encoder, decoder
from mango_time_series.models.utils.plots import (
    plot_actual_and_reconstructed,
    plot_loss_history,
)
from mango_time_series.models.utils.sequences import time_series_to_sequence

logger = logging.getLogger(__name__)


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
        num_layers: int = None,
        hidden_dim: Union[int, List[int]] = None,
        bidirectional_encoder: bool = False,
        bidirectional_decoder: bool = False,
        activation_encoder: str = None,
        activation_decoder: str = None,
        normalize: bool = True,
        normalization_method: str = "minmax",
        optimizer: str = "adam",
        batch_size: int = 32,
        split_size: float = 0.7,
        epochs: int = 100,
        save_path: str = None,
        checkpoint: int = 10,
        use_early_stopping: bool = True,
        patience: int = 10,
        verbose: bool = False,
        feature_names: Optional[List[str]] = None,
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
        :param num_layers: number of internal layers. This number is used to
          know the number of encoding layers after the input (the result of
          the last layer is going to be the embedding of the autoencoder) and
          the number of decoding layers before the output (the result of the
          last layer is going to be the reconstructed data).
        :type num_layers: int
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

        # First we check if hidden dim is a list it has a number of elements
        # equal to num_layers
        if isinstance(hidden_dim, list):
            if len(hidden_dim) != num_layers:
                raise ValueError(
                    "hidden_dim must have a number of elements equal to num_layers"
                )

        # If hidden dim is not a list, we check if it is an integer and if it is greater
        #  than 0
        elif isinstance(hidden_dim, int):
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be greater than 0")
            else:
                hidden_dim = [hidden_dim] * num_layers

        # If hidden dim is not a list or an integer, we raise an error
        else:
            raise ValueError("hidden_dim must be a list of integers or an integer")

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
        self.prepare_datasets(data, context_window, normalize, split_size)

        if isinstance(feature_to_check, int):
            feature_to_check = [feature_to_check]
        self.feature_to_check = feature_to_check

        self.input_features = self.x_train.shape[2]
        self.output_features = len(self.feature_to_check)

        train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        train_dataset = train_dataset.cache().batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(self.x_val)
        val_dataset = val_dataset.cache().batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(self.x_test)
        test_dataset = test_dataset.cache().batch(batch_size)

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

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.split_size = split_size

        self.last_epoch = 0
        self.epochs = epochs

        self.save_path = save_path
        self.checkpoint = checkpoint

        self.verbose = verbose

        self.train_loss_history = None
        self.val_loss_history = None

        self.use_early_stopping = use_early_stopping
        self.patience = patience

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

    def prepare_datasets(
        self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        context_window: int,
        normalize: bool,
        split_size: float,
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
        :param split_size: size of the split for the train, validation and test datasets
        :type split_size: float
        :return: True if the datasets are prepared successfully
        """
        # we need to set up two functions to prepare the datasets. One when data is a
        # single numpy array and one when data is a tuple with three numpy arrays.
        if isinstance(data, np.ndarray):
            return self._prepare_numpy_dataset(
                data, context_window, normalize, split_size
            )
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
        # If normalize is True, we need to normalize the data before splitting it into train, validation and test datasets and transforming it into a sequence of data.
        # Normalization can be done using minmax or zscore methods.
        # minmax method scales the data between 0 and 1, while zscore method scales the data to have a mean of 0 and a standard deviation of 1. We store the min and max values of the data for later use.
        if normalize:
            if self.normalization_method == "minmax":
                self.max_x = np.max(data, axis=0)
                self.min_x = np.min(data, axis=0)
                data = (data - self.min_x) / (self.max_x - self.min_x)

            elif self.normalization_method == "zscore":
                self.mean_ = np.mean(data, axis=0)
                # Avoid division by zero
                self.std_ = np.std(data, axis=0) + 1e-8
                data = (data - self.mean_) / self.std_

        # We need to transform the data into a sequence of data.
        self.data = np.copy(data)
        temp_data = time_series_to_sequence(self.data, context_window)

        # We need to split the data into train, validation and test datasets.
        # Validation should be 10% of the total data,
        # so train is split_size - 10% and test is 100% - split_size
        self.samples = temp_data.shape[0]

        train_split_point = round((split_size - 0.1) * self.samples)
        val_split_point = train_split_point + round(0.1 * self.samples)

        self.x_train = temp_data[:train_split_point, :, :]
        self.x_val = temp_data[train_split_point:val_split_point, :, :]
        self.x_test = temp_data[val_split_point:, :, :]

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
        if normalize:
            train = data[0].shape[0]
            val = data[1].shape[0]

            data = np.concatenate((data[0], data[1], data[2]), axis=0)

            if self.normalization_method == "minmax":
                self.max_x = np.max(data, axis=0)
                self.min_x = np.min(data, axis=0)
                data = (data - self.min_x) / (self.max_x - self.min_x)
            elif self.normalization_method == "zscore":
                self.mean_ = np.mean(data, axis=0)
                self.std_ = np.std(data, axis=0) + 1e-8
                data = (data - self.mean_) / self.std_

            data_train = data[:train, :]
            data_val = data[train : train + val, :]
            data_test = data[train + val :, :]

            data = tuple([data_train, data_val, data_test])

        self.data = data

        self.x_train = time_series_to_sequence(data[0], context_window)
        self.x_val = time_series_to_sequence(data[1], context_window)
        self.x_test = time_series_to_sequence(data[2], context_window)

        self.samples = (
            self.x_train.shape[0] + self.x_val.shape[0] + self.x_test.shape[0]
        )

        return True

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
        def train_step(x):
            """
            Training step for the model.
            :param x: input data
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
                train_loss = mean_squared_error(x_real, x_pred)

            autoencoder_gradient = autoencoder_tape.gradient(
                train_loss, self.model.trainable_variables
            )

            self.model_optimizer.apply_gradients(
                zip(autoencoder_gradient, self.model.trainable_variables)
            )

            return train_loss

        @tf.function
        def validation_step(x):
            """
            Validation step for the model.
            :param x: input data
            """
            x = tf.cast(x, tf.float32)

            hx = self.model.get_layer(f"{self.form}_encoder")(x)
            x_hat = self.model.get_layer(f"{self.form}_decoder")(hx)

            # Gather all required time steps
            x_real = tf.gather(x, self.time_step_to_check, axis=1)
            x_real = tf.gather(x_real, self.feature_to_check, axis=2)

            x_pred = tf.expand_dims(x_hat, axis=1)
            # Calculate mean loss across all selected points
            val_loss = mean_squared_error(x_real, x_pred)

            return val_loss

        # Lists to store loss history
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            # Training loop
            epoch_train_losses = []
            for data in self.train_dataset:
                loss = train_step(data)
                epoch_train_losses.append(float(loss))

            # Calculate average training loss for the epoch
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_loss_history.append(avg_train_loss)

            # Validation loop
            epoch_val_losses = []
            for data in self.val_dataset:
                val_loss = validation_step(data)
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
        x_hat_train = self.model(self.x_train)
        x_hat_val = self.model(self.x_val)
        x_hat_test = self.model(self.x_test)

        # Convert to numpy arrays
        x_hat_train = x_hat_train.numpy()
        x_hat_val = x_hat_val.numpy()
        x_hat_test = x_hat_test.numpy()

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

        # Get feature labels for the selected features
        if hasattr(self, "features_name") and self.features_name:
            # If we have feature names, extract only those that correspond to feature_to_check
            feature_labels = [self.features_name[i] for i in self.feature_to_check]
        else:
            feature_labels = None

        plot_actual_and_reconstructed(
            actual=x_converted,
            reconstructed=x_hat,
            save_path=os.path.join(self.save_path, "plots"),
            feature_labels=feature_labels,
            split_size=self.split_size,
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
