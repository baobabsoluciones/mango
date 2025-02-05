import logging
import os
from datetime import datetime
from typing import Union, List, Tuple

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.optimizers import Adam

from mango_time_series.models.losses import mean_squared_error
from mango_time_series.models.modules import encoder, decoder
from mango_time_series.models.utils.plots import plot_actual_and_reconstructed
from mango_time_series.models.utils.sequences import time_series_to_sequence

logger = logging.getLogger(__name__)


# TODO: Current implementation checks all the input variables for reconstruction
#  error. Maybe we want to do a subset of them for the reconstruction error.
# TODO: Current implementation gives the same weight to all input variables for
#  the reconstruction error. Maybe we want to be able to set up different weights
# TODO: Current implementation just can check one timestep for reconstruction error.
#  Maybe we want to check the last n or something like that
# TODO: implement backwards pass for RNN, LSTM and GRU.
# TODO: implement early stopping criteria


class AutoEncoder:
    """
    Autoencoder model

    This Autoencoder model can be highly configurable but is already set up so that quick training and profiling can be done.
    """

    def __init__(
        self,
        form: str = "dense",
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
        context_window: int = None,
        time_step_to_check: Union[int, List[int]] = -1,
        num_layers: int = None,
        hidden_dim: Union[int, List[int]] = None,
        normalize: bool = True,
        batch_size: int = 32,
        split_size: float = 0.7,
        epochs: int = 100,
        save_path: str = None,
        verbose: bool = False,
    ):
        """
        Initialize the Autoencoder model

        :param form: type of encoder, one of "dense", "rnn", "gru" or "lstm".
          Currently, these types of cells are both used on the encoder and
          decoder. In the future each part could have a different structure?
        :type form: str
        :param data: data to train the model. it can be a single numpy array
          with the whole dataset from which a train, validation and test split
          is going to be set up, or a tuple with three numpy arrays, one for
          the train, one for the validation and one for the test.
        :type data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        :param context_window: context window for the model. This is used
          to transform the tabular data into a sequence of data
          (from 2D tensor to 3D tensor)
        :type context_window: int
        :param time_step_to_check: time steps to check for the autoencoder.
          Currently only int value is supported and it should be the index
          of the context window to check. In the future this could be a list of
          indices to check. For taking only the last timestep of the context
          window this should be set to -1.
        :type timesteps_to_check: Union[int, List[int]]
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
        :param normalize: whether to normalize the data or not.
        :type normalize: bool
        :param batch_size: batch size for the model
        :type batch_size: int
        :param split_size: size of the split for the train (train + validation,
          validation always 10% of total data) and test datasets.
          Default value is 60% train, 10% validation and 30% test.
        :type split_size: float
        :param epochs: number of epochs to train the model
        :type epochs: int
        """
        # First we check if hidden dim is a list it has a number of elements equal to num_layers
        if isinstance(hidden_dim, list):
            if len(hidden_dim) != num_layers:
                raise ValueError(
                    "hidden_dim must have a number of elements equal to num_layers"
                )

        # If hidden dim is not a list, we check if it is an integer and if it is greater than 0
        elif isinstance(hidden_dim, int):
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be greater than 0")
            else:
                hidden_dim = [hidden_dim] * num_layers

        # If hidden dim is not a list or an integer, we raise an error
        else:
            raise ValueError("hidden_dim must be a list of integers or an integer")

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

        self.prepare_datasets(data, context_window, normalize, split_size)

        train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        train_dataset = train_dataset.cache().batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices(self.x_val)
        val_dataset = val_dataset.cache().batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(self.x_test)
        test_dataset = test_dataset.cache().batch(batch_size)

        model = Sequential(
            [
                encoder(
                    form=form,
                    context_window=context_window,
                    features=self.features,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    verbose=verbose,
                ),
                decoder(
                    form=form,
                    context_window=context_window,
                    features=self.features,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    verbose=verbose,
                ),
            ],
            name="autoencoder",
        )
        model.build()

        if verbose:
            logger.info(f"The model has the following structure: {model.summary()}")

        model_optimizer = Adam()

        self.form = form
        self.model = model
        self.model_optimizer = model_optimizer

        self.context_window = context_window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.last_epoch = 0
        self.epochs = epochs

        self.time_step_to_check = time_step_to_check

        self.save_path = save_path

        self.verbose = verbose

    def prepare_datasets(
        self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        context_window: int,
        normalize: bool,
        split_size: float,
    ):
        # we need to set up two functions to prepare the datasets. One when data is a single numpy array and one when data is a tuple with three numpy arrays.
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
        # If normalize is True, we need to normalize the data with min and max values so all data lays between 0 and 1.
        # We need to normalize all the data, not only the fist column
        if normalize:
            self.max_x = np.max(data, axis=0)
            self.min_x = np.min(data, axis=0)
            data = (data - self.min_x) / (self.max_x - self.min_x)

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

        self.features = self.x_train.shape[2]

        return True

    def _prepare_tuple_dataset(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        context_window: int,
        normalize: bool,
    ):
        if normalize:
            train = data[0].shape[0]
            val = data[1].shape[0]

            data = np.concatenate((data[0], data[1], data[2]), axis=0)

            self.max_x = np.max(data, axis=0)
            self.min_x = np.min(data, axis=0)
            data = (data - self.min_x) / (self.max_x - self.min_x)

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
        self.features = self.x_train.shape[1]

        return True

    def train(self):
        @tf.function
        def train_step(x):
            with tf.GradientTape() as autoencoder_tape:
                x = tf.cast(x, tf.float32)

                hx = self.model.get_layer(f"{self.form}_encoder")(x)
                x_hat = self.model.get_layer(f"{self.form}_decoder")(hx)

                x_real = x[:, self.time_step_to_check, :]

                x_pred = tf.reshape(
                    x_hat, [x.shape[0], self.time_step_to_check, self.features]
                )

                loss = mean_squared_error(x_real, x_pred)

            autoencoder_gradient = autoencoder_tape.gradient(
                loss, self.model.trainable_variables
            )

            self.model_optimizer.apply_gradients(
                zip(autoencoder_gradient, self.model.trainable_variables)
            )

            return loss

        @tf.function
        def validation_step(x):
            x = tf.cast(x, tf.float32)

            hx = self.model.get_layer(f"{self.form}_encoder")(x)
            x_hat = self.model.get_layer(f"{self.form}_decoder")(hx)

            x_real = x[:, self.time_step_to_check :, :]

            x_pred = tf.reshape(
                x_hat, [x.shape[0], self.time_step_to_check, self.features]
            )

            loss = mean_squared_error(x_real, x_pred)
            return loss

        # Lists to store loss history
        train_loss_history = []
        val_loss_history = []

        for epoch in range(self.epochs):
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

            if epoch % 10 == 0:
                if self.verbose:
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"Training Loss: {avg_train_loss:.6f} | "
                        f"Validation Loss: {avg_val_loss:.6f}"
                    )

                self.save()

        # Store the loss history in the model instance
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history
        self.save()

    def reconstruct(self):
        x_hat_train = self.model(self.x_train)
        x_hat_train = x_hat_train.numpy()
        x_hat_train = x_hat_train * (self.max_x - self.min_x) + self.min_x

        # TODO: review this
        x_hat_train = sequences_to_multiple_series(x_hat_train, self.time_step_to_check)

        x_hat_test = self.model(self.x_test)
        x_hat_test = x_hat_test.numpy()

        x_hat_test = x_hat_test * (self.max_x - self.min_x) + self.min_x

        # TODO: review this
        x_hat_test = sequences_to_multiple_series(x_hat_test, self.time_step_to_check)

        # TODO: review this
        x_hat = np.concatenate((x_hat_train[0], x_hat_test[0]), axis=1).reshape(
            (1, self.samples)
        )

        x_train_converted = np.copy(self.x_train[:, -1, 0])
        x_train_converted = x_train_converted * (self.max_x - self.min_x) + self.min_x

        x_test_converted = np.copy(self.x_test[:, -1, 0])
        x_test_converted = x_test_converted * (self.max_x - self.min_x) + self.min_x

        x_converted = np.concatenate(
            (x_train_converted, x_test_converted), axis=0
        ).reshape((1, self.samples))

        plot_actual_and_reconstructed(
            actual=x_converted,
            reconstructed=x_hat,
            split_size=0.7,
        )

        return x_hat_train, x_hat_test

    def save(self, save_path: str = None):
        if save_path is None:
            save_path = self.save_path

        self.model.save(os.path.join(save_path, f"{self.last_epoch}.keras"))
