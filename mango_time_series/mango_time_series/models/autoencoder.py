import numpy as np
import tensorflow as tf
from typing import Union, List
from keras import Sequential
from keras.src.optimizers import Adam

from mango_time_series.models.losses import mean_squared_error
from mango_time_series.models.modules import encoder, decoder
from mango_time_series.models.utils.plots import (
    plot_actual_and_reconstructed,
    plot_actual_and_multiple_reconstructed,
)
from mango_time_series.models.utils.sequences import (
    time_series_to_sequence,
    sequences_to_multiple_series,
)


class AutoEncoder:
    def __init__(
        self,
        form: str = "dense",
        data: np.array = None,
        context_window: int = None,
        timesteps_to_check: int = 1,
        hidden_dim: Union[int, List[int]] = None,
        num_layers: int = None,
        batch_size: int = 32,
        split_size: float = 0.7,
        epochs: int = 100,
    ):
        # First we make a copy to avoid problems
        x = np.copy(data)
        samples = x.shape[0] - context_window + 1
        features = x.shape[1]
        split = split_size

        max_x = np.max(x[:, 0], axis=0)
        min_x = np.min(x[:, 0], axis=0)
        x[:, 0] = (x[:, 0] - min_x) / (max_x - min_x)

        x = time_series_to_sequence(x, context_window)

        split_point = round(split * samples)
        x_train = x[:split_point, :, :]
        x_test = x[split_point:, :, :]

        train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
        train_dataset = train_dataset.cache().batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
        test_dataset = test_dataset.cache().batch(batch_size)

        model = Sequential(
            [
                encoder(
                    form=form,
                    context_window=context_window,
                    features=features,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    verbose=True,
                ),
                decoder(
                    form=form,
                    context_window=context_window,
                    features=features,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    verbose=True,
                ),
            ],
            name="autoencoder",
        )
        model.build()

        print(model.summary())

        model_optimizer = Adam()

        self.model = model
        self.model_optimizer = model_optimizer
        self.min_x = min_x
        self.max_x = max_x
        self.samples = samples
        self.features = features
        self.context_window = context_window
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.x = x
        self.x_train = x_train
        self.x_test = x_test
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.timesteps_to_check = timesteps_to_check

    def train(self):
        @tf.function
        def train_step(x):
            with tf.GradientTape() as autoencoder_tape:
                x = tf.cast(x, tf.float32)

                ex = self.model.get_layer("encoder_embedder")(x)
                hx = self.model.get_layer("encoder")(ex)

                x_hat = self.model.get_layer("decoder")(hx)

                x_real = tf.reshape(
                    x[:, -self.timesteps_to_check :, 0],
                    [x.shape[0], self.timesteps_to_check],
                )
                x_pred = tf.reshape(x_hat, [x.shape[0], self.timesteps_to_check])

                loss = mean_squared_error(
                    x_real,
                    x_pred,
                )

            autoencoder_gradient = autoencoder_tape.gradient(
                loss, self.model.trainable_variables
            )

            self.model_optimizer.apply_gradients(
                zip(autoencoder_gradient, self.model.trainable_variables)
            )

            return loss

        for epoch in range(self.epochs):
            for data in self.train_dataset:
                loss = train_step(data)

            if epoch % 10 == 0:
                print(f"Epoch {epoch} Loss {loss}")

    def reconstruct(self):
        x_hat_train = self.model(self.x_train)
        x_hat_train = x_hat_train.numpy()
        x_hat_train = x_hat_train * (self.max_x - self.min_x) + self.min_x

        x_hat_train = sequences_to_multiple_series(x_hat_train, self.timesteps_to_check)

        x_hat_test = self.model(self.x_test)
        x_hat_test = x_hat_test.numpy()

        x_hat_test = x_hat_test * (self.max_x - self.min_x) + self.min_x

        x_hat_test = sequences_to_multiple_series(x_hat_test, self.timesteps_to_check)

        if self.timesteps_to_check == 1:
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

        if self.timesteps_to_check == 1:
            plot_actual_and_reconstructed(
                actual=x_converted,
                reconstructed=x_hat,
                split_size=0.7,
            )
        else:
            plot_actual_and_multiple_reconstructed(
                actual=x_converted,
                reconstructed_train=x_hat_train,
                reconstructed_test=x_hat_test,
                split_size=0.7,
            )

        return x_hat_train, x_hat_test
