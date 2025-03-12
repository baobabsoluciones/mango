import os
import shutil
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import Bidirectional
from mango.processing.data_imputer import DataImputer

from mango_time_series.models.autoencoder import AutoEncoder
from mango_time_series.models.modules import encoder, decoder
from mango_time_series.models.utils.sequences import time_series_to_sequence


class TestEncoder(unittest.TestCase):
    def test_lstm(self):
        """
        Test LSTM encoder creation and output shape
        """
        model = encoder(
            form="lstm",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 LSTM + Dense
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(model.name, "lstm_encoder")

        # Test output shape
        batch_size = 16
        # (batch, context_window, features)
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, context_window, hidden_dim)
        self.assertEqual(output.shape, (batch_size, 10, 32))

    def test_gru(self):
        """
        Test GRU encoder creation and output shape
        """
        model = encoder(
            form="gru",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 GRU + Dense
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(model.name, "gru_encoder")

        # Test output shape
        batch_size = 16
        # (batch, context_window, features)
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, context_window, hidden_dim)
        self.assertEqual(output.shape, (batch_size, 10, 32))

    def test_rnn(self):
        """
        Test RNN encoder creation and output shape
        """
        model = encoder(
            form="rnn",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 RNN + Dense
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(model.name, "rnn_encoder")

        # Test output shape
        batch_size = 16
        # (batch, context_window, features)
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, context_window, hidden_dim)
        self.assertEqual(output.shape, (batch_size, 10, 32))

    def test_dense(self):
        """
        Test Dense encoder creation and output shape
        """
        model = encoder(
            form="dense",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        self.assertEqual(len(model.layers), 3)  # Input + 2 Dense
        self.assertEqual(model.name, "dense_encoder")

        # Test output shape
        batch_size = 16
        # (batch, features)
        input_shape = (batch_size, 5)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, hidden_dim)
        self.assertEqual(output.shape, (batch_size, 32))

    def test_invalid_type(self):
        """
        Test encoder with invalid type
        """
        with self.assertRaisesRegex(ValueError, "Invalid encoder type: invalid"):
            encoder(form="invalid", features=5, hidden_dim=32, num_layers=2)

    def test_lstm_variable_hidden_dims(self):
        """
        Test LSTM encoder with different hidden dimensions per layer
        """
        hidden_dims = [32, 16]
        model = encoder(
            form="lstm",
            context_window=10,
            features=5,
            hidden_dim=hidden_dims,
            num_layers=2,
        )

        batch_size = 16
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # Should match last hidden dim
        self.assertEqual(output.shape, (batch_size, 10, 16))

    def test_bidirectional_encoder(self):
        """
        Test encoder with bidirectional LSTM
        """
        hidden_dims = [32, 16]
        model = encoder(
            form="lstm",
            context_window=10,
            features=5,
            hidden_dim=hidden_dims,
            num_layers=2,
            use_bidirectional=True,
        )

        encoder_has_bidirectional = any(
            isinstance(layer, Bidirectional) for layer in model.layers
        )
        self.assertTrue(
            encoder_has_bidirectional, "Bidirectional layer missing in encoder."
        )

        # Verify shape of bidirectional layer
        batch_size = 16
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)

        model_layers = tf.keras.Model(
            inputs=model.input, outputs=[layer.output for layer in model.layers]
        )
        intermediate_outputs = model_layers(test_input)

        bidirectional_index = 0
        # Verify that the bidirectional layer has the correct shape
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Bidirectional):
                output_shape = intermediate_outputs[i].shape
                expected_shape = (batch_size, 10, hidden_dims[bidirectional_index] * 2)

                bidirectional_index += 1
                self.assertEqual(
                    output_shape,
                    expected_shape,
                    f"Bidirectional layer {i} has incorrect shape {output_shape}, expected {expected_shape}",
                )

        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, 10, hidden_dims[-1]))


class TestDecoder(unittest.TestCase):
    def test_lstm(self):
        """
        Test LSTM decoder creation and output shape
        """
        model = decoder(
            form="lstm",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 LSTM + Dense + Dense
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.name, "lstm_decoder")

        # Test output shape
        batch_size = 16
        # (batch, context_window, hidden_dim)
        input_shape = (batch_size, 10, 32)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, features)
        self.assertEqual(output.shape, (batch_size, 5))

    def test_gru(self):
        """
        Test GRU decoder creation and output shape
        """
        model = decoder(
            form="gru",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 GRU + Dense + Dense
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.name, "gru_decoder")

        # Test output shape
        batch_size = 16
        # (batch, context_window, hidden_dim)
        input_shape = (batch_size, 10, 32)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, features)
        self.assertEqual(output.shape, (batch_size, 5))

    def test_rnn(self):
        """
        Test RNN decoder creation and output shape
        """
        model = decoder(
            form="rnn",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 RNN + Dense + Dense
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.name, "rnn_decoder")

        # Test output shape
        batch_size = 16
        # (batch, context_window, hidden_dim)
        input_shape = (batch_size, 10, 32)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, features)
        self.assertEqual(output.shape, (batch_size, 5))

    def test_dense(self):
        """
        Test Dense decoder creation and output shape
        """
        model = decoder(
            form="dense",
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 Dense + Output Dense
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(model.name, "dense_decoder")

        # Test output shape
        batch_size = 16
        # (batch, hidden_dim)
        input_shape = (batch_size, 32)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        # (batch, features)
        self.assertEqual(output.shape, (batch_size, 5))

    def test_invalid_type(self):
        """
        Test decoder with invalid type
        """
        with self.assertRaisesRegex(ValueError, "Invalid decoder type: invalid"):
            decoder(form="invalid", features=5, hidden_dim=32, num_layers=2)

    def test_lstm_variable_hidden_dims(self):
        """
        Test LSTM decoder with different hidden dimensions per layer
        """
        hidden_dims = [32, 16]
        model = decoder(
            form="lstm",
            context_window=10,
            features=5,
            hidden_dim=hidden_dims,
            num_layers=2,
        )

        batch_size = 16
        # First hidden dim must match input
        input_shape = (batch_size, 10, 16)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, 5))

    def test_bidirectional_decoder(self):
        """
        Test that bidirectional is applied correctly to the decoder
        """
        hidden_dims = [32, 16]
        model = decoder(
            form="lstm",
            context_window=10,
            features=5,
            hidden_dim=hidden_dims,
            num_layers=2,
            use_bidirectional=True,
        )

        decoder_has_bidirectional = any(
            isinstance(layer, Bidirectional) for layer in model.layers
        )
        self.assertTrue(
            decoder_has_bidirectional, "Bidirectional layer missing in decoder."
        )

        # Verify that the output shape is correct
        batch_size = 16
        input_shape = (batch_size, 10, 16)
        test_input = tf.random.normal(input_shape)

        model_layers = tf.keras.Model(
            inputs=model.input, outputs=[layer.output for layer in model.layers]
        )
        intermediate_outputs = model_layers(test_input)

        hidden_dims = hidden_dims[::-1]
        bidirectional_index = 0
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Bidirectional):
                output_shape = intermediate_outputs[i].shape
                expected_shape = (batch_size, 10, hidden_dims[bidirectional_index] * 2)
                bidirectional_index += 1
                self.assertEqual(
                    output_shape,
                    expected_shape,
                    f"Bidirectional layer {i} has incorrect shape {output_shape}, expected {expected_shape}",
                )

        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, 5))


class TestAutoEncoder(unittest.TestCase):
    def test_bidirectional_autoencoder(self):
        """
        Test full autoencoder with bidirectional LSTM in encoder and decoder
        """
        hidden_dims = [32, 16]

        model = AutoEncoder(
            form="lstm",
            data=np.random.rand(100, 5),
            context_window=10,
            hidden_dim=hidden_dims,
            bidirectional_encoder=True,
            bidirectional_decoder=True,
            epochs=1,
        )

        encoder = model.model.get_layer("lstm_encoder")
        decoder = model.model.get_layer("lstm_decoder")

        encoder_has_bidirectional = any(
            isinstance(layer, Bidirectional) for layer in encoder.layers
        )
        decoder_has_bidirectional = any(
            isinstance(layer, Bidirectional) for layer in decoder.layers
        )

        self.assertTrue(
            encoder_has_bidirectional, "Bidirectional layer missing in encoder."
        )
        self.assertTrue(
            decoder_has_bidirectional, "Bidirectional layer missing in decoder."
        )

        batch_size = 16
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)
        output = model.model(test_input)
        self.assertEqual(output.shape, (batch_size, len(model.feature_to_check)))

    def test_bidirectional_not_allowed_for_dense(self):
        """
        Ensure AutoEncoder raises an error when bidirectional is used with 'dense'
        """

        data = np.random.rand(100, 5)
        with self.assertRaises(ValueError) as context:
            AutoEncoder(
                form="dense",
                data=data,
                context_window=10,
                hidden_dim=[32, 16],
                bidirectional_encoder=True,
            )
        self.assertIn(
            "Bidirectional is not supported for encoder type 'dense'",
            str(context.exception),
        )

        with self.assertRaises(ValueError) as context:
            AutoEncoder(
                form="dense",
                data=data,
                context_window=10,
                hidden_dim=[32, 16],
                bidirectional_decoder=True,
            )
        self.assertIn(
            "Bidirectional is not supported for decoder type 'dense'",
            str(context.exception),
        )

        with self.assertRaises(ValueError) as context:
            AutoEncoder(
                form="dense",
                data=data,
                context_window=10,
                hidden_dim=[32, 16],
                bidirectional_encoder=True,
                bidirectional_decoder=True,
            )
        self.assertIn(
            "Bidirectional is not supported for encoder and decoder type 'dense'",
            str(context.exception),
        )


class TestAutoEncoderLoss(unittest.TestCase):
    def setUp(self):
        """
        Generate synthetic sensor data for testing.
        """
        np.random.seed(42)
        self.samples = 500
        self.features = 4

        self.data = self._generate_random_data()
        self.mask = self._generate_binary_mask(self.data)

        self.data_no_nans = self._impute_nans(self.data)

        self.reconstructed_data = self._generate_reconstructed_data(self.data)

        self.feature_weights = [1.0, 0.5, 1.2, 0.8]

        self.data_df, self.mask_df = self._generate_dataframe_with_mask()

    def tearDown(self):
        """
        Remove any created folders after the tests.
        """
        save_path = os.path.abspath(os.path.join(os.getcwd(), "autoencoder"))
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

    def _generate_random_data(self):
        """Generates random data."""
        return np.random.rand(self.samples, self.features) * 10

    @staticmethod
    def _generate_binary_mask(data):
        """Generates a binary mask with 20% missing values."""
        mask = np.ones_like(data)
        missing_entries = np.random.choice([0, 1], size=data.shape, p=[0.2, 0.8])
        data[missing_entries == 0] = np.nan
        mask[missing_entries == 0] = 0
        return mask

    @staticmethod
    def _impute_nans(data):
        """Imputes NaNs with the mean of each feature."""
        return np.where(np.isnan(data), np.nanmean(data, axis=0), data)

    @staticmethod
    def _generate_reconstructed_data(data):
        """Simulates reconstructed data with added noise."""
        return np.nan_to_num(data) + np.random.normal(0, 0.5, data.shape)

    def _generate_dataframe_with_mask(self):
        """Generates a random DataFrame and a corresponding mask."""
        df = pd.DataFrame(
            np.random.rand(self.samples, self.features) * 10,
            columns=[f"feature_{i}" for i in range(self.features)],
        )
        df[self.mask == 0] = np.nan
        mask_df = df.notna().astype(float)
        mask_df[df > df.quantile(0.75)] = 0.5
        return df, mask_df

    def test_loss_no_mask_no_weights(self):
        """
        Test loss calculation without mask and without feature weights.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data_no_nans,
            context_window=10,
            hidden_dim=[32, 16],
            epochs=1,
            use_mask=False,
        )

        loss = autoencoder.masked_weighted_mse(
            self.data_no_nans, self.reconstructed_data
        )
        self.assertGreater(loss.numpy(), 0, "Loss should be positive")

    def test_loss_weights_no_mask(self):
        """
        Test loss calculation with feature weights but without mask.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data_no_nans,
            context_window=10,
            hidden_dim=[32, 16],
            feature_weights=self.feature_weights,
            epochs=1,
            use_mask=False,
        )

        loss = autoencoder.masked_weighted_mse(
            self.data_no_nans, self.reconstructed_data
        )
        self.assertGreater(loss.numpy(), 0, "Loss should be positive")

    def test_loss_mask_no_weights(self):
        """
        Test loss calculation with mask but without feature weights.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data,
            context_window=10,
            hidden_dim=[32, 16],
            epochs=1,
            use_mask=True,
        )

        loss = autoencoder.masked_weighted_mse(
            self.data, self.reconstructed_data, mask=self.mask
        )
        self.assertGreater(loss.numpy(), 0, "Loss should be positive")

    def test_loss_mask_and_weights(self):
        """
        Test loss calculation with both mask and feature weights.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data,
            context_window=10,
            hidden_dim=[32, 16],
            feature_weights=self.feature_weights,
            epochs=1,
            use_mask=True,
        )

        loss = autoencoder.masked_weighted_mse(
            self.data, self.reconstructed_data, mask=self.mask
        )
        self.assertGreater(loss.numpy(), 0, "Loss should be positive")

    def test_raise_error_nan_no_mask(self):
        """
        Test that an error is raised when NaNs are present and no mask is used.
        """
        with self.assertRaises(ValueError) as context:
            AutoEncoder(
                form="lstm",
                data=self.data,
                context_window=10,
                hidden_dim=[32, 16],
                epochs=1,
                use_mask=False,
            )
        self.assertIn(
            "Data contains NaNs, but use_mask is False. Please remove or impute NaNs before training.",
            str(context.exception),
        )

    def test_custom_mask_application(self):
        """
        Test that a custom mask is correctly applied.
        """
        custom_mask = np.ones_like(self.data)
        custom_mask[np.isnan(self.data)] = 0
        custom_mask[self.data > np.nanpercentile(self.data, 75)] = 0.5
        context_window = 10

        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data,
            context_window=context_window,
            hidden_dim=[32, 16],
            epochs=1,
            normalize=True,
            use_mask=True,
            custom_mask=custom_mask,
        )

        # Ensure mask is correctly stored
        expected_mask_train, _, _ = autoencoder._time_series_split(
            custom_mask,
            autoencoder.train_size,
            autoencoder.val_size,
            autoencoder.test_size,
        )
        expected_mask_train = time_series_to_sequence(
            expected_mask_train, context_window
        )
        np.testing.assert_array_equal(autoencoder.mask_train, expected_mask_train)

    def test_custom_mask_wrong_shape(self):
        """
        Test that providing a custom mask with incorrect shape raises an error.
        """
        custom_mask_wrong_shape = np.random.choice([0, 1], size=(50, 4), p=[0.2, 0.8])

        with self.assertRaises(ValueError) as context:
            AutoEncoder(
                form="lstm",
                data=self.data,
                context_window=10,
                hidden_dim=[32, 16],
                epochs=1,
                use_mask=True,
                custom_mask=custom_mask_wrong_shape,
            )
        self.assertIn(
            "custom_mask must have the same shape as the original input data before transformation",
            str(context.exception),
        )

    def test_custom_mask_as_tuple(self):
        """
        Test that custom mask provided as a tuple is applied correctly.
        """
        context_window = 10
        custom_mask_train = np.random.randint(0, 2, size=(400, self.features))
        custom_mask_val = np.random.randint(0, 2, size=(50, self.features))
        custom_mask_test = np.random.randint(0, 2, size=(50, self.features))

        autoencoder = AutoEncoder(
            form="lstm",
            data=(self.data[:400], self.data[400:450], self.data[450:]),
            context_window=context_window,
            hidden_dim=[32, 16],
            epochs=1,
            normalize=True,
            use_mask=True,
            custom_mask=(custom_mask_train, custom_mask_val, custom_mask_test),
        )

        # Ensure masks are correctly assigned
        np.testing.assert_array_equal(
            autoencoder.mask_train,
            time_series_to_sequence(custom_mask_train, context_window),
        )
        np.testing.assert_array_equal(
            autoencoder.mask_val,
            time_series_to_sequence(custom_mask_val, context_window),
        )
        np.testing.assert_array_equal(
            autoencoder.mask_test,
            time_series_to_sequence(custom_mask_test, context_window),
        )

    def test_data_split_default(self):
        """
        Test that when no split sizes are provided, the default 80-10-10 split is applied.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data,
            context_window=10,
            hidden_dim=[32, 16],
            epochs=1,
            use_mask=True,
            normalize=False,
        )

        train_size = int(0.8 * self.samples)
        val_size = int(0.1 * self.samples)
        test_size = self.samples - train_size - val_size

        self.assertEqual(autoencoder.data[0].shape[0], train_size)
        self.assertEqual(autoencoder.data[1].shape[0], val_size)
        self.assertEqual(autoencoder.data[2].shape[0], test_size)

    def test_data_split_with_two_sizes(self):
        """
        Test that if only two sizes are provided, the third is inferred correctly.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data,
            context_window=10,
            hidden_dim=[32, 16],
            epochs=1,
            use_mask=True,
            normalize=True,
            train_size=0.7,
            val_size=0.2,  # Test size should be inferred as 0.1
        )

        train_size = int(0.7 * self.samples)
        val_size = int(0.2 * self.samples)
        test_size = self.samples - train_size - val_size

        self.assertEqual(autoencoder.data[0].shape[0], train_size)
        self.assertEqual(autoencoder.data[1].shape[0], val_size)
        self.assertEqual(autoencoder.data[2].shape[0], test_size)

    def test_normalization_with_nans_and_mask(self):
        """
        Test that normalization does not consider NaNs when use_mask=True and custom_mask=None.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data,
            context_window=10,
            hidden_dim=[32, 16],
            epochs=1,
            normalize=True,
            use_mask=True,
            custom_mask=None,
        )
        self.assertFalse(
            np.isnan(autoencoder.x_train).any(), "Normalized data contains NaNs."
        )

    def test_dataframe_input_with_custom_mask(self):
        """
        Test that AutoEncoder correctly processes a pandas DataFrame for both data and custom_mask.
        """
        context_window = 10

        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data_df,
            context_window=context_window,
            hidden_dim=[32, 16],
            epochs=1,
            normalize=True,
            use_mask=True,
            custom_mask=self.mask_df,
        )
        expected_mask_train, _, _ = autoencoder._time_series_split(
            self.mask_df.to_numpy(),
            autoencoder.train_size,
            autoencoder.val_size,
            autoencoder.test_size,
        )

        expected_mask_train = time_series_to_sequence(
            expected_mask_train, context_window
        )

        np.testing.assert_array_equal(autoencoder.mask_train, expected_mask_train)

        self.assertIsInstance(autoencoder.x_train, np.ndarray)
        self.assertIsInstance(autoencoder.mask_train, np.ndarray)

    def test_imputer_usage(self):
        """
        Test that the imputer is used when provided.
        """
        imputer = DataImputer(strategy="mean")

        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data,
            context_window=10,
            hidden_dim=[32, 16],
            epochs=1,
            use_mask=True,
            imputer=imputer,
        )

        self.assertFalse(np.isnan(autoencoder.x_train).any())

    def test_shuffle_behavior(self):
        """
        Test that shuffling behaves as expected.
        """
        autoencoder = AutoEncoder(
            form="lstm",
            data=self.data_no_nans,
            context_window=10,
            hidden_dim=[32, 16],
            epochs=1,
            use_mask=False,
            shuffle=True,
            shuffle_buffer_size=50,
        )

        self.assertEqual(autoencoder.shuffle_buffer_size, 50)

    def test_shuffle_invalid_buffer_size(self):
        """
        Test that an invalid shuffle buffer size raises an error.
        """
        with self.assertRaises(ValueError) as context:
            AutoEncoder(
                form="lstm",
                data=self.data_no_nans,
                context_window=10,
                hidden_dim=[32, 16],
                epochs=1,
                use_mask=False,
                shuffle=True,
                shuffle_buffer_size=-10,
            )
        self.assertIn(
            "shuffle_buffer_size must be a positive integer", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
