import unittest

import numpy as np
import tensorflow as tf
from keras.src.layers import Bidirectional

from mango_time_series.models.autoencoder import AutoEncoder
from mango_time_series.models.modules import encoder, decoder


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
            num_layers=2,
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
                num_layers=2,
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
                num_layers=2,
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
                num_layers=2,
                hidden_dim=[32, 16],
                bidirectional_encoder=True,
                bidirectional_decoder=True,
            )
        self.assertIn(
            "Bidirectional is not supported for encoder and decoder type 'dense'",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
