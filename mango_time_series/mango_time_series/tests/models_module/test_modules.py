import unittest
import tensorflow as tf

from mango_time_series.models.modules.decoder import decoder
from mango_time_series.models.modules.encoder import encoder
from mango_time_series.models.modules.encoder_embedder import encoder_embedder


class TestEncoder(unittest.TestCase):
    def test_lstm(self):
        """
        Test LSTM encoder creation and output shape
        """
        model = encoder(
            type="lstm",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 LSTM + Dense
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(model.name, "encoder")

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
            type="dense",
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        self.assertEqual(len(model.layers), 3)  # Input + 2 Dense
        self.assertEqual(model.name, "encoder")

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
            encoder(type="invalid", features=5, hidden_dim=32, num_layers=2)

    def test_lstm_variable_hidden_dims(self):
        """
        Test LSTM encoder with different hidden dimensions per layer
        """
        hidden_dims = [32, 16]
        model = encoder(
            type="lstm",
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


class TestEncoderEmbedder(unittest.TestCase):
    def test_basic(self):
        """
        Test basic encoder embedder creation and output shape
        """
        model = encoder_embedder(
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 LSTM
        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.name, "encoder_embedder")

        # Test output shape
        batch_size = 16
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, 10, 32))

    def test_single_layer(self):
        """
        Test encoder embedder with single layer
        """
        model = encoder_embedder(
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=1,
        )

        # Input + 1 LSTM
        self.assertEqual(len(model.layers), 2)

        batch_size = 16
        input_shape = (batch_size, 10, 5)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, 10, 32))

    def test_invalid_layers(self):
        """
        Test encoder embedder with invalid number of layers
        """
        with self.assertRaisesRegex(
            ValueError, "Number of layers must be greater than 0"
        ):
            encoder_embedder(context_window=10, features=5, hidden_dim=32, num_layers=0)


class TestDecoder(unittest.TestCase):
    def test_lstm(self):
        """
        Test LSTM decoder creation and output shape
        """
        model = decoder(
            type="lstm",
            context_window=10,
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 LSTM + Flatten + Dense
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.name, "decoder")

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
            type="dense",
            features=5,
            hidden_dim=32,
            num_layers=2,
        )

        # Check model type and layers
        # Input + 2 Dense + Output Dense
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(model.name, "decoder")

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
            decoder(type="invalid", features=5, hidden_dim=32, num_layers=2)

    def test_lstm_variable_hidden_dims(self):
        """
        Test LSTM decoder with different hidden dimensions per layer
        """
        hidden_dims = [32, 16]
        model = decoder(
            type="lstm",
            context_window=10,
            features=5,
            hidden_dim=hidden_dims,
            num_layers=2,
        )

        batch_size = 16
        # First hidden dim must match input
        input_shape = (batch_size, 10, 32)
        test_input = tf.random.normal(input_shape)
        output = model(test_input)
        self.assertEqual(output.shape, (batch_size, 5))


if __name__ == "__main__":
    unittest.main()
