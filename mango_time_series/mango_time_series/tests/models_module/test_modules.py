import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
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
            # context_window=10,
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


class TestAutoEncoderBidirectional(unittest.TestCase):
    def test_bidirectional_autoencoder(self):
        """
        Test full autoencoder with bidirectional LSTM in encoder and decoder
        """
        hidden_dims = [32, 16]

        model = AutoEncoder()
        model.build_model(
            form="lstm",
            data=np.random.rand(500, 5),
            context_window=10,
            hidden_dim=hidden_dims,
            bidirectional_encoder=True,
            bidirectional_decoder=True,
            feature_to_check=[0, 1, 2, 3, 4],
            time_step_to_check=9,
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

    def test_not_implemented_for_dense(self):
        """
        Test that NotImplementedError is raised for dense model type
        """
        data = np.random.rand(500, 5)

        with self.assertRaises(NotImplementedError):
            model = AutoEncoder()
            model.build_model(
                form="dense",
                data=data,
                context_window=10,
                hidden_dim=[32, 16],
                bidirectional_encoder=True,
                bidirectional_decoder=True,
                feature_to_check=[0, 1, 2, 3, 4],
                time_step_to_check=9,
            )


class TestAutoEncoderCases(unittest.TestCase):
    """
    Test class for testing AutoEncoder with various configurations and data scenarios.

    This class implements comprehensive testing of the AutoEncoder model with different
    combinations of configurations and data scenarios. It tests:

    * Data formats (with/without IDs)
    * Missing values (with/without NaNs)
    * Mask types (None, auto, custom)
    * Data structures (single DataFrame vs tuple of DataFrames)

    The test class uses temporary directories for model outputs and ensures proper
    cleanup after each test.
    """

    def setUp(self):
        """
        Set up test environment before each test.

        Creates a temporary directory for model outputs and defines test cases
        with different configurations.

        :return: None
        :rtype: None
        """
        self.base_dir = Path(tempfile.mkdtemp())
        self.test_cases = [
            {
                "with_ids": False,
                "with_nans": False,
                "mask_type": None,
                "desc": "Sin IDs, Sin NaNs",
            },
            {
                "with_ids": True,
                "with_nans": False,
                "mask_type": None,
                "desc": "Con IDs, Sin NaNs",
            },
            {
                "with_ids": False,
                "with_nans": True,
                "mask_type": "auto",
                "desc": "Sin IDs, Con NaNs (Auto mask)",
            },
            {
                "with_ids": True,
                "with_nans": True,
                "mask_type": "auto",
                "desc": "Con IDs, Con NaNs (Auto mask)",
            },
            {
                "with_ids": False,
                "with_nans": True,
                "mask_type": "custom",
                "desc": "Sin IDs, Con NaNs (Custom mask)",
            },
            {
                "with_ids": True,
                "with_nans": True,
                "mask_type": "custom",
                "desc": "Con IDs, Con NaNs (Custom mask)",
            },
        ]

    def tearDown(self):
        """
        Clean up after each test.

        Removes the temporary directory and all its contents to ensure
        a clean state for the next test.

        :return: None
        :rtype: None
        """
        shutil.rmtree(self.base_dir)

    @staticmethod
    def generate_synthetic_data_standard(num_samples=500, num_features=3, **kwargs):
        """
        Generate synthetic data in single DataFrame format.

        :param num_samples: Number of samples to generate
        :type num_samples: int
        :param num_features: Number of features in the data
        :type num_features: int
        :param **kwargs: Additional parameters for data generation
            - with_ids (bool): Whether to include ID column
            - with_nans (bool): Whether to include NaN values
            - mask_type (str): Type of mask to generate ('auto', 'custom', or None)
        :type **kwargs: dict
        :return: Tuple containing DataFrame, use_mask flag, and mask array
        :rtype: tuple[pd.DataFrame, bool, Optional[np.ndarray]]
        """
        np.random.seed(42)
        data = np.random.rand(num_samples, num_features)

        if kwargs.get("with_ids"):
            ids = np.random.choice(["A", "B", "C"], size=num_samples)
            df = pd.DataFrame(
                np.column_stack((ids, data)),
                columns=["id"] + [f"feature_{i}" for i in range(num_features)],
            )
        else:
            df = pd.DataFrame(
                data, columns=[f"feature_{i}" for i in range(num_features)]
            )

        if kwargs.get("with_nans"):
            nan_indices = np.random.choice(
                num_samples, size=int(0.1 * num_samples), replace=False
            )
            for idx in nan_indices:
                col = np.random.randint(
                    1 if kwargs.get("with_ids") else 0,
                    num_features + (1 if kwargs.get("with_ids") else 0),
                )
                df.iloc[idx, col] = np.nan

        use_mask = kwargs.get("mask_type") is not None
        mask = None
        if use_mask and kwargs.get("mask_type") == "custom":
            if kwargs.get("with_ids"):
                # Create mask with same shape as data including ID column
                mask = np.ones((num_samples, num_features + 1), dtype=object)
                # Copy IDs from data to mask
                mask[:, 0] = df.iloc[:, 0].to_numpy()
                # Set feature columns to 2 (will be updated based on NaN values)
                mask[:, 1:] = 2
            else:
                mask = np.ones((num_samples, num_features))

        return df, use_mask, mask

    @staticmethod
    def generate_synthetic_data_tuple(num_samples=500, num_features=3, **kwargs):
        """
        Generate synthetic data in tuple format (train, val, test).

        :param num_samples: Number of samples to generate
        :type num_samples: int
        :param num_features: Number of features in the data
        :type num_features: int
        :param **kwargs: Additional parameters for data generation
            - with_ids (bool): Whether to include ID column
            - with_nans (bool): Whether to include NaN values
            - mask_type (str): Type of mask to generate ('auto', 'custom', or None)
        :type **kwargs: dict
        :return: Tuple containing split DataFrames, use_mask flag, and mask arrays
        :rtype: tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], bool, Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]]
        """
        np.random.seed(42)
        data = np.random.rand(num_samples, num_features)

        if kwargs.get("with_ids"):
            ids = np.random.choice(["A", "B", "C"], size=num_samples)
            df = pd.DataFrame(
                np.column_stack((ids, data)),
                columns=["id"] + [f"feature_{i}" for i in range(num_features)],
            )
        else:
            df = pd.DataFrame(
                data, columns=[f"feature_{i}" for i in range(num_features)]
            )

        if kwargs.get("with_nans"):
            nan_indices = np.random.choice(
                num_samples, size=int(0.1 * num_samples), replace=False
            )
            for idx in nan_indices:
                col = np.random.randint(
                    1 if kwargs.get("with_ids") else 0,
                    num_features + (1 if kwargs.get("with_ids") else 0),
                )
                df.iloc[idx, col] = np.nan

        train_size = int(num_samples * 0.8)
        val_size = int(num_samples * 0.1)

        df_train = df.iloc[:train_size].copy()
        df_val = df.iloc[train_size : train_size + val_size].copy()
        df_test = df.iloc[train_size + val_size :].copy()

        use_mask = kwargs.get("mask_type") is not None
        mask_train, mask_val, mask_test = None, None, None

        if use_mask and kwargs.get("mask_type") == "custom":
            if kwargs.get("with_ids"):
                # Create masks with same shape as data including ID column
                mask_train = np.ones((len(df_train), num_features + 1), dtype=object)
                mask_val = np.ones((len(df_val), num_features + 1), dtype=object)
                mask_test = np.ones((len(df_test), num_features + 1), dtype=object)
                
                # Copy IDs from data to mask
                mask_train[:, 0] = df_train.iloc[:, 0].to_numpy()
                mask_val[:, 0] = df_val.iloc[:, 0].to_numpy()
                mask_test[:, 0] = df_test.iloc[:, 0].to_numpy()
                
                # Set feature columns to 2 (will be updated based on NaN values)
                mask_train[:, 1:] = 2
                mask_val[:, 1:] = 2
                mask_test[:, 1:] = 2
            else:
                mask_train = np.ones((len(df_train), num_features))
                mask_val = np.ones((len(df_val), num_features))
                mask_test = np.ones((len(df_test), num_features))

        return (df_train, df_val, df_test), use_mask, (mask_train, mask_val, mask_test)

    def test_autoencoder_standard_format(self):
        """
        Tests all combinations of configurations using a single DataFrame format.
        For each test case:
        
        1. Generates synthetic data
        2. Creates and trains an AutoEncoder model
        3. Verifies reconstruction
        4. Checks output files existence

        :return: None
        :rtype: None
        """
        for i, case in enumerate(self.test_cases):
            with self.subTest(case=case["desc"]):
                # Generate data
                data, use_mask, mask = self.generate_synthetic_data_standard(**case)

                # Prepare save path
                save_path = self.base_dir / f"test_standard_{i+1}"
                save_path.mkdir(parents=True, exist_ok=True)

                # For case 6 (with IDs and custom mask)
                if case["mask_type"] == "custom" and case["with_ids"]:
                    mask[:, 1:][data.iloc[:, 1:].isna().to_numpy()] = 0
                    mask[:, 1:][data.iloc[:, 1:].notna().to_numpy()] = 2

                # Create and train model
                model = AutoEncoder()
                model.build_model(
                    form="lstm",
                    data=data,
                    context_window=3,
                    time_step_to_check=[2],
                    hidden_dim=[2, 1],
                    feature_to_check=[0],
                    bidirectional_encoder=False,
                    bidirectional_decoder=False,
                    normalize=True,
                    normalization_method="minmax",
                    batch_size=32,
                    save_path=str(save_path),
                    verbose=False,
                    feature_names=data.columns.tolist(),
                    use_mask=use_mask,
                    id_columns="id" if case["with_ids"] else None,
                    custom_mask=mask if case["mask_type"] == "custom" else None,
                    use_post_decoder_dense=False,
                )

                # Train and reconstruct
                model.train(epochs=1)
                reconstruction_result = model.reconstruct()

                # Assertions
                self.assertTrue(
                    reconstruction_result,
                    f"Reconstruction failed for case: {case['desc']}",
                )
                self.assertTrue(
                    save_path.exists(),
                    f"Save path not created for case: {case['desc']}",
                )

    def test_autoencoder_tuple_format(self):
        """
        Tests all combinations of configurations using a tuple format.
        For each test case:
        
        1. Generates synthetic data split into train/val/test
        2. Creates and trains an AutoEncoder model
        3. Verifies reconstruction
        4. Checks output files existence

        :return: None
        :rtype: None
        """
        for i, case in enumerate(self.test_cases):
            with self.subTest(case=case["desc"]):
                # Generate data
                (
                    (df_train, df_val, df_test),
                    use_mask,
                    (mask_train, mask_val, mask_test),
                ) = self.generate_synthetic_data_tuple(**case)
                data = (df_train, df_val, df_test)
                
                # Ajustar máscara solo a los datos numéricos
                if case["mask_type"] == "custom" and not case["with_ids"]:
                    for mask, df in zip(
                        [mask_train, mask_val, mask_test], [df_train, df_val, df_test]
                    ):
                        mask[df.isna().to_numpy()] = 0
                        mask[df.notna().to_numpy()] = 2
                elif case["mask_type"] == "custom" and case["with_ids"]:
                    for mask, df in zip(
                        [mask_train, mask_val, mask_test], [df_train, df_val, df_test]
                    ):
                        mask[:, 1:][df.iloc[:, 1:].isna().to_numpy()] = 0
                        mask[:, 1:][df.iloc[:, 1:].notna().to_numpy()] = 2
                
                use_mask_flag = case["mask_type"] is not None
                custom_mask_flag = (
                    (mask_train, mask_val, mask_test)
                    if case["mask_type"] == "custom"
                    else None
                )

                # Prepare save path
                save_path = self.base_dir / f"test_tuple_{i+1}"
                save_path.mkdir(parents=True, exist_ok=True)

                # Create and train model
                model = AutoEncoder()
                model.build_model(
                    form="lstm",
                    data=data,
                    context_window=3,
                    time_step_to_check=[2],
                    hidden_dim=[2, 1],
                    feature_to_check=[0],
                    bidirectional_encoder=False,
                    bidirectional_decoder=False,
                    normalize=True,
                    normalization_method="minmax",
                    batch_size=32,
                    save_path=str(save_path),
                    verbose=False,
                    feature_names=df_train.columns.tolist(),
                    use_mask=use_mask_flag,
                    id_columns="id" if case["with_ids"] else None,
                    custom_mask=custom_mask_flag,
                    use_post_decoder_dense=False,
                )

                # Train and reconstruct
                model.train(epochs=1)
                reconstruction_result = model.reconstruct()

                # Assertions
                self.assertTrue(
                    reconstruction_result,
                    f"Reconstruction failed for case: {case['desc']}",
                )
                self.assertTrue(
                    save_path.exists(),
                    f"Save path not created for case: {case['desc']}",
                )


if __name__ == "__main__":
    unittest.main()
