# isort:skip_file
import shutil
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from keras.src.layers import Bidirectional

from mango_autoencoder.autoencoder import AutoEncoder
from mango_autoencoder.modules import decoder, encoder
from mango_autoencoder.utils import processing


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
            time_step_to_check=[9],
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
                time_step_to_check=[9],
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
        self.root_dir = Path.cwd()
        self.base_dir = self.root_dir / "test_autoencoder_cases"
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
        """

        autoencoder_dir = self.root_dir / "autoencoder"
        plots_dir = self.root_dir / "plots"
        if autoencoder_dir.exists() and autoencoder_dir.is_dir():
            shutil.rmtree(autoencoder_dir)

        if plots_dir.exists() and plots_dir.is_dir():
            shutil.rmtree(plots_dir)
        if self.base_dir.exists() and self.base_dir.is_dir():
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
                self.assertFalse(
                    reconstruction_result.empty,
                    f"Reconstruction empty for case: {case['desc']}",
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

                # Adjust mask only for numeric data values
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
                self.assertFalse(
                    reconstruction_result.empty,
                    f"Reconstruction empty for case: {case['desc']}",
                )
                self.assertTrue(
                    save_path.exists(),
                    f"Save path not created for case: {case['desc']}",
                )

    def test_actual_data_and_reconstruction_consistency(self):
        """
        Test reconstruct() and reconstruct_new_data() functions for different
        cases of time_step_to_check, feature_to_check, with_ids, and with_nans.
        Ensures that the original data is not altered, reconstruction has expected
        dimensions, and reconstructions from both functions align.

        :return: None
        :rtype: None
        """
        # Set parameters, data cases, and loop through each
        # Reconstruction should align when reconstruct_new_data iterations is 1
        context_window = 10
        rec_new_data_iterations = 1
        time_step_to_check_list = [[0], [5], [9]]
        feature_to_check_list = [[0], [1, 5, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
        for feature_to_check in feature_to_check_list:
            for time_step_to_check in time_step_to_check_list:
                data_cases = [
                    self.generate_synthetic_data_standard(
                        num_samples=111,
                        num_features=8,
                        with_ids=False,
                        with_nans=False,
                        mask_type=None,
                    ),
                    self.generate_synthetic_data_standard(
                        num_samples=1111,
                        num_features=8,
                        with_ids=True,
                        with_nans=False,
                        mask_type=None,
                    ),
                    self.generate_synthetic_data_standard(
                        num_samples=111,
                        num_features=8,
                        with_ids=False,
                        with_nans=True,
                        mask_type="auto",
                    ),
                    self.generate_synthetic_data_standard(
                        num_samples=1111,
                        num_features=8,
                        with_ids=True,
                        with_nans=True,
                        mask_type="auto",
                    ),
                ]
                for data, use_mask, _ in data_cases:
                    with self.subTest(
                        time_step_to_check=time_step_to_check,
                        feature_to_check=feature_to_check,
                        use_mask=use_mask,
                    ):
                        id_columns = "id" if "id" in data.columns else None
                        model = AutoEncoder()
                        model.build_model(
                            form="lstm",
                            data=data,
                            context_window=context_window,
                            time_step_to_check=time_step_to_check,
                            hidden_dim=[8, 4],
                            feature_to_check=feature_to_check,
                            id_columns=id_columns,
                            bidirectional_encoder=True,
                            bidirectional_decoder=False,
                            normalize=True,
                            normalization_method="minmax",
                            batch_size=16,
                            save_path=None,
                            verbose=False,
                            use_mask=use_mask,
                            shuffle=True,
                        )

                        # Save _create_data_points_df original function and create a new one
                        # in order to store df_actual
                        original_create_data_points = model._create_data_points_df
                        stored_data = {}

                        def edited_create_data_points_df(*args, **kargs):
                            df_actual, df_reconstructed = original_create_data_points(
                                *args, **kargs
                            )
                            stored_data["df_actual"] = df_actual.copy()
                            stored_data["df_reconstructed"] = df_reconstructed.copy()
                            return df_actual, df_reconstructed

                        model._create_data_points_df = edited_create_data_points_df
                        model.reconstruct()
                        model._create_data_points_df = original_create_data_points
                        nan_positions = model._nan_coordinates
                        df_actual = stored_data["df_actual"]
                        df_reconstructed = stored_data["df_reconstructed"]

                        # Define context offset
                        initial_context_offset = time_step_to_check[0]
                        ending_context_offset = (
                            context_window - 1 - initial_context_offset
                        )

                        # Case with multiple datasets (with_ids=True)
                        if "id" in df_actual.columns:
                            for id_i in df_actual["id"].unique():
                                # Get df_actual (original data used in reconstruct())
                                df_actual_i = df_actual[df_actual["id"] == id_i]
                                df_actual_i = pd.pivot(
                                    df_actual_i,
                                    columns="feature",
                                    index="time_step",
                                    values="value",
                                )

                                # Get expected_data (original raw data)
                                data_i = data[data["id"] == id_i]
                                expected_data = data_i.drop(columns="id")
                                expected_data = expected_data.iloc[
                                    initial_context_offset : len(expected_data)
                                    - ending_context_offset
                                ]
                                expected_data = expected_data.iloc[:, feature_to_check]
                                expected_data = expected_data.reset_index(drop=True)

                                # Add back NaN positions to df_actual
                                # since reconstruct pipeline imputes them
                                for row_idx, col_idx in nan_positions[id_i]:
                                    adjusted_row_idx = row_idx - initial_context_offset
                                    adjusted_col_idx = col_idx
                                    if 0 <= adjusted_row_idx < len(df_actual_i):
                                        df_actual_i.iloc[
                                            adjusted_row_idx, adjusted_col_idx
                                        ] = np.nan

                                # Check df_actual and expected_data match
                                pd.testing.assert_frame_equal(
                                    df_actual_i.astype(float).round(6),
                                    expected_data.astype(float).round(6),
                                    check_names=False,
                                )

                                # Check single reconstruction
                                df_reconstruct_i = df_reconstructed[
                                    df_reconstructed.id == id_i
                                ]
                                df_reconstruct_i = pd.pivot(
                                    df_reconstruct_i,
                                    columns="feature",
                                    index=["time_step"],
                                    values="value",
                                )
                                reconstruct_new_data = model.reconstruct_new_data(
                                    data=data_i,
                                    iterations=rec_new_data_iterations,
                                    save_path=None,
                                    id_columns=id_columns,
                                )
                                df_reconstruct_new_data_i = reconstruct_new_data[id_i]
                                self.assertEqual(
                                    len(df_reconstruct_new_data_i), len(expected_data)
                                )
                                self.assertEqual(
                                    df_reconstruct_new_data_i.index.min(),
                                    initial_context_offset,
                                )

                                df_reconstruct_new_data_i = (
                                    df_reconstruct_new_data_i.reset_index(drop=True)
                                )

                                # Check that reconstructions match
                                pd.testing.assert_frame_equal(
                                    df_reconstruct_i,
                                    df_reconstruct_new_data_i,
                                    check_names=False,
                                    check_exact=False,
                                    atol=0.00001,
                                    rtol=0,
                                )

                            # Check multiple reconstructions
                            reconstruct_new_data = model.reconstruct_new_data(
                                data=data,
                                iterations=rec_new_data_iterations,
                                save_path=None,
                                id_columns=id_columns,
                            )
                            for (
                                id_i,
                                df_reconstruct_new_data_i,
                            ) in reconstruct_new_data.items():
                                data_i = data[data["id"] == id_i]
                                expected_data = data_i.iloc[
                                    initial_context_offset : len(data_i)
                                    - ending_context_offset
                                ]

                                self.assertEqual(
                                    len(df_reconstruct_new_data_i), len(expected_data)
                                )
                                self.assertEqual(
                                    df_reconstruct_new_data_i.index.min(),
                                    initial_context_offset,
                                )

                                df_reconstruct_i = df_reconstructed[
                                    df_reconstructed.id == id_i
                                ]
                                df_reconstruct_i = pd.pivot(
                                    df_reconstruct_i,
                                    columns="feature",
                                    index=["time_step"],
                                    values="value",
                                )
                                df_reconstruct_new_data_i = (
                                    df_reconstruct_new_data_i.reset_index(drop=True)
                                )

                                # Check that reconstructions match
                                pd.testing.assert_frame_equal(
                                    df_reconstruct_i,
                                    df_reconstruct_new_data_i,
                                    check_names=False,
                                    check_exact=False,
                                    atol=0.00001,
                                    rtol=0,
                                )

                        # Case with single dataset (with_ids=False)
                        else:
                            # Get df_actual (original data used in reconstruct())
                            df_actual_i = pd.pivot(
                                df_actual,
                                columns="feature",
                                index="time_step",
                                values="value",
                            )

                            # Get expected_data (original raw data)
                            expected_data = data.iloc[
                                initial_context_offset : len(data)
                                - ending_context_offset
                            ]
                            expected_data = expected_data.iloc[:, feature_to_check]

                            expected_data = expected_data.reset_index(drop=True)

                            # Add back NaN positions to df_actual
                            # since reconstruct pipeline imputes them
                            for row_idx, col_idx in nan_positions["global"]:
                                adjusted_row_idx = row_idx - initial_context_offset
                                adjusted_col_idx = col_idx
                                if 0 <= adjusted_row_idx < len(df_actual_i):
                                    df_actual_i.iloc[
                                        adjusted_row_idx, adjusted_col_idx
                                    ] = np.nan

                            # Check df_actual and expected_data match
                            pd.testing.assert_frame_equal(
                                df_actual_i.astype(float).round(6),
                                expected_data.astype(float).round(6),
                                check_names=False,
                            )

                            # Check single reconstruction
                            df_reconstruct_i = df_reconstructed
                            df_reconstruct_i = pd.pivot(
                                df_reconstruct_i,
                                columns="feature",
                                index=["time_step"],
                                values="value",
                            )
                            reconstruct_new_data = model.reconstruct_new_data(
                                data=data,
                                iterations=rec_new_data_iterations,
                                save_path=None,
                                id_columns=id_columns,
                            )
                            df_reconstruct_new_data_i = reconstruct_new_data["global"]
                            self.assertEqual(
                                len(df_reconstruct_new_data_i), len(expected_data)
                            )
                            self.assertEqual(
                                df_reconstruct_new_data_i.index.min(),
                                initial_context_offset,
                            )

                            df_reconstruct_new_data_i = (
                                df_reconstruct_new_data_i.reset_index(drop=True)
                            )

                            # Check that reconstructions match
                            pd.testing.assert_frame_equal(
                                df_reconstruct_i,
                                df_reconstruct_new_data_i,
                                check_names=False,
                                check_exact=False,
                                atol=0.00001,
                                rtol=0,
                            )

    def test_accepts_all_input_formats(self):
        """
        Ensure AutoEncoder accepts data as pandas DataFrame, polars DataFrame,
        numpy array, and tuple of numpy arrays.

        :return: None
        :rtype: None
        """
        # Define data types
        data = self.generate_synthetic_data_standard(
            num_samples=60,
            num_features=4,
            with_ids=False,
            with_nans=False,
            mask_type=None,
        )

        pandas_df = data[0]
        polars_df = pl.from_pandas(pandas_df)
        numpy_array = pandas_df.to_numpy()
        tuple_numpy_arrays = tuple(
            processing.time_series_split(
                data=numpy_array, train_size=0.4, val_size=0.3, test_size=0.3
            )
        )

        data_list = {
            "pandas DataFrame": pandas_df,
            "polars DataFrame": polars_df,
            "numpy array": numpy_array,
            "tuple numpy arrays": tuple_numpy_arrays,
        }

        # Loop through different data types
        for data_type, data_i in data_list.items():
            with self.subTest(data_type=data_type):
                model = AutoEncoder()
                model.build_model(
                    form="lstm",
                    data=data_i,
                    context_window=4,
                    time_step_to_check=[2],
                    hidden_dim=[8, 4],
                    feature_to_check=[0, 1, 2, 3],
                    id_columns=None,
                    bidirectional_encoder=True,
                    bidirectional_decoder=False,
                    normalize=True,
                    batch_size=16,
                    save_path=None,
                    verbose=False,
                    use_mask=True,
                    shuffle=True,
                )

                # Train and reconstruct
                model.train(epochs=1)
                reconstruct_result = model.reconstruct()
                reconstruct_result = pd.pivot(
                    reconstruct_result,
                    columns="feature",
                    index=["time_step"],
                    values="value",
                )
                reconstruct_new_data_result = model.reconstruct_new_data(
                    data=(
                        data_i
                        if data_type != "tuple numpy arrays"
                        else np.concatenate(data_i, axis=0)
                    ),
                    iterations=1,
                )
                reconstruct_new_data_result = reconstruct_new_data_result["global"]
                reconstruct_new_data_result = reconstruct_new_data_result.reset_index(
                    drop=True
                )

                # Assertions
                self.assertFalse(
                    reconstruct_result.empty,
                    "Reconstruction empty.",
                )
                pd.testing.assert_frame_equal(
                    reconstruct_result,
                    reconstruct_new_data_result,
                    check_names=False,
                    check_exact=False,
                    atol=0.00001,
                    rtol=0,
                )

    def test_load_from_pickle(self):
        """
        Test load_from_pickle() works when normalize = False.

        :return: None
        :rtype: None
        """
        # Generate data
        data, _, _ = self.generate_synthetic_data_standard(
            num_samples=50,
            num_features=4,
            with_ids=False,
            with_nans=False,
            mask_type=None,
        )

        # Prepare save path
        save_path = self.base_dir / "test_load_from_pickle"

        # Build model
        model = AutoEncoder()
        model.build_model(
            form="lstm",
            data=data,
            context_window=4,
            time_step_to_check=[2],
            hidden_dim=[8, 4],
            feature_to_check=[0, 1, 2, 3],
            id_columns=None,
            bidirectional_encoder=True,
            bidirectional_decoder=False,
            normalize=False,
            batch_size=16,
            save_path=save_path,
            verbose=False,
            use_mask=True,
            shuffle=True,
        )

        # Train and reconstruct
        model.train(epochs=1)
        reconstruct_result = model.reconstruct()
        reconstruct_result = pd.pivot(
            reconstruct_result,
            columns="feature",
            index=["time_step"],
            values="value",
        )

        # Get model path, reconstruct new data based on pickle
        model_path = save_path / "models" / "best_model.pkl"
        model_pickle = AutoEncoder()
        model_pickle = model_pickle.load_from_pickle(path=model_path)
        reconstruct_new_data_result = model_pickle.reconstruct_new_data(
            data=data,
            iterations=1,
        )
        reconstruct_new_data_result = reconstruct_new_data_result["global"]
        reconstruct_new_data_result = reconstruct_new_data_result.reset_index(drop=True)

        # Assertions
        self.assertFalse(
            reconstruct_result.empty,
            "Reconstruction empty.",
        )
        pd.testing.assert_frame_equal(
            reconstruct_result,
            reconstruct_new_data_result,
            check_dtype=False,
            check_names=False,
            check_exact=False,
            atol=0.00001,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
