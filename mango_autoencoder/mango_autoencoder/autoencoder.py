import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from keras import Sequential
from keras.src.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, Nadam, RMSprop
from mango.processing.data_imputer import DataImputer
from mango_autoencoder.logging import get_configured_logger
from mango_autoencoder.modules import anomaly_detector, decoder, encoder
from mango_autoencoder.utils import plots, processing
from mango_autoencoder.utils.sequences import time_series_to_sequence
from tensorflow.keras.layers import Dense

logger = get_configured_logger()


class AutoEncoder:
    """
    Autoencoder model for time series data reconstruction and anomaly detection.

    An autoencoder is a neural network that learns to compress and reconstruct data.
    This implementation is designed specifically for time series data, allowing for
    sequence-based encoding and decoding with various architectures (LSTM, GRU, RNN).

    The model can be highly configurable but is already set up for quick training
    and profiling. It supports data normalization, masking for missing values,
    and various training options including early stopping and checkpointing.

    :param TRAIN_SIZE: Proportion of data used for training (default: 0.8)
    :type TRAIN_SIZE: float
    :param VAL_SIZE: Proportion of data used for validation (default: 0.1)
    :type VAL_SIZE: float
    :param TEST_SIZE: Proportion of data used for testing (default: 0.1)
    :type TEST_SIZE: float

    Example:
        >>> autoencoder = AutoEncoder()
        >>> autoencoder.form = "LSTM"
        >>> autoencoder.context_window = 10
        >>> autoencoder.fit(data)
        >>> predictions = autoencoder.predict(test_data)
    """

    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    TEST_SIZE = 0.1

    def __init__(self) -> None:
        """
        Initialize the Autoencoder model with default parameters.

        Initializes internal state variables including paths, model configuration,
        and normalization settings. Sets up default values for all configurable
        parameters and prepares the model for training.

        :return: None
        :rtype: None

        Example:
            >>> autoencoder = AutoEncoder()
            >>> print(autoencoder.form)  # Default form
            >>> print(autoencoder.context_window)  # Default context window

        """
        # Path settings
        self._num_layers = None
        self.root_dir = os.path.abspath(os.getcwd())
        self._save_path = None

        # Model architecture settings
        self._form = "lstm"
        self.model = None
        self.layers = []
        self._model_optimizer = None
        self._bidirectional_encoder = False
        self._bidirectional_decoder = False
        self._activation_encoder = None
        self._activation_decoder = None

        # Data processing settings
        self._normalization_method = None
        self.normalization_values = {}
        self._normalize = False
        self.imputer = None
        self._id_columns_indices = None
        self._data = None
        self._features_name = None
        self._feature_weights = None
        self._id_data = None
        self._id_data_dict = None
        self._id_data_mask = None
        self._id_data_dict_mask = None

        # Dataset splits
        self.x_train = None
        self.x_val = None
        self.x_test = None

        # Mask settings
        self._mask_train = None
        self._mask_val = None
        self._mask_test = None

        # Training settings
        self._verbose = False
        self._shuffle_buffer_size = None
        self._x_train_no_shuffle = None
        self._feature_to_check = None
        self._time_step_to_check = None
        self._context_window = None
        self._hidden_dim = None
        self._train_size = self.TRAIN_SIZE
        self._val_size = self.VAL_SIZE
        self._test_size = self.TEST_SIZE
        self._use_mask = False
        self._custom_mask = None
        self._shuffle = False
        self._nan_coordinates = {}
        self._seed = None

    @property
    def nan_coordinates(self) -> Dict:
        """
        Get the NaN coordinates from the input data.

        Dictionary maps each id (id_columns or global if not used)
        to a NumPy array of shape (n, 2), where each row contains the
        [row_index, column_index] of a NaN found in the input data.

        :return: Dictionary mapping id to array of NaN positions
        :rtype: Dict
        """
        return self._nan_coordinates

    @property
    def seed(self) -> Optional[int]:
        """
        Get the seed for reproducibility (all random generators).

        :return: Seed for reproducibility
        :rtype: Optional[str]
        """
        return self._seed

    @seed.setter
    def seed(self, value: Optional[int]) -> None:
        """
        Set the seed for reproducibility (all random generators).

        :return: None
        :rtype: None
        """
        if value is None:
            self._seed = None
        elif type(value) is not int:
            raise ValueError("value must be type int.")
        elif value < 0 or value > 2**32 - 1:
            raise ValueError("value must be between 0 and 2**32 - 1.")
        else:
            self._set_all_seeds(value)

    def _set_all_seeds(self, value: int) -> None:
        """
        Helper to set seed for all random generators.

        :return: None
        :rtype: None
        """
        self._seed = value
        np.random.seed(value)
        tf.random.set_seed(value)
        random.seed(value)

    @property
    def save_path(self) -> Optional[str]:
        """
        Get the path where model artifacts will be saved.

        :return: Path to save model or None if not set
        :rtype: Optional[str]
        """
        return self._save_path

    @save_path.setter
    def save_path(self, path: Optional[str]) -> None:
        """
        Set the path where model artifacts will be saved.

        :param path: Directory path for saving model artifacts
        :type path: Optional[str]
        :return: None
        :rtype: None
        """
        self._save_path = path or os.path.join(self.root_dir, "autoencoder")

        self.create_folder_structure(
            [
                os.path.join(self._save_path, "models"),
                os.path.join(self._save_path, "plots"),
            ]
        )

    @property
    def form(self) -> str:
        """
        Get the encoder/decoder architecture type.

        :return: Architecture type ('lstm', 'gru', 'rnn', or 'dense')
        :rtype: str
        """
        return self._form

    @form.setter
    def form(self, value: str) -> None:
        """
        Set the encoder/decoder architecture type.

        :param value: Architecture type ('lstm', 'gru', 'rnn', or 'dense')
        :type value: str
        :return: None
        :rtype: None
        :raises ValueError: If value is not one of the supported architectures or if
                           attempting to change after model is built
        """
        self._validate_form_change(value)
        self._form = value

    def _validate_form_change(self, value: str) -> None:
        """
        Validate that the form can be changed to the specified value.

        :param value: The new form value to validate
        :type value: str
        :raises ValueError: If the form cannot be changed to the specified value
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError(
                "Cannot change form after model is built. "
                "Call build_model() with the new form instead."
            )

        valid_forms = ["lstm", "gru", "rnn", "dense"]
        if value not in valid_forms:
            raise ValueError(f"Form must be one of {valid_forms}")

        if value == "dense":
            raise NotImplementedError("Dense model type is not yet implemented")

    @property
    def time_step_to_check(self) -> Optional[List[int]]:
        """
        Get the time step indices to check during reconstruction.

        :return: Time step indices
        :rtype: Optional[List[int]]
        """
        if not hasattr(self, "_time_step_to_check"):
            return None
        return self._time_step_to_check

    @time_step_to_check.setter
    def time_step_to_check(self, value: List[int]) -> None:
        """
        Set the time step indices to check during reconstruction.

        :param value: Time step indices to check
        :type value: List[int]
        :return: None
        :rtype: None
        :raises ValueError: If value is not valid or if attempting to change after model is built
        """
        # If model is built, don't allow changes
        if hasattr(self, "model") and self.model is not None:
            raise ValueError(
                "Cannot change time_step_to_check after model is built. "
                "Call build_model() with the new time_step_to_check instead."
            )

        if not isinstance(value, list):
            raise ValueError("time_step_to_check must be a list of integers")

        if len(value) != 1:
            raise NotImplementedError(
                "Currently time_step_to_check is implemented to consider only one integer index."
            )

        # Validate all values are integers
        if not all(isinstance(t, int) for t in value):
            raise ValueError("All elements in time_step_to_check must be integers")

        # If context_window is set, validate indices are in range
        if not hasattr(self, "_context_window"):
            raise ValueError("Context window is not set")
        if self._context_window is not None:
            if any(t < 0 or t >= self._context_window for t in value):
                raise ValueError(
                    "time_step_to_check contains invalid indices. "
                    f"Must be between 0 and {self._context_window - 1}."
                )

        self._time_step_to_check = value

    @property
    def context_window(self) -> Optional[int]:
        """
        Get the context window size.

        :return: Context window size or None if not initialized
        :rtype: Optional[int]
        """
        return self._context_window

    @context_window.setter
    def context_window(self, value: int) -> None:
        """
        Set the context window size.

        :param value: Context window size
        :type value: int
        :return: None
        :rtype: None
        :raises ValueError: If value is not a positive integer or if attempting to change after model is built
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change context_window after model is built")

        if not isinstance(value, int) or value <= 0:
            raise ValueError("Context window must be a positive integer")

        self._context_window = value

    @property
    def feature_weights(self) -> Optional[List[float]]:
        """
        Get the feature weights.

        :return: Feature weights
        :rtype: Optional[List[float]]
        """
        return self._feature_weights

    @feature_weights.setter
    def feature_weights(self, value: Optional[List[float]]) -> None:
        """
        Set the feature weights for loss calculation.

        Feature weights allow for different importance levels for each feature
        during model training. Higher weights increase the influence of that
        feature in the loss calculation.

        :param value: List of weights for each feature (None to disable weighting)
        :type value: Optional[List[float]]
        :return: None
        :rtype: None
        :raises ValueError: If weights list length doesn't match number of features

        Example:
            >>> autoencoder = AutoEncoder()
            >>> autoencoder.feature_weights = [1.0, 2.0, 0.5]  # Weight features differently
            >>> autoencoder.feature_weights = None  # Disable weighting

        """
        self._feature_weights = value

    @property
    def features_name(self) -> Optional[List[str]]:
        """
        Get the feature names used for training the model.

        :return: List of feature names or None if not set
        :rtype: Optional[List[str]]
        """
        return self._features_name

    @features_name.setter
    def features_name(self, value: List[str]) -> None:
        """
        Set the features name used for training the model.

        :param value: List of feature names
        :type value: List[str]
        :return: None
        :rtype: None
        """
        self._features_name = value

    @property
    def data(self) -> Optional[np.ndarray]:
        """
        Get the data used for training the model.

        :return: Data used for training the model
        :rtype: Optional[np.ndarray]
        """
        return self._data

    @data.setter
    def data(self, value: Optional[np.ndarray]) -> None:
        """
        Set the data used for training the model.

        :param value: Training data as numpy array or tuple of three arrays (train, val, test)
        :type value: Optional[np.ndarray]
        :return: None
        :rtype: None
        :raises ValueError: If data format is invalid

        Example:
            >>> autoencoder = AutoEncoder()
            >>> # Single dataset (will be split automatically)
            >>> autoencoder.data = np.random.randn(1000, 5)
            >>> # Pre-split datasets
            >>> train_data = np.random.randn(800, 5)
            >>> val_data = np.random.randn(100, 5)
            >>> test_data = np.random.randn(100, 5)
            >>> autoencoder.data = (train_data, val_data, test_data)

        """
        # Validate that data is a single array or a tuple of three arrays
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError(
                    "If data is a tuple, it must be three numpy arrays. "
                    "First array is train set, second is validation, and third is test."
                )
        elif not isinstance(value, np.ndarray):
            raise ValueError(
                "Data must be a numpy array or a tuple with three numpy arrays"
            )

        self._data = value

    @property
    def feature_to_check(self) -> Optional[List[int]]:
        """
        Get the feature index or indices to check during reconstruction.

        :return: Feature index or indices to check
        :rtype: Optional[Union[int, List[int]]]
        """
        return self._feature_to_check

    @feature_to_check.setter
    def feature_to_check(self, value: List[int]) -> None:
        """
        Set the feature index or indices to check during reconstruction.

        :param value: Feature index or indices to check
        :type value: List[int]
        :return: None
        :rtype: None
        :raises ValueError: If value is not valid or if attempting to change after model is built
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change feature_to_check after model is built")

        if not isinstance(value, list):
            raise ValueError("feature_to_check must be a list of integers")

        self._feature_to_check = value

    @property
    def normalize(self) -> bool:
        """
        Get the normalization flag.

        :return: Normalization flag
        :rtype: bool
        """
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        """
        Set the normalization flag.

        :param value: Normalization flag
        :type value: bool
        :raises ValueError: If trying to enable normalization without a method set
        """
        if value and self._normalization_method is None:
            raise ValueError(
                "Cannot enable normalization without setting a normalization method. "
                "Set normalization_method first."
            )
        self._normalize = value

    @property
    def normalization_method(self) -> Optional[str]:
        """
        Get the normalization method.

        :return: Normalization method
        :rtype: Optional[str]
        """
        return self._normalization_method

    @normalization_method.setter
    def normalization_method(self, value: str) -> None:
        """
        Set the normalization method.

        :param value: Normalization method
        :type value: str
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change normalization_method after model is built")

        if value not in ["minmax", "zscore", None]:
            raise ValueError(
                f"Invalid normalization method: {value}. Choose 'minmax', 'zscore', or None."
            )

        self._normalization_method = value

    @property
    def hidden_dim(self) -> Optional[Union[int, List[int]]]:
        """
        Get the hidden dimensions.

        :return: Hidden dimensions
        :rtype: Optional[Union[int, List[int]]]
        """
        return self._hidden_dim

    @hidden_dim.setter
    def hidden_dim(self, value: Union[int, List[int]]) -> None:
        """
        Set the hidden dimensions.

        :param value: Hidden dimensions
        :type value: Union[int, List[int]]
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change hidden_dim after model is built")

        if not isinstance(value, (int, list)):
            raise ValueError("hidden_dim must be an int or list of ints")

        if isinstance(value, int):
            value = [value]

        self._hidden_dim = value

    @property
    def bidirectional_encoder(self) -> bool:
        """
        Get the bidirectional encoder flag.

        :return: Bidirectional encoder flag
        :rtype: bool
        """
        return self._bidirectional_encoder

    @bidirectional_encoder.setter
    def bidirectional_encoder(self, value: bool) -> None:
        """
        Set the bidirectional encoder flag.

        :param value: Bidirectional encoder flag
        :type value: bool
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change bidirectional_encoder after model is built")

        bidirectional_allowed = {"lstm", "gru", "rnn"}
        if getattr(self, "_form") not in bidirectional_allowed:
            raise ValueError(
                f"Bidirectional not supported for encoder/decoder type '{self.form}'"
            )

        self._bidirectional_encoder = value

    @property
    def bidirectional_decoder(self) -> bool:
        """
        Get the bidirectional decoder flag.

        :return: Bidirectional decoder flag
        :rtype: bool
        """
        return self._bidirectional_decoder

    @bidirectional_decoder.setter
    def bidirectional_decoder(self, value: bool) -> None:
        """
        Set the bidirectional decoder flag.

        :param value: Bidirectional decoder flag
        :type value: bool
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change bidirectional_decoder after model is built")

        bidirectional_allowed = {"lstm", "gru", "rnn"}
        if getattr(self, "_form") not in bidirectional_allowed:
            raise ValueError(
                f"Bidirectional not supported for encoder/decoder type '{self.form}'"
            )

        self._bidirectional_decoder = value

    @property
    def activation_encoder(self) -> Optional[str]:
        """
        Get the activation function for the encoder.

        :return: Activation function
        :rtype: Optional[str]
        """
        return self._activation_encoder

    @activation_encoder.setter
    def activation_encoder(self, value: Optional[str]) -> None:
        """
        Set the activation function for the encoder.

        :param value: Activation function
        :type value: str
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change activation_encoder after model is built")

        valid_activations = {
            "relu",
            "sigmoid",
            "softmax",
            "softplus",
            "softsign",
            "tanh",
            "selu",
            "elu",
            "exponential",
            "linear",
            "swish",
        }
        if value is not None and value not in valid_activations:
            raise ValueError(
                f"Invalid activation_encoder '{value}'. Must be one of: {sorted(valid_activations)}"
            )

        self._activation_encoder = value

    @property
    def activation_decoder(self) -> Optional[str]:
        """
        Get the activation function for the decoder.

        :return: Activation function
        :rtype: Optional[str]
        """
        return self._activation_decoder

    @activation_decoder.setter
    def activation_decoder(self, value: Optional[str]) -> None:
        """
        Set the activation function for the decoder.
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change activation_decoder after model is built")

        valid_activations = {
            "relu",
            "sigmoid",
            "softmax",
            "softplus",
            "softsign",
            "tanh",
            "selu",
            "elu",
            "exponential",
            "linear",
            "swish",
        }
        if value is not None and value not in valid_activations:
            raise ValueError(
                f"Invalid activation_decoder '{value}'. Must be one of: {sorted(valid_activations)}"
            )

        self._activation_decoder = value

    @property
    def verbose(self) -> bool:
        """
        Get the verbose flag.

        :return: Verbose flag
        :rtype: bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """
        Set the verbose flag.

        :param value: Verbose flag
        :type value: bool
        :return: None
        :rtype: None
        """
        self._verbose = value

    @property
    def train_size(self) -> float:
        """
        Get the training set size proportion.

        :return: Training set size (0.0-1.0)
        :rtype: float
        """
        return self._train_size

    @train_size.setter
    def train_size(self, value: float) -> None:
        """
        Set the training set size proportion.

        :param value: Training set size (0.0-1.0)
        :type value: float
        :return: None
        :rtype: None
        :raises ValueError: If value is not between 0 and 1
        """
        if not 0 <= value <= 1:
            raise ValueError("train_size must be between 0 and 1")

        self._train_size = value

    @property
    def val_size(self) -> float:
        """
        Get the validation set size proportion.

        :return: Validation set size (0.0-1.0)
        :rtype: float
        """
        return self._val_size

    @val_size.setter
    def val_size(self, value: float) -> None:
        """
        Set the validation set size proportion.

        :param value: Validation set size (0.0-1.0)
        :type value: float
        :return: None
        :rtype: None
        :raises ValueError: If value is not between 0 and 1
        """
        if not 0 <= value <= 1:
            raise ValueError("val_size must be between 0 and 1")

        self._val_size = value

    @property
    def test_size(self) -> float:
        """
        Get the test set size proportion.

        :return: Test set size (0.0-1.0)
        :rtype: float
        """
        return self._test_size

    @test_size.setter
    def test_size(self, value: float) -> None:
        """
        Set the test set size proportion.

        :param value: Test set size (0.0-1.0)
        :type value: float
        :return: None
        :rtype: None
        :raises ValueError: If value is not between 0 and 1
        """
        if not 0 <= value <= 1:
            raise ValueError("test_size must be between 0 and 1")

        self._test_size = value

    @property
    def num_layers(self) -> int:
        """
        Get the number of layers in the encoder/decoder architecture.

        :return: Number of layers (0 if hidden_dim is not set)
        :rtype: int
        """
        if self._hidden_dim is None:
            return 0
        return len(self._hidden_dim)

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        """
        Set the number of layers in the encoder/decoder architecture.

        :param value: Number of layers
        :type value: int
        :return: None
        :rtype: None
        :raises ValueError: If value is negative
        """
        self._num_layers = value

    @property
    def id_data(self) -> Optional[np.ndarray]:
        """
        Get the ID data used for grouping time series.

        :return: ID data array or None if not using ID-based processing
        :rtype: Optional[np.ndarray]
        """
        return self._id_data

    @id_data.setter
    def id_data(self, value: Optional[np.ndarray]) -> None:
        """
        Set the ID data for grouping time series.

        :param value: ID data array or None to disable ID-based processing
        :type value: Optional[np.ndarray]
        :return: None
        :rtype: None
        """
        self._id_data = value

    @property
    def id_data_dict(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the ID data dictionary mapping IDs to their respective datasets.

        :return: Dictionary mapping ID strings to numpy arrays or None if not using ID-based processing
        :rtype: Optional[Dict[str, np.ndarray]]
        """
        return self._id_data_dict

    @id_data_dict.setter
    def id_data_dict(self, value: Optional[Dict[str, np.ndarray]]) -> None:
        """
        Set the ID data dictionary mapping IDs to their respective datasets.

        :param value: Dictionary mapping ID strings to numpy arrays or None to disable ID-based processing
        :type value: Optional[Dict[str, np.ndarray]]
        :return: None
        :rtype: None
        """
        self._id_data_dict = value

    @property
    def id_data_mask(self) -> Optional[np.ndarray]:
        """
        Get the ID data mask.
        """
        return self._id_data_mask

    @id_data_mask.setter
    def id_data_mask(self, value: Optional[np.ndarray]) -> None:
        """
        Set the ID data mask.
        """
        self._id_data_mask = value

    @property
    def id_data_dict_mask(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the ID data mask dictionary.
        """
        return self._id_data_dict_mask

    @id_data_dict_mask.setter
    def id_data_dict_mask(self, value: Optional[Dict[str, np.ndarray]]) -> None:
        """
        Set the ID data mask dictionary.
        """
        self._id_data_dict_mask = value

    @property
    def id_columns_indices(self) -> List[int]:
        """
        Get the indices of the ID columns used for grouping data.

        :return: List of column indices that contain ID information
        :rtype: List[int]
        """
        return self._id_columns_indices

    @id_columns_indices.setter
    def id_columns_indices(self, value: List[int]) -> None:
        """
        Set the indices of the ID columns used for grouping data.

        :param value: List of column indices that contain ID information
        :type value: List[int]
        :return: None
        :rtype: None
        """
        self._id_columns_indices = value

    @property
    def use_mask(self) -> bool:
        """
        Get the use_mask flag indicating whether masking is enabled for missing values.

        :return: True if masking is enabled, False otherwise
        :rtype: bool
        """
        return self._use_mask

    @use_mask.setter
    def use_mask(self, value: bool) -> None:
        """
        Set the use_mask flag for handling missing values.

        :param value: True to enable masking for missing values, False to disable
        :type value: bool
        :return: None
        :rtype: None
        :raises ValueError: If data contains NaNs but use_mask is False
        """
        if not value and getattr(self, "_data", False):
            arrays_to_check = (
                self._data if isinstance(self._data, tuple) else [self._data]
            )
            if any(np.isnan(arr).any() for arr in arrays_to_check):
                raise ValueError(
                    "Data contains NaNs but use_mask is False. Clean or impute data."
                )

        self._use_mask = value

    @property
    def custom_mask(self) -> Optional[np.ndarray]:
        """
        Get the custom mask for missing values.

        :return: Custom mask array or None if not set
        :rtype: Optional[np.ndarray]
        """
        return self._custom_mask

    @custom_mask.setter
    def custom_mask(self, value: Optional[np.ndarray]) -> None:
        """
        Set the custom mask for missing values.

        :param value: Custom mask array or None to disable custom masking
        :type value: Optional[np.ndarray]
        :return: None
        :rtype: None
        :raises ValueError: If mask format is invalid or doesn't match data shape
        """
        if self._use_mask and value is not None and self.id_data is not None:
            if isinstance(self.id_data, tuple) and isinstance(self.id_data_mask, tuple):
                for id_d, id_m in zip(self.id_data, self.id_data_mask):
                    if (id_d != id_m).any():
                        raise ValueError("The mask must have the same IDs as the data.")
            elif (self.id_data_mask != self.id_data).any():
                raise ValueError("The mask must have the same IDs as the data.")

        if value is not None:
            if isinstance(self._data, tuple) and (
                not isinstance(value, tuple) or len(value) != 3
            ):
                raise ValueError(
                    "If data is a tuple, custom_mask must also be a tuple of the same length (train, val, test)."
                )

            if not isinstance(self._data, tuple) and isinstance(
                self.custom_mask, tuple
            ):
                raise ValueError(
                    "If data is a single array, custom_mask cannot be a tuple."
                )

            if isinstance(value, tuple):
                if (
                    value[0].shape != self._data[0].shape
                    or value[1].shape != self._data[1].shape
                    or value[2].shape != self._data[2].shape
                ):
                    raise ValueError(
                        "Each element of custom_mask must have the same shape as its corresponding dataset "
                        "(mask_train with x_train, mask_val with x_val, mask_test with x_test)."
                    )
            else:
                if value.shape != self._data.shape:
                    raise ValueError(
                        "custom_mask must have the same shape as the original input data before transformation"
                    )

        self._custom_mask = value

    @property
    def mask_train(self) -> np.ndarray:
        """
        Get the training mask for missing values.

        :return: Training mask array
        :rtype: np.ndarray
        """
        return self._mask_train

    @mask_train.setter
    def mask_train(self, value: np.ndarray) -> None:
        """
        Set the training mask for missing values.

        :param value: Training mask array
        :type value: np.ndarray
        :return: None
        :rtype: None
        :raises ValueError: If mask shape doesn't match training data shape
        """
        if hasattr(value, "shape"):
            if value.shape != self.x_train.shape:
                raise ValueError(
                    "mask_train must have the same shape as x_train after transformation."
                )
        self._mask_train = value

    @property
    def mask_val(self) -> np.ndarray:
        """
        Get the validation mask for missing values.

        :return: Validation mask array
        :rtype: np.ndarray
        """
        return self._mask_val

    @mask_val.setter
    def mask_val(self, value: np.ndarray) -> None:
        """
        Set the validation mask for missing values.

        :param value: Validation mask array
        :type value: np.ndarray
        :return: None
        :rtype: None
        :raises ValueError: If mask shape doesn't match validation data shape
        """
        if hasattr(value, "shape"):
            if value.shape != self.x_val.shape:
                raise ValueError(
                    "mask_val must have the same shape as x_val after transformation."
                )
        self._mask_val = value

    @property
    def mask_test(self) -> np.ndarray:
        """
        Get the test mask for missing values.

        :return: Test mask array
        :rtype: np.ndarray
        """
        return self._mask_test

    @mask_test.setter
    def mask_test(self, value: np.ndarray) -> None:
        """
        Set the test mask for missing values.

        :param value: Test mask array
        :type value: np.ndarray
        :return: None
        :rtype: None
        :raises ValueError: If mask shape doesn't match test data shape
        """
        if hasattr(value, "shape"):
            if value.shape != self.x_test.shape:
                raise ValueError(
                    "mask_test must have the same shape as x_test after transformation."
                )
        self._mask_test = value

    @property
    def imputer(self) -> Optional[DataImputer]:
        """
        Get the data imputer used for handling missing values.

        :return: DataImputer instance or None if not set
        :rtype: Optional[DataImputer]
        """
        return self._imputer

    @imputer.setter
    def imputer(self, value: Optional[DataImputer]) -> None:
        """
        Set the data imputer for handling missing values.

        :param value: DataImputer instance or None to disable imputation
        :type value: Optional[DataImputer]
        :return: None
        :rtype: None
        """
        self._imputer = value

    @property
    def shuffle(self) -> bool:
        """
        Get the shuffle flag indicating whether training data should be shuffled.

        :return: True if shuffling is enabled, False otherwise
        :rtype: bool
        """
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value: bool) -> None:
        """
        Set the shuffle flag for training data.

        :param value: True to enable shuffling of training data, False to disable
        :type value: bool
        :return: None
        :rtype: None
        """
        self._shuffle = value

    @property
    def shuffle_buffer_size(self) -> Optional[int]:
        """
        Get the shuffle buffer size for training data shuffling.

        :return: Buffer size for shuffling or None if not set
        :rtype: Optional[int]
        """
        return self._shuffle_buffer_size

    @shuffle_buffer_size.setter
    def shuffle_buffer_size(self, value: Optional[int]) -> None:
        """
        Set the shuffle buffer size for training data shuffling.

        :param value: Buffer size for shuffling (None to disable or use default)
        :type value: Optional[int]
        :return: None
        :rtype: None
        :raises ValueError: If value is not a positive integer
        """
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("shuffle_buffer_size must be a positive integer.")

        self._shuffle_buffer_size = value

    @property
    def x_train_no_shuffle(self) -> np.ndarray:
        """
        Get the unshuffled training data for reconstruction purposes.

        :return: Training data without shuffling applied
        :rtype: np.ndarray
        """
        if self._x_train_no_shuffle is None:
            return self.x_train
        return self._x_train_no_shuffle

    @x_train_no_shuffle.setter
    def x_train_no_shuffle(self, value: np.ndarray) -> None:
        """
        Set the unshuffled training data for reconstruction purposes.

        :param value: Training data without shuffling applied
        :type value: np.ndarray
        :return: None
        :rtype: None
        """
        self._x_train_no_shuffle = value

    @property
    def checkpoint(self) -> int:
        """
        Get the checkpoint value.

        :return: Number of epochs between checkpoints (0 to disable)
        :rtype: int
        """
        return getattr(self, "_checkpoint", 0)

    @checkpoint.setter
    def checkpoint(self, value: int) -> None:
        """
        Set the checkpoint value.

        :param value: Number of epochs between checkpoints (0 to disable)
        :type value: int
        :raises ValueError: If value is negative
        """
        if not isinstance(value, int):
            raise ValueError("checkpoint must be an integer")
        if value < 0:
            raise ValueError("checkpoint cannot be negative")
        self._checkpoint = value

    @property
    def model_optimizer(
        self,
    ) -> Union[Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]:
        """
        Get the model's optimizer.

        :return: The optimizer instance
        :rtype: Union[Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
        """
        return self._model_optimizer

    @model_optimizer.setter
    def model_optimizer(
        self, value: Union[Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
    ) -> None:
        """
        Set the model's optimizer.

        :param value: The optimizer instance or name
        :type value: Union[Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, str]
        :return: None
        :rtype: None
        :raises ValueError: If optimizer_name is not a valid optimizer
        """
        if isinstance(value, str):
            optimizers = {
                "adam": Adam(),
                "sgd": SGD(),
                "rmsprop": RMSprop(),
                "adagrad": Adagrad(),
                "adadelta": Adadelta(),
                "adamax": Adamax(),
                "nadam": Nadam(),
            }

            if value.lower() not in optimizers:
                raise ValueError(
                    f"Invalid optimizer '{value}'. Choose from {list(optimizers.keys())}."
                )

            self._model_optimizer = optimizers[value.lower()]
        else:
            self._model_optimizer = value

    @classmethod
    def load_from_pickle(cls, path: str) -> "AutoEncoder":
        """
        Load an AutoEncoder model from a pickle file.

        :param path: Path to the pickle file containing the saved model
        :type path: str
        :return: An instance of AutoEncoder with loaded parameters
        :rtype: AutoEncoder
        :raises FileNotFoundError: If the pickle file does not exist
        :raises ValueError: If the pickle file format is invalid
        :raises RuntimeError: If there's an error loading the model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pickle file not found: {path}")

        try:
            # Load the model and parameters from the pickle file
            with open(path, "rb") as f:
                saved_data = pickle.load(f)

            if "model" not in saved_data or "params" not in saved_data:
                raise ValueError("Invalid pickle file format: missing required keys.")

            model = saved_data["model"]
            params = saved_data["params"]

            # Create an instance of AutoEncoder
            instance = cls()

            # Assign loaded parameters
            instance.context_window = params.get("context_window")
            instance.time_step_to_check = params.get("time_step_to_check")
            instance.normalization_method = params.get("normalization_method")
            instance.features_name = params.get("features_name", None)
            instance.feature_to_check = params.get("feature_to_check", None)
            instance.normalization_values = params.get("normalization_values", {})
            instance.seed = params.get("seed", None)

            # Model must be the last element in the saved data
            instance.model = model

            logger.info(f"Model successfully loaded from {path}")

            return instance

        except Exception as e:
            raise RuntimeError(f"Error loading the AutoEncoder model: {e}") from e

    @staticmethod
    def create_folder_structure(folder_structure: List[str]) -> None:
        """
        Create a folder structure if it does not exist.

        :param folder_structure: List of folder paths to create
        :type folder_structure: List[str]
        :return: None
        :rtype: None
        """
        for path in folder_structure:
            os.makedirs(path, exist_ok=True)

    def _create_datasets(self, batch_size: int) -> None:
        """
        Create training, validation and test datasets.

        :param batch_size: Size of batches for training
        :type batch_size: int
        :return: None
        :rtype: None
        """
        if self._use_mask:
            train_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=(self.x_train, self.mask_train)
            )
            val_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=(self.x_val, self.mask_val)
            )
            test_dataset = tf.data.Dataset.from_tensor_slices(
                tensors=(self.x_test, self.mask_test)
            )
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices(tensors=self.x_train)
            val_dataset = tf.data.Dataset.from_tensor_slices(tensors=self.x_val)
            test_dataset = tf.data.Dataset.from_tensor_slices(tensors=self.x_test)

        if self._shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=self._shuffle_buffer_size)

        self.train_dataset = train_dataset.cache().batch(batch_size=batch_size)
        self.val_dataset = val_dataset.cache().batch(batch_size=batch_size)
        self.test_dataset = test_dataset.cache().batch(batch_size=batch_size)

    def build_model(
        self,
        context_window: int,
        data: Union[
            np.ndarray,
            pd.DataFrame,
            pl.DataFrame,
            Tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
        time_step_to_check: Union[int, List[int]],
        feature_to_check: Union[int, List[int]],
        hidden_dim: Union[int, List[int]],
        form: str = "lstm",
        bidirectional_encoder: bool = False,
        bidirectional_decoder: bool = False,
        activation_encoder: Optional[str] = None,
        activation_decoder: Optional[str] = None,
        normalize: bool = False,
        normalization_method: str = "minmax",
        optimizer: str = "adam",
        batch_size: int = 32,
        save_path: Optional[str] = None,
        verbose: bool = False,
        feature_names: Optional[List[str]] = None,
        feature_weights: Optional[List[float]] = None,
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None,
        use_mask: bool = False,
        custom_mask: Any = None,
        imputer: Optional[DataImputer] = None,
        train_size: float = TRAIN_SIZE,
        val_size: float = VAL_SIZE,
        test_size: float = TEST_SIZE,
        id_columns: Union[str, int, List[str], List[int], None] = None,
        use_post_decoder_dense: bool = False,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Build the Autoencoder model with specified configuration.

        :param context_window: Size of the context window for sequence transformation
        :type context_window: int
        :param form: Type of encoder architecture to use
        :type form: str
        :param data: Input data for model training. Can be:
            * A single numpy array/pandas DataFrame for automatic train/val/test split
            * A tuple of three arrays/DataFrames for predefined splits
        :type data: Union[np.ndarray, pd.DataFrame, pl.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        :param time_step_to_check: Index or indices of time steps to check in prediction
        :type time_step_to_check: Union[int, List[int]]
        :param feature_to_check: Index or indices of features to check in prediction
        :type feature_to_check: Union[int, List[int]]
        :param hidden_dim: Dimensions of hidden layers. Can be single int or list of ints
        :type hidden_dim: Union[int, List[int]]
        :param bidirectional_encoder: Whether to use bidirectional layers in encoder
        :type bidirectional_encoder: bool
        :param bidirectional_decoder: Whether to use bidirectional layers in decoder
        :type bidirectional_decoder: bool
        :param activation_encoder: Activation function for encoder layers
        :type activation_encoder: Optional[str]
        :param activation_decoder: Activation function for decoder layers
        :type activation_decoder: Optional[str]
        :param normalize: Whether to normalize input data
        :type normalize: bool
        :param normalization_method: Method for data normalization ('minmax' or 'zscore')
        :type normalization_method: str
        :param optimizer: Name of optimizer to use for training
        :type optimizer: str
        :param batch_size: Size of batches for training
        :type batch_size: int
        :param save_path: Directory path to save model checkpoints
        :type save_path: Optional[str]
        :param verbose: Whether to print detailed information during training
        :type verbose: bool
        :param feature_names: Custom names for features
        :type feature_names: Optional[List[str]]
        :param feature_weights: Weights for each feature in loss calculation
        :type feature_weights: Optional[List[float]]
        :param shuffle: Whether to shuffle training data
        :type shuffle: bool
        :param shuffle_buffer_size: Size of buffer for shuffling
        :type shuffle_buffer_size: Optional[int]
        :param use_mask: Whether to use masking for missing values
        :type use_mask: bool
        :param custom_mask: Custom mask for missing values
        :type custom_mask: Any
        :param imputer: Instance of DataImputer for handling missing values
        :type imputer: Optional[DataImputer]
        :param train_size: Proportion of data for training (0-1)
        :type train_size: float
        :param val_size: Proportion of data for validation (0-1)
        :type val_size: float
        :param test_size: Proportion of data for testing (0-1)
        :type test_size: float
        :param id_columns: Column(s) to use for grouping data
        :type id_columns: Union[str, int, List[str], List[int], None]
        :param use_post_decoder_dense: Whether to add dense layer after decoder
        :type use_post_decoder_dense: bool
        :param seed: Seed for reproducibility (sets all random generators)
        :type seed: Optional[int]
        :raises NotImplementedError: If form='dense' is specified
        :raises ValueError: If invalid parameters are provided
        :return: None
        :rtype: None

        Example:
            >>> autoencoder = AutoEncoder()
            >>> data = np.random.randn(1000, 5)  # 1000 samples, 5 features
            >>> autoencoder.build_model(
            ...     context_window=10,
            ...     data=data,
            ...     time_step_to_check=[5, 7],
            ...     feature_to_check=[0, 1, 2],
            ...     hidden_dim=[64, 32],
            ...     form="lstm",
            ...     normalize=True
            ... )

        """
        self.seed = seed
        self.form = form
        self.save_path = save_path
        self.context_window = context_window
        self.time_step_to_check = time_step_to_check
        self.feature_to_check = feature_to_check
        self.normalization_method = normalization_method
        self.normalize = normalize
        self.hidden_dim = hidden_dim
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.verbose = verbose
        self.feature_weights = feature_weights
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.use_mask = use_mask
        self.imputer = imputer
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size

        # Extract names and convert data to numpy
        self.data, extracted_feature_names = processing.convert_data_to_numpy(data=data)
        self.features_name = (
            feature_names
            or extracted_feature_names
            or [
                f"feature_{i}"
                for i in range(
                    self._data[0].shape[1]
                    if isinstance(self._data, tuple)
                    else self._data.shape[1]
                )
            ]
        )

        (self.data, self.id_data, self.id_data_dict, self.id_columns_indices) = (
            processing.handle_id_columns(
                data=self._data,
                id_columns=id_columns,
                features_name=self._features_name,
                context_window=self._context_window,
            )
        )

        if self._use_mask and custom_mask is not None:
            custom_mask, _ = processing.convert_data_to_numpy(custom_mask)
            custom_mask, self.id_data_mask, self.id_data_dict_mask, _ = (
                processing.handle_id_columns(
                    data=custom_mask,
                    id_columns=id_columns,
                    features_name=self._features_name,
                    context_window=self._context_window,
                )
            )
            self.custom_mask = custom_mask

        if self.id_data_dict:
            self.concatenate_by_id()
        else:
            self.prepare_datasets(
                data=self._data,
                context_window=self._context_window,
                normalize=self._normalize,
            )

        if self._shuffle and self._shuffle_buffer_size is None:
            self.shuffle_buffer_size = len(self.x_train)

        self.x_train_no_shuffle = np.copy(self.x_train)

        #########################################################
        ################# BUILD MODEL ###########################
        #########################################################
        self.num_layers = len(self._hidden_dim)
        self.layers = [
            encoder(
                form=self._form,
                context_window=self._context_window,
                features=self.x_train.shape[2],
                hidden_dim=self._hidden_dim,
                num_layers=self.num_layers,
                use_bidirectional=self._bidirectional_encoder,
                activation=self._activation_encoder,
                verbose=self._verbose,
            ),
            decoder(
                form=self._form,
                context_window=self._context_window,
                features=len(self._feature_to_check),
                hidden_dim=self._hidden_dim,
                num_layers=self.num_layers,
                use_bidirectional=self._bidirectional_decoder,
                activation=self._activation_decoder,
                verbose=self._verbose,
            ),
        ]

        if use_post_decoder_dense:
            self.layers.append(
                Dense(len(self._feature_to_check), name="post_decoder_dense")
            )

        self.model = Sequential(self.layers, name="autoencoder")
        self.model.build()
        self.model_optimizer = optimizer

        self._create_datasets(batch_size)

    def train(
        self,
        epochs: int = 100,
        checkpoint: int = 10,
        use_early_stopping: bool = True,
        patience: int = 10,
    ) -> None:
        """
        Train the model using the train and validation datasets and save the best model.

        :param epochs: Number of epochs to train the model
        :type epochs: int
        :param checkpoint: Number of epochs to save a checkpoint (0 to disable)
        :type checkpoint: int
        :param use_early_stopping: Whether to use early stopping or not
        :type use_early_stopping: bool
        :param patience: Number of epochs to wait before stopping the training
        :type patience: int
        :return: None
        :rtype: None
        """
        self.last_epoch = 0
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.train_loss_history = None
        self.val_loss_history = None
        self.use_early_stopping = use_early_stopping
        self.patience = patience

        # Define training functions
        @tf.function
        def forward_pass(
            x: tf.Tensor, mask: Optional[tf.Tensor] = None
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            """
            Perform a forward pass through the model.

            :param x: Input data
            :type x: tf.Tensor
            :param mask: Optional binary mask for missing values
            :type mask: Optional[tf.Tensor]
            :return: Tuple of (loss, x_hat)
            :rtype: Tuple[tf.Tensor, tf.Tensor]
            """
            x = tf.cast(x, tf.float32)

            hx = self.model.get_layer(f"{self._form}_encoder")(x)
            x_hat = self.model.get_layer(f"{self._form}_decoder")(hx)

            if "post_decoder_dense" in [layer.name for layer in self.model.layers]:
                x_hat = self.model.get_layer("post_decoder_dense")(x_hat)

            # Gather all required time steps
            x_real = tf.gather(x, self._time_step_to_check, axis=1)
            x_real = tf.gather(x_real, self._feature_to_check, axis=2)

            x_pred = tf.expand_dims(x_hat, axis=1)

            # Calculate mean loss across all selected points
            loss = self.masked_weighted_mse(
                y_true=x_real,
                y_pred=x_pred,
                feature_weights=self._feature_weights,
                feature_to_check=self._feature_to_check,
                time_step_to_check=self._time_step_to_check,
                mask=mask,
            )

            return loss, x_hat

        @tf.function
        def train_step(x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
            """
            Single step for model training.

            :param x: Input data
            :type x: tf.Tensor
            :param mask: Optional binary mask for missing values
            :type mask: Optional[tf.Tensor]
            :return: Loss value
            :rtype: tf.Tensor
            """
            with tf.GradientTape() as tape:
                loss, _ = forward_pass(x=x, mask=mask)

            autoencoder_gradient = tape.gradient(loss, self.model.trainable_variables)
            self.model_optimizer.apply_gradients(
                grads_and_vars=zip(autoencoder_gradient, self.model.trainable_variables)
            )

            return loss

        @tf.function
        def validation_step(
            x: tf.Tensor, mask: Optional[tf.Tensor] = None
        ) -> tf.Tensor:
            """
            Single step for model validation.

            :param x: Input data
            :type x: tf.Tensor
            :param mask: Optional binary mask for missing values
            :type mask: Optional[tf.Tensor]
            :return: Loss value
            :rtype: tf.Tensor
            """
            loss, _ = forward_pass(x=x, mask=mask)
            return loss

        # Run training loop
        self._run_training_loop(
            train_step=train_step,
            validation_step=validation_step,
            epochs=epochs,
            checkpoint=checkpoint,
            use_early_stopping=use_early_stopping,
            patience=patience,
        )

    def _run_training_loop(
        self,
        train_step: callable,
        validation_step: callable,
        epochs: int,
        checkpoint: int,
        use_early_stopping: bool,
        patience: int,
    ) -> None:
        """
        Run the training loop for the model.

        :param train_step: Function to perform a training step
        :type train_step: callable
        :param validation_step: Function to perform a validation step
        :type validation_step: callable
        :param epochs: Number of epochs to train
        :type epochs: int
        :param checkpoint: Number of epochs between checkpoints
        :type checkpoint: int
        :param use_early_stopping: Whether to use early stopping
        :type use_early_stopping: bool
        :param patience: Number of epochs to wait before early stopping
        :type patience: int
        """
        # Lists to store loss history
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Training loop
            avg_train_loss = self._run_epoch_training(train_step)
            train_loss_history.append(avg_train_loss)

            # Validation loop
            avg_val_loss = self._run_epoch_validation(validation_step)
            val_loss_history.append(avg_val_loss)

            self.last_epoch = epoch

            # Early stopping logic
            if use_early_stopping:
                should_stop, best_val_loss, patience_counter = (
                    self._check_early_stopping(
                        epoch=epoch,
                        avg_val_loss=avg_val_loss,
                        best_val_loss=best_val_loss,
                        patience_counter=patience_counter,
                        patience=patience,
                    )
                )
                if should_stop:
                    break

            # Checkpoint logic
            if epoch % checkpoint == 0:
                self._handle_checkpoint(epoch, avg_train_loss, avg_val_loss)

        # Store the loss history in the model instance
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

        # Plot loss history
        plots.plot_loss_history(
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            save_path=os.path.join(self._save_path, "plots"),
        )

        self.save(filename=f"{self.last_epoch}.pkl")

    def _run_epoch_training(self, train_step: callable) -> float:
        """
        Run a single epoch of training.

        :param train_step: Function to perform a training step
        :type train_step: callable
        :return: Average training loss for the epoch
        :rtype: float
        """
        epoch_train_losses = []
        for batch in self.train_dataset:
            if self._use_mask:
                data, mask = batch
            else:
                data = batch
                mask = None

            loss = train_step(x=data, mask=mask)
            epoch_train_losses.append(float(loss))

        # Calculate average training loss for the epoch
        return sum(epoch_train_losses) / len(epoch_train_losses)

    def _run_epoch_validation(self, validation_step: callable) -> float:
        """
        Run a single epoch of validation.

        :param validation_step: Function to perform a validation step
        :type validation_step: callable
        :return: Average validation loss for the epoch
        :rtype: float
        """
        epoch_val_losses = []
        for batch in self.val_dataset:
            if self._use_mask:
                data, mask = batch
            else:
                data = batch
                mask = None

            val_loss = validation_step(x=data, mask=mask)
            epoch_val_losses.append(float(val_loss))

        # Calculate average validation loss for the epoch
        return sum(epoch_val_losses) / len(epoch_val_losses)

    def _check_early_stopping(
        self,
        epoch: int,
        avg_val_loss: float,
        best_val_loss: float,
        patience_counter: int,
        patience: int,
    ) -> Tuple[bool, float, int]:
        """
        Check if early stopping should be applied.

        :param epoch: Current epoch
        :type epoch: int
        :param avg_val_loss: Average validation loss for the current epoch
        :type avg_val_loss: float
        :param best_val_loss: Best validation loss so far
        :type best_val_loss: float
        :param patience_counter: Counter for patience
        :type patience_counter: int
        :param patience: Number of epochs to wait before early stopping
        :type patience: int
        :return: Tuple of (should_stop, best_val_loss, patience_counter)
        :rtype: Tuple[bool, float, int]
        """
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            self.save(filename="best_model.pkl")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} | Best Validation Loss: {best_val_loss:.6f}"
            )
            return True, best_val_loss, patience_counter

        return False, best_val_loss, patience_counter

    def _handle_checkpoint(
        self, epoch: int, avg_train_loss: float, avg_val_loss: float
    ) -> None:
        """
        Handle checkpoint saving and logging.

        :param epoch: Current epoch
        :type epoch: int
        :param avg_train_loss: Average training loss for the current epoch
        :type avg_train_loss: float
        :param avg_val_loss: Average validation loss for the current epoch
        :type avg_val_loss: float
        """
        if self._verbose:
            logger.info(
                f"Epoch {epoch:4d} | "
                f"Training Loss: {avg_train_loss:.6f} | "
                f"Validation Loss: {avg_val_loss:.6f}"
            )

        self.save(filename=f"{epoch}.pkl")

    def _create_data_points_df(
        self,
        x_converted: np.ndarray,
        x_hat: np.ndarray,
        feature_labels: List[str],
        train_split: int,
        val_split: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a DataFrame containing all data points for plotting.

        :param x_converted: Original data array
        :type x_converted: np.ndarray
        :param x_hat: Reconstructed data array
        :type x_hat: np.ndarray
        :param feature_labels: List of feature names
        :type feature_labels: List[str]
        :param train_split: Index where training data ends
        :type train_split: int
        :param val_split: Index where validation data ends
        :type val_split: int
        :return: DataFrames for actual and reconstructed data points
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """
        data_points_actual = []
        data_points_reconstructed = []

        # Process data with IDs if provided
        if self.id_data_dict:
            # Initialize current position in x_converted and x_hat
            current_pos = 0
            # Initialize time step in respective datasets
            time_step = {id_: 0 for id_ in self.length_datasets.keys()}
            for data_split in ["train", "validation", "test"]:
                for id_value in sorted(self.length_datasets.keys()):
                    # Get length of dataset
                    split_len = self.length_datasets[id_value][data_split]
                    for i in range(split_len):
                        t = time_step[id_value]
                        for feature_idx, feature_name in enumerate(feature_labels):
                            data_points_actual.append(
                                {
                                    "id": id_value,
                                    "feature": feature_name,
                                    "time_step": t,
                                    "value": x_converted[feature_idx, current_pos],
                                    "data_split": data_split,
                                }
                            )
                            data_points_reconstructed.append(
                                {
                                    "id": id_value,
                                    "feature": feature_name,
                                    "time_step": t,
                                    "value": x_hat[feature_idx, current_pos],
                                    "data_split": data_split,
                                }
                            )
                        current_pos = current_pos + 1
                        time_step[id_value] = time_step[id_value] + 1

        else:
            # Process data without IDs
            current_pos = 0
            split_len_dic = {
                "train": train_split,
                "validation": val_split - train_split,
                "test": x_converted.shape[1] - val_split,
            }
            for data_split in ["train", "validation", "test"]:
                split_len = split_len_dic[data_split]
                for i in range(split_len):
                    for feature_idx, feature_name in enumerate(feature_labels):
                        data_points_actual.append(
                            {
                                "feature": feature_name,
                                "time_step": current_pos,
                                "value": x_converted[feature_idx, current_pos],
                                "data_split": data_split,
                            }
                        )
                        data_points_reconstructed.append(
                            {
                                "feature": feature_name,
                                "time_step": current_pos,
                                "value": x_hat[feature_idx, current_pos],
                                "data_split": data_split,
                            }
                        )
                    current_pos = current_pos + 1

        if current_pos != x_converted.shape[1]:
            raise ValueError(
                f"Indices to create data points are misaligned."
                f"Expected {x_converted.shape[1]} but got {current_pos}"
            )

        return pd.DataFrame(data_points_actual), pd.DataFrame(data_points_reconstructed)

    def reconstruct(
        self,
        save_path: Optional[str] = None,
        reconstruction_diagnostic: bool = False,
    ) -> pd.DataFrame:
        """
        Reconstruct the data using the trained model and plot the actual and reconstructed values.

        :param save_path: Path to save reconstruction results, plots, and diagnostics
        :type save_path: Optional[str]
        :param reconstruction_diagnostic: If True, shows and optionally saves reconstruction error data and plots
        :type reconstruction_diagnostic: bool
        :return: Reconstruction results
        :rtype: pd.DataFrame
        """
        if self.x_train_no_shuffle is None:
            raise ValueError(
                "self.x_train_no_shuffle is None so reconstruction on training data can't be done. "
                "recontruct() can only be done after build_model() and train_autoencoder() have been run."
            )

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
            self.x_train[:, self._time_step_to_check, self._feature_to_check]
        )
        x_val_converted = np.copy(
            self.x_val[:, self._time_step_to_check, self._feature_to_check]
        )
        x_test_converted = np.copy(
            self.x_test[:, self._time_step_to_check, self._feature_to_check]
        )

        # Handle denormalization if normalization was applied
        if self._normalize:
            # Use the global normalization values if ID-based normalization wasn't used
            if "global" in self.normalization_values:
                norm_values = self.normalization_values["global"]

                # Denormalize predictions
                x_hat_train = processing.denormalize_data(
                    data=x_hat_train,
                    normalization_method=self._normalization_method,
                    min_x=norm_values["min_x"][self._feature_to_check],
                    max_x=norm_values["max_x"][self._feature_to_check],
                    mean_=(
                        norm_values["mean_"][self._feature_to_check]
                        if "mean_" in norm_values
                        else None
                    ),
                    std_=(
                        norm_values["std_"][self._feature_to_check]
                        if "std_" in norm_values
                        else None
                    ),
                )
                x_hat_val = processing.denormalize_data(
                    data=x_hat_val,
                    normalization_method=self._normalization_method,
                    min_x=norm_values["min_x"][self._feature_to_check],
                    max_x=norm_values["max_x"][self._feature_to_check],
                    mean_=(
                        norm_values["mean_"][self._feature_to_check]
                        if "mean_" in norm_values
                        else None
                    ),
                    std_=(
                        norm_values["std_"][self._feature_to_check]
                        if "std_" in norm_values
                        else None
                    ),
                )
                x_hat_test = processing.denormalize_data(
                    data=x_hat_test,
                    normalization_method=self._normalization_method,
                    min_x=norm_values["min_x"][self._feature_to_check],
                    max_x=norm_values["max_x"][self._feature_to_check],
                    mean_=(
                        norm_values["mean_"][self._feature_to_check]
                        if "mean_" in norm_values
                        else None
                    ),
                    std_=(
                        norm_values["std_"][self._feature_to_check]
                        if "std_" in norm_values
                        else None
                    ),
                )

                # Denormalize original data
                x_train_converted = processing.denormalize_data(
                    data=x_train_converted,
                    normalization_method=self._normalization_method,
                    min_x=norm_values["min_x"][self._feature_to_check],
                    max_x=norm_values["max_x"][self._feature_to_check],
                    mean_=(
                        norm_values["mean_"][self._feature_to_check]
                        if "mean_" in norm_values
                        else None
                    ),
                    std_=(
                        norm_values["std_"][self._feature_to_check]
                        if "std_" in norm_values
                        else None
                    ),
                )
                x_val_converted = processing.denormalize_data(
                    data=x_val_converted,
                    normalization_method=self._normalization_method,
                    min_x=norm_values["min_x"][self._feature_to_check],
                    max_x=norm_values["max_x"][self._feature_to_check],
                    mean_=(
                        norm_values["mean_"][self._feature_to_check]
                        if "mean_" in norm_values
                        else None
                    ),
                    std_=(
                        norm_values["std_"][self._feature_to_check]
                        if "std_" in norm_values
                        else None
                    ),
                )
                x_test_converted = processing.denormalize_data(
                    data=x_test_converted,
                    normalization_method=self._normalization_method,
                    min_x=norm_values["min_x"][self._feature_to_check],
                    max_x=norm_values["max_x"][self._feature_to_check],
                    mean_=(
                        norm_values["mean_"][self._feature_to_check]
                        if "mean_" in norm_values
                        else None
                    ),
                    std_=(
                        norm_values["std_"][self._feature_to_check]
                        if "std_" in norm_values
                        else None
                    ),
                )

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
                    val_length = self.length_datasets[id_key]["validation"]
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

                    # Denormalize data for this ID
                    id_x_hat_train = processing.denormalize_data(
                        data=id_x_hat_train,
                        normalization_method=self._normalization_method,
                        min_x=norm_values["min_x"][self._feature_to_check],
                        max_x=norm_values["max_x"][self._feature_to_check],
                        mean_=(
                            norm_values["mean_"][self._feature_to_check]
                            if "mean_" in norm_values
                            else None
                        ),
                        std_=(
                            norm_values["std_"][self._feature_to_check]
                            if "std_" in norm_values
                            else None
                        ),
                    )
                    id_x_hat_val = processing.denormalize_data(
                        data=id_x_hat_val,
                        normalization_method=self._normalization_method,
                        min_x=norm_values["min_x"][self._feature_to_check],
                        max_x=norm_values["max_x"][self._feature_to_check],
                        mean_=(
                            norm_values["mean_"][self._feature_to_check]
                            if "mean_" in norm_values
                            else None
                        ),
                        std_=(
                            norm_values["std_"][self._feature_to_check]
                            if "std_" in norm_values
                            else None
                        ),
                    )
                    id_x_hat_test = processing.denormalize_data(
                        data=id_x_hat_test,
                        normalization_method=self._normalization_method,
                        min_x=norm_values["min_x"][self._feature_to_check],
                        max_x=norm_values["max_x"][self._feature_to_check],
                        mean_=(
                            norm_values["mean_"][self._feature_to_check]
                            if "mean_" in norm_values
                            else None
                        ),
                        std_=(
                            norm_values["std_"][self._feature_to_check]
                            if "std_" in norm_values
                            else None
                        ),
                    )

                    id_x_train = processing.denormalize_data(
                        data=id_x_train,
                        normalization_method=self._normalization_method,
                        min_x=norm_values["min_x"][self._feature_to_check],
                        max_x=norm_values["max_x"][self._feature_to_check],
                        mean_=(
                            norm_values["mean_"][self._feature_to_check]
                            if "mean_" in norm_values
                            else None
                        ),
                        std_=(
                            norm_values["std_"][self._feature_to_check]
                            if "std_" in norm_values
                            else None
                        ),
                    )
                    id_x_val = processing.denormalize_data(
                        data=id_x_val,
                        normalization_method=self._normalization_method,
                        min_x=norm_values["min_x"][self._feature_to_check],
                        max_x=norm_values["max_x"][self._feature_to_check],
                        mean_=(
                            norm_values["mean_"][self._feature_to_check]
                            if "mean_" in norm_values
                            else None
                        ),
                        std_=(
                            norm_values["std_"][self._feature_to_check]
                            if "std_" in norm_values
                            else None
                        ),
                    )
                    id_x_test = processing.denormalize_data(
                        data=id_x_test,
                        normalization_method=self._normalization_method,
                        min_x=norm_values["min_x"][self._feature_to_check],
                        max_x=norm_values["max_x"][self._feature_to_check],
                        mean_=(
                            norm_values["mean_"][self._feature_to_check]
                            if "mean_" in norm_values
                            else None
                        ),
                        std_=(
                            norm_values["std_"][self._feature_to_check]
                            if "std_" in norm_values
                            else None
                        ),
                    )

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
            for i, feature in enumerate(self._features_name)
            if i not in self.id_columns_indices
        ]
        feature_labels = (
            [features_names_without_id[i] for i in self._feature_to_check]
            if hasattr(self, "features_name")
            else None
        )

        # Get the split indices
        train_split = self.x_train.shape[0]
        val_split = train_split + self.x_val.shape[0]

        # Create DataFrame with all data points
        df_actual, df_reconstructed = self._create_data_points_df(
            x_converted=x_converted,
            x_hat=x_hat,
            feature_labels=feature_labels,
            train_split=train_split,
            val_split=val_split,
        )

        # Define save_path
        if save_path is not None:
            save_path = os.path.join(save_path, "reconstruct")

        # Display and save reconstruction errors
        if reconstruction_diagnostic:
            # Check if there are id_columns through id_data_dict
            if self.id_data_dict:
                for id_i in df_actual.id.unique().tolist():
                    # Get appropriate actual and reconstructed data based on id
                    df_actual_i = processing.id_pivot(df=df_actual, id=id_i)
                    df_reconstructed_i = processing.id_pivot(
                        df=df_reconstructed, id=id_i
                    )

                    # Reintroduce nulls from original data (before imputation)
                    df_actual_i = processing.reintroduce_nans(
                        self, df=df_actual_i, id=id_i
                    )

                    # Calculate reconstruction error and other metrics
                    reconstruction_error_df = anomaly_detector.reconstruction_error(
                        actual_data_df=df_actual_i,
                        autoencoder_output_df=df_reconstructed_i,
                        save_path=save_path,
                        filename=f"{id_i}_reconstruction_error.csv",
                    )
                    anomaly_detector.reconstruction_error_summary(
                        reconstruction_error_df,
                        save_path=save_path,
                        filename=f"{id_i}_reconstruction_error_summary.csv",
                    )
                    plots.boxplot_reconstruction_error(
                        reconstruction_error_df,
                        save_path=save_path,
                        filename=f"{id_i}_reconstruction_error_boxplot.html",
                        show=True,
                    )
            else:
                id_i = "global"
                df_reconstructed = df_reconstructed.copy()
                df_actual = df_actual.copy()
                df_reconstructed["id"] = "global"
                df_actual["id"] = "global"

                # Get appropriate actual and reconstructed data based on id
                df_actual_i = processing.id_pivot(df=df_actual, id=id_i)
                df_reconstructed_i = processing.id_pivot(df=df_reconstructed, id=id_i)

                # Reintroduce nulls from original data (before imputation)
                df_actual_i = processing.reintroduce_nans(self, df=df_actual_i, id=id_i)

                # Calculate reconstruction error and other metrics
                reconstruction_error_df = anomaly_detector.reconstruction_error(
                    actual_data_df=df_actual_i,
                    autoencoder_output_df=df_reconstructed_i,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_error.csv",
                )
                anomaly_detector.reconstruction_error_summary(
                    reconstruction_error_df,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_error_summary.csv",
                )
                plots.boxplot_reconstruction_error(
                    reconstruction_error_df,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_error_boxplot.html",
                    show=True,
                )

        # Save reconstruction if save_path is provided
        if save_path is not None:
            if "id" in df_reconstructed.columns:
                ids = df_reconstructed.id.unique().tolist()
            else:
                ids = ["global"]
                df_reconstructed = df_reconstructed.copy()
                df_reconstructed["id"] = "global"
            for id_i in ids:
                df_reconstructed_i = processing.id_pivot(df=df_reconstructed, id=id_i)
                processing.save_csv(
                    df_reconstructed_i,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_results.csv",
                )
            # Plot the data
            plots.plot_actual_and_reconstructed(
                df_actual=df_actual,
                df_reconstructed=df_reconstructed,
                save_path=os.path.join(save_path, "plots"),
                feature_labels=feature_labels,
            )

        return df_reconstructed

    def save(
        self, save_path: Optional[str] = None, filename: str = "model.pkl"
    ) -> None:
        """
        Save the model (Keras model + training parameters) into a single .pkl file.

        :param save_path: Path to save the model
        :type save_path: Optional[str]
        :param filename: Name of the file to save the model
        :type filename: str
        :raises Exception: If there's an error saving the model
        :return: None
        :rtype: None
        """
        try:
            save_path = save_path or self._save_path
            os.makedirs(os.path.join(save_path, "models"), exist_ok=True)

            model_path = os.path.join(save_path, "models", filename)

            training_params = {
                "context_window": self._context_window,
                "time_step_to_check": self._time_step_to_check,
                "normalization_method": (
                    self._normalization_method if self._normalize else None
                ),
                "normalization_values": {},
                "features_name": self._features_name,
                "feature_to_check": self._feature_to_check,
                "seed": self._seed,
            }

            if self._normalize:
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
        context_window: int,
        data: Union[
            np.ndarray,
            pd.DataFrame,
            pl.DataFrame,
            Tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
        time_step_to_check: Union[int, List[int]],
        feature_to_check: Union[int, List[int]],
        hidden_dim: Union[int, List[int]],
        form: str = "lstm",
        bidirectional_encoder: bool = False,
        bidirectional_decoder: bool = False,
        activation_encoder: Optional[str] = None,
        activation_decoder: Optional[str] = None,
        normalize: bool = False,
        normalization_method: str = "minmax",
        optimizer: str = "adam",
        batch_size: int = 32,
        save_path: Optional[str] = None,
        verbose: bool = False,
        feature_names: Optional[List[str]] = None,
        feature_weights: Optional[List[float]] = None,
        shuffle: bool = False,
        shuffle_buffer_size: Optional[int] = None,
        use_mask: bool = False,
        custom_mask: Any = None,
        imputer: Optional[DataImputer] = None,
        train_size: float = TRAIN_SIZE,
        val_size: float = VAL_SIZE,
        test_size: float = TEST_SIZE,
        id_columns: Union[str, int, List[str], List[int], None] = None,
        epochs: int = 100,
        checkpoint: int = 10,
        use_early_stopping: bool = True,
        patience: int = 10,
        use_post_decoder_dense: bool = False,
        seed: Optional[int] = 42,
    ) -> "AutoEncoder":
        """
        Build and train the Autoencoder model in a single step.

        This method combines the functionality of build_model() and train() methods,
        allowing for a more streamlined workflow.

        :param context_window: Context window for the model used to transform
            tabular data into sequence data (2D tensor to 3D tensor)
        :type context_window: int
        :param data: Data to train the model. It can be:
            * A single numpy array/pandas DataFrame for automatic train/val/test split
            * A tuple of three arrays/DataFrames for predefined splits
        :type data: Union[np.ndarray, pd.DataFrame, Tuple[np.ndarray, np.ndarray, np.ndarray]]
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
        :type activation_encoder: Optional[str]
        :param activation_decoder: Activation function for the decoder layers
        :type activation_decoder: Optional[str]
        :param normalize: Whether to normalize the data
        :type normalize: bool
        :param normalization_method: Method to normalize the data "minmax" or "zscore"
        :type normalization_method: str
        :param optimizer: Optimizer to use for training
        :type optimizer: str
        :param batch_size: Batch size for training
        :type batch_size: int
        :param save_path: Folder path to save model checkpoints
        :type save_path: Optional[str]
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
        :param use_post_decoder_dense: Whether to use a dense layer after the decoder
        :type use_post_decoder_dense: bool
        :param seed: Seed for reproducibility (sets all random generators)
        :type seed: Optional[int]
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
            use_post_decoder_dense=use_post_decoder_dense,
            seed=seed,
        )

        self.train(
            epochs=epochs,
            checkpoint=checkpoint,
            use_early_stopping=use_early_stopping,
            patience=patience,
        )

        return self

    def reconstruct_new_data(
        self,
        data: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        iterations: int = 1,
        id_columns: Optional[Union[str, int, List[str], List[int]]] = None,
        save_path: Optional[str] = None,
        reconstruction_diagnostic: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Predict and reconstruct unknown data, iterating over NaN values to improve predictions.
        Uses stored `context_window`, normalization parameters, and the trained model.

        :param data: Input data (numpy array or pandas DataFrame or polars DataFrame)
        :type data: Union[np.ndarray, pd.DataFrame]
        :param iterations: Number of reconstruction iterations (None = no iteration)
        :type iterations: int
        :param id_columns: Column(s) that define IDs to process reconstruction separately
        :type id_columns: Optional[Union[str, int, List[str], List[int]]]
        :param save_path: Path to save reconstruction results, plots, and diagnostics
        :type save_path: Optional[str]
        :param reconstruction_diagnostic: If True, shows and optionally saves reconstruction error data and plots
        :type reconstruction_diagnostic: bool
        :return: Dictionary with reconstructed data per ID (or "global" if no ID)
        :rtype: Dict[str, pd.DataFrame]
        :raises ValueError: If no model is loaded or if id_columns format is invalid
        """
        if self.model is None:
            raise ValueError(
                "No model loaded. Use load_from_pickle() before calling reconstruct_new_data()."
            )
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        if not isinstance(data, (np.ndarray, pd.DataFrame, pl.DataFrame)):
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                "Expect np.ndarray, pd.DataFrame, or pl.DataFrame"
            )

        data, feature_names = processing.convert_data_to_numpy(data=data)

        if feature_names == []:
            feature_count = data.shape[1]
            feature_names = [f"feature_{i}" for i in range(feature_count)]

        if self.features_name != feature_names:
            raise ValueError(
                f"Feature names in recontruct_new_data(): {feature_names} "
                f"do not match those from build_model(): {self.features_name}"
            )

        # Create features_names_to_check, excluding ID columns if they exist
        if id_columns is not None and feature_names:
            if isinstance(id_columns, (str, int)):
                id_columns = [id_columns]

            if not isinstance(id_columns, list):
                raise ValueError("id_columns must be a list of strings or integers")

            # Get indices of ID columns
            id_indices = [
                feature_names.index(col) if isinstance(col, str) else col
                for col in id_columns
                if isinstance(col, str) and col in feature_names or isinstance(col, int)
            ]
            # Remove ID columns from feature names
            feature_names_without_id = [
                name for i, name in enumerate(feature_names) if i not in id_indices
            ]
            # Filter out ID columns from features_names_to_check
            features_names_to_check = (
                [feature_names_without_id[i] for i in self._feature_to_check]
                if feature_names_without_id
                else None
            )
        else:
            if self.id_data is not None and len(self.id_data) > 0:
                raise ValueError(
                    "The input data contains more columns than expected, "
                    "but 'id_columns' was not provided. Please specify which columns "
                    "are identifiers using the 'id_columns' parameter."
                )
            features_names_to_check = (
                [feature_names[i] for i in self._feature_to_check]
                if feature_names
                else None
            )

        # Handle ID columns
        if id_columns is not None:
            data, _, id_data_dict, self.id_columns_indices = (
                processing.handle_id_columns(
                    data=data,
                    id_columns=id_columns,
                    features_name=feature_names,
                    context_window=self._context_window,
                )
            )
        else:
            id_data_dict = {"global": data}

        reconstructed_results = {}

        if id_columns is not None:
            for id_iter, data_id in id_data_dict.items():
                if len(data_id) < self._context_window:
                    raise ValueError(
                        f"{id_iter} has length {len(data_id)} but needs to be "
                        f"at least context window ({self._context_window}) in length"
                    )

                nan_positions_id = np.isnan(data_id)
                has_nans_id = np.any(nan_positions_id)

                reconstructed_results[id_iter] = self._reconstruct_single_dataset(
                    data=data_id,
                    feature_names=features_names_to_check,
                    nan_positions=nan_positions_id[:, self._feature_to_check],
                    has_nans=has_nans_id,
                    iterations=iterations,
                    id_iter=id_iter,
                    save_path=save_path,
                )
        else:
            if len(data) < self._context_window:
                raise ValueError(
                    f"Data has length {len(data)} but needs to be "
                    f"at least context window ({self._context_window}) in length"
                )

            nan_positions = np.isnan(data)
            has_nans = np.any(nan_positions)
            reconstructed_results["global"] = self._reconstruct_single_dataset(
                data=data,
                feature_names=features_names_to_check,
                nan_positions=nan_positions[:, self._feature_to_check],
                has_nans=has_nans,
                iterations=iterations,
                id_iter=None,
                save_path=save_path,
            )

        # Save reconstruction if save_path is provided
        if save_path is not None:
            save_path = os.path.join(save_path, "reconstruct_new_data")
            for id_i, autoencoder_output_df in reconstructed_results.items():
                processing.save_csv(
                    autoencoder_output_df,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_results.csv",
                )

        # Display and save reconstruction errors
        if reconstruction_diagnostic:
            # Define context offset
            initial_context_offset = self._time_step_to_check[0]
            ending_context_offset = self._context_window - 1 - initial_context_offset
            # use keys from reconstructed_results to get approriate data from id_data_dict
            for id_i, autoencoder_output_df in reconstructed_results.items():
                actual_data_array = id_data_dict[id_i][:, self._feature_to_check]
                actual_data_df = pd.DataFrame(
                    actual_data_array, columns=autoencoder_output_df.columns
                )
                actual_data_df = actual_data_df[
                    initial_context_offset : len(actual_data_df) - ending_context_offset
                ]
                if not actual_data_df.index.equals(autoencoder_output_df.index):
                    raise ValueError(
                        f"actual_data_df index ({actual_data_df.index}) "
                        f"does not equal autoencoder_output_df index ({autoencoder_output_df.index})"
                    )

                # Calculate reconstruction error and other metrics
                reconstruction_error_df = anomaly_detector.reconstruction_error(
                    actual_data_df=actual_data_df,
                    autoencoder_output_df=autoencoder_output_df,
                    split_column=None,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_error.csv",
                )
                anomaly_detector.reconstruction_error_summary(
                    reconstruction_error_df,
                    split_column=None,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_error_summary.csv",
                )
                plots.boxplot_reconstruction_error(
                    reconstruction_error_df,
                    save_path=save_path,
                    filename=f"{id_i}_reconstruction_error_boxplot.html",
                    show=True,
                )

        return reconstructed_results

    def _reconstruct_single_dataset(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]],
        nan_positions: np.ndarray,
        has_nans: bool,
        iterations: int = 1,
        id_iter: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Reconstruct missing values for a single dataset (either global or for a specific ID).

        :param data: Subset of data to reconstruct (global dataset or per ID)
        :type data: np.ndarray
        :param feature_names: Feature labels
        :type feature_names: Optional[List[str]]
        :param nan_positions: Boolean mask indicating NaN positions
        :type nan_positions: np.ndarray
        :param has_nans: Boolean flag indicating if the dataset contains NaNs
        :type has_nans: bool
        :param iterations: Number of iterations for reconstruction
        :type iterations: int
        :param id_iter: ID of the subset being reconstructed (or None for global)
        :type id_iter: Optional[str]
        :param save_path: Path to save the reconstructed data plots
        :type save_path: Optional[str]
        :return: Reconstructed dataset as a pandas DataFrame
        :rtype: pd.DataFrame
        :raises ValueError: If normalization fails or if there are issues with the reconstruction process
        """
        data_original = np.copy(data)

        reconstructed_iterations = {}

        # Get normalization values for the current ID or global
        normalization_values = (
            self.normalization_values.get(f"{id_iter}")
            if id_iter
            else self.normalization_values.get("global")
        ) or {}

        if not normalization_values:
            # Simulate train/val/test split using only current data
            x_train = x_val = x_test = data

            _, _, _, normalization_values = processing.normalize_data_for_training(
                x_train=x_train,
                x_val=x_val,
                x_test=x_test,
                normalization_method=self._normalization_method,
            )

        # Set normalization parameters
        self.min_x = normalization_values.get("min_x", None)
        self.max_x = normalization_values.get("max_x", None)
        self.mean_ = normalization_values.get("mean_", None)
        self.std_ = normalization_values.get("std_", None)

        # Case 1: No NaNs - Simple prediction
        if not has_nans:
            if self._normalization_method:
                try:
                    data = processing.normalize_data_for_prediction(
                        normalization_method=self._normalization_method,
                        data=data,
                        min_x=self.min_x,
                        max_x=self.max_x,
                        mean_=self.mean_,
                        std_=self.std_,
                    )
                except Exception as e:
                    raise ValueError(f"Error during normalization: {e}")

            data_seq = time_series_to_sequence(
                data=data, context_window=self._context_window
            )
            reconstructed_data = self.model.predict(data_seq)

            if self._normalization_method:
                reconstructed_data = processing.denormalize_data(
                    data=reconstructed_data,
                    normalization_method=self._normalization_method,
                    min_x=(
                        self.min_x[self._feature_to_check]
                        if self.min_x is not None
                        else None
                    ),
                    max_x=(
                        self.max_x[self._feature_to_check]
                        if self.max_x is not None
                        else None
                    ),
                    mean_=(
                        self.mean_[self._feature_to_check]
                        if self.mean_ is not None
                        else None
                    ),
                    std_=(
                        self.std_[self._feature_to_check]
                        if self.std_ is not None
                        else None
                    ),
                )

            padded_reconstructed = processing.apply_padding(
                data=data[:, self._feature_to_check],
                reconstructed=reconstructed_data,
                context_window=self._context_window,
                time_step_to_check=self._time_step_to_check,
            )

            # Generate plot path based on ID
            plot_path = (
                os.path.join(save_path or self.root_dir, "plots", str(id_iter))
                if id_iter
                else os.path.join(save_path or self.root_dir, "plots")
            )

            # Plot actual vs reconstructed data
            # Create DataFrame with actual and reconstructed data
            actual_df = pd.DataFrame(
                data_original[:, self._feature_to_check], columns=feature_names
            )
            actual_df["type"] = "actual"

            reconstructed_df = pd.DataFrame(padded_reconstructed, columns=feature_names)
            reconstructed_df["type"] = "reconstructed"

            plots.plot_actual_and_reconstructed(
                df_actual=actual_df,
                df_reconstructed=reconstructed_df,
                save_path=plot_path,
                feature_labels=feature_names,
            )

            if reconstructed_df.isna().sum().sum() != (self._context_window - 1) * len(
                feature_names
            ):
                raise ValueError(
                    f"Expect context_window-1={(self._context_window - 1)} NaN values per feature."
                    f"There are {reconstructed_df.isna().sum().sum()} NaN values across all {len(feature_names)} features"
                )

            # Remove padding rows
            reconstructed_df = reconstructed_df.drop(columns=["type"], errors="ignore")
            reconstructed_df = reconstructed_df.dropna(axis=0, how="any")
            if len(reconstructed_df) != len(actual_df) - (self._context_window - 1):
                raise ValueError(
                    f"Reconstructed data has {len(reconstructed_df)} rows."
                    f"This should be length of actual data ({len(actual_df)}) minus context offset ({self._context_window-1})"
                )

        # Case 2: With NaNs - Iterative reconstruction
        else:
            reconstruction_records = []
            reconstructed_iterations[0] = np.copy(data[:, self._feature_to_check])

            if self._normalization_method:
                try:
                    data = processing.normalize_data_for_prediction(
                        normalization_method=self._normalization_method,
                        data=data,
                        min_x=self.min_x,
                        max_x=self.max_x,
                        mean_=self.mean_,
                        std_=self.std_,
                    )
                except Exception as e:
                    raise ValueError(
                        f"Error during normalization for ID {id_iter}: {e}"
                    )

            # Handle missing values
            if self.imputer is not None:
                data = self.imputer.apply_imputation(pd.DataFrame(data)).to_numpy()
            else:
                data = np.nan_to_num(data, nan=0)

            # Iterative reconstruction loop
            for iter_num in range(1, iterations):
                # Generate sequence and predict
                data_seq = time_series_to_sequence(
                    data=data, context_window=self._context_window
                )
                reconstructed_data = self.model.predict(data_seq)

                if self._normalization_method:
                    reconstructed_data = processing.denormalize_data(
                        data=reconstructed_data,
                        normalization_method=self._normalization_method,
                        min_x=(
                            self.min_x[self._feature_to_check]
                            if self.min_x is not None
                            else None
                        ),
                        max_x=(
                            self.max_x[self._feature_to_check]
                            if self.max_x is not None
                            else None
                        ),
                        mean_=(
                            self.mean_[self._feature_to_check]
                            if self.mean_ is not None
                            else None
                        ),
                        std_=(
                            self.std_[self._feature_to_check]
                            if self.std_ is not None
                            else None
                        ),
                    )

                # Apply padding and store results
                padded_reconstructed = processing.apply_padding(
                    data=data[:, self._feature_to_check],
                    reconstructed=reconstructed_data,
                    context_window=self._context_window,
                    time_step_to_check=self._time_step_to_check,
                )

                reconstructed_iterations[iter_num] = np.copy(padded_reconstructed)

                # Record reconstruction progress
                normalized_reconstructed = None
                if self._normalization_method:
                    normalized_reconstructed = processing.normalize_data_for_prediction(
                        normalization_method=self._normalization_method,
                        data=padded_reconstructed,
                        feature_to_check_filter=True,
                        feature_to_check=self._feature_to_check,
                        min_x=self.min_x,
                        max_x=self.max_x,
                        mean_=self.mean_,
                        std_=self.std_,
                    )

                for i, j in zip(*np.where(nan_positions)):
                    # Don't update values outside context window
                    if i < self._time_step_to_check[0] or i >= len(data) - (
                        self._context_window - 1 - self._time_step_to_check[0]
                    ):
                        continue

                    col_idx = self._feature_to_check[j]
                    recon_value = padded_reconstructed[i, j]

                    reconstruction_records.append(
                        {
                            "ID": id_iter if id_iter else "global",
                            "Column": j + 1,
                            "Timestep": i,
                            "Iteration": iter_num,
                            "Reconstructed value": recon_value,
                        }
                    )

                    data[i, col_idx] = (
                        normalized_reconstructed[i, j]
                        if self._normalization_method
                        else recon_value
                    )

            # Final reconstruction step
            data_seq = time_series_to_sequence(
                data=data, context_window=self._context_window
            )
            reconstructed_data_final = self.model.predict(data_seq)

            if self._normalization_method:
                reconstructed_data_final = processing.denormalize_data(
                    reconstructed_data_final,
                    normalization_method=self._normalization_method,
                    min_x=(
                        self.min_x[self._feature_to_check]
                        if self.min_x is not None
                        else None
                    ),
                    max_x=(
                        self.max_x[self._feature_to_check]
                        if self.max_x is not None
                        else None
                    ),
                    mean_=(
                        self.mean_[self._feature_to_check]
                        if self.mean_ is not None
                        else None
                    ),
                    std_=(
                        self.std_[self._feature_to_check]
                        if self.std_ is not None
                        else None
                    ),
                )

            padded_reconstructed_final = processing.apply_padding(
                data=data[:, self._feature_to_check],
                reconstructed=reconstructed_data_final,
                context_window=self._context_window,
                time_step_to_check=self._time_step_to_check,
            )
            reconstructed_iterations[iterations] = np.copy(padded_reconstructed_final)

            # Record final reconstruction results
            for i, j in zip(*np.where(nan_positions)):
                # Don't update values outside context window
                if i < self._time_step_to_check[0] or i >= len(data) - (
                    self._context_window - 1 - self._time_step_to_check[0]
                ):
                    continue

                recon_value = padded_reconstructed_final[i, j]

                reconstruction_records.append(
                    {
                        "ID": id_iter if id_iter else "global",
                        "Column": j + 1,
                        "Timestep": i,
                        "Iteration": iterations,
                        "Reconstructed value": recon_value,
                    }
                )

            # Save reconstruction progress
            if save_path:
                progress_df = pd.DataFrame(reconstruction_records)
                filename = (
                    f"{id_iter}_reconstruction_progress.csv.zip"
                    if id_iter
                    else "global_reconstruction_progress.csv.zip"
                )
                processing.save_csv(
                    data=progress_df,
                    save_path=save_path,
                    filename=filename,
                    compression="zip",
                )

            # Plot reconstruction iterations
            plots.plot_reconstruction_iterations(
                original_data=data_original[:, self._feature_to_check].T,
                reconstructed_iterations={
                    k: v.T for k, v in reconstructed_iterations.items()
                },
                save_path=os.path.join(
                    save_path if save_path else self.root_dir, "plots"
                ),
                feature_labels=feature_names,
                id_iter=id_iter,
            )

            # Remove padding rows
            reconstructed_df = pd.DataFrame(
                padded_reconstructed_final, columns=feature_names
            )
            actual_df = pd.DataFrame(
                data_original[:, self._feature_to_check], columns=feature_names
            )

            if reconstructed_df.isna().sum().sum() != (self._context_window - 1) * len(
                feature_names
            ):
                raise ValueError(
                    f"Expect context_window-1={(self._context_window - 1)} NaN values per feature."
                    f"There are {reconstructed_df.isna().sum().sum()} NaN values across all {len(feature_names)} features"
                )

            reconstructed_df = reconstructed_df.dropna(axis=0, how="any")
            if len(reconstructed_df) != len(actual_df) - (self._context_window - 1):
                raise ValueError(
                    f"Reconstructed data has {len(reconstructed_df)} rows."
                    f"This should be length of actual data ({len(actual_df)}) minus context offset ({self._context_window-1})"
                )
            if np.any(np.isnan(reconstructed_df)):
                raise ValueError("There are NaNs after reconstruction.")

        return reconstructed_df

    def prepare_datasets(
        self,
        data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        context_window: int,
        normalize: bool,
        id_iter: Optional[Union[str, int]] = None,
    ) -> bool:
        """
        Prepare the datasets for the model training and testing.

        :param data: Data to train the model. It can be a single numpy array
            with the whole dataset from which a train, validation and test split
            is created, or a tuple with three numpy arrays, one for
            the train, one for the validation and one for the test.
        :type data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        :param context_window: Context window for the model
        :type context_window: int
        :param normalize: Whether to normalize the data or not
        :type normalize: bool
        :param id_iter: ID of the iteration
        :type id_iter: Optional[Union[str, int]]
        :return: True if the datasets are prepared successfully
        :rtype: bool
        :raises ValueError: If data format is invalid or if NaNs are present when use_mask is False
        """
        # we need to set up two functions to prepare the datasets. One when data is a
        # single numpy array and one when data is a tuple with three numpy arrays.
        if isinstance(data, np.ndarray):
            x_train, x_val, x_test = processing.time_series_split(
                data=data,
                train_size=self._train_size,
                val_size=self._val_size,
                test_size=self._test_size,
            )
            data = tuple([x_train, x_val, x_test])
        else:
            if not isinstance(data, tuple) or len(data) != 3:
                raise ValueError(
                    "Data must be a numpy array or a tuple with three numpy arrays"
                )

        return self._prepare_dataset(
            data=data,
            context_window=context_window,
            normalize=normalize,
            id_iter=id_iter,
        )

    def _prepare_dataset(
        self,
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        context_window: int,
        normalize: bool,
        id_iter: Optional[Union[str, int]] = None,
    ) -> bool:
        """
        Prepare the dataset for the model training and testing when the data is a tuple with three numpy arrays.

        :param data: Tuple with three numpy arrays for the train, validation and test datasets
        :type data: Tuple[np.ndarray, np.ndarray, np.ndarray]
        :param context_window: Context window for the model
        :type context_window: int
        :param normalize: Whether to normalize the data or not
        :type normalize: bool
        :param id_iter: ID of the iteration
        :type id_iter: Optional[Union[str, int]]
        :return: True if the dataset is prepared successfully
        :rtype: bool
        :raises ValueError: If mask shapes do not match data shapes or if custom mask format is invalid
        """
        x_train, x_val, x_test = data

        # Make sure train, validation, and test sizes are at least context window length
        for split_name, split_data in zip(
            ["train", "validation", "test"], [x_train, x_val, x_test]
        ):
            if split_data is None:
                raise ValueError(
                    f"{split_name} data is None. Check your data splitting configuration."
                )
            if len(split_data) < context_window:
                raise ValueError(
                    f"Length of {split_name} data ({len(split_data)}) must be at least context window ({context_window})."
                )

        # Save positions where there are NaNs (used in reconstruct())
        x_data = np.concatenate((x_train, x_val, x_test), axis=0)
        id_key = id_iter if id_iter is not None else "global"
        feature_data = x_data[:, self._feature_to_check]
        nan_coordinates = np.argwhere(np.isnan(feature_data))
        self._nan_coordinates[id_key] = nan_coordinates

        # Determine mask
        if self._use_mask:
            if getattr(self, "_custom_mask", None) is None:
                mask_train = np.where(np.isnan(np.copy(x_train)), 0, 1)
                mask_val = np.where(np.isnan(np.copy(x_val)), 0, 1)
                mask_test = np.where(np.isnan(np.copy(x_test)), 0, 1)
            else:
                if isinstance(self._custom_mask, tuple):
                    if id_iter is not None:
                        mask_train = self.id_data_dict_mask[id_iter][0]
                        mask_val = self.id_data_dict_mask[id_iter][1]
                        mask_test = self.id_data_dict_mask[id_iter][2]
                    else:
                        mask_train, mask_val, mask_test = self._custom_mask
                else:
                    mask_train, mask_val, mask_test = processing.time_series_split(
                        data=(
                            self.id_data_dict_mask[id_iter]
                            if id_iter is not None
                            else self._custom_mask
                        ),
                        train_size=self._train_size,
                        val_size=self._val_size,
                        test_size=self._test_size,
                    )

            seq_mask_train, seq_mask_val, seq_mask_test = time_series_to_sequence(
                data=mask_train,
                val_data=mask_val,
                test_data=mask_test,
                context_window=context_window,
            )

        # After determining mask, do normalization
        if normalize:
            x_train, x_val, x_test, norm_values = (
                processing.normalize_data_for_training(
                    x_train=x_train,
                    x_val=x_val,
                    x_test=x_test,
                    normalization_method=self._normalization_method,
                )
            )
            if id_iter is not None:
                self.normalization_values[id_iter] = norm_values
            else:
                self.normalization_values = {"global": norm_values}

        # Impute data
        if self._use_mask and self.imputer is not None:
            x_train = self.imputer.apply_imputation(
                data=pd.DataFrame(x_train)
            ).to_numpy()
            x_val = self.imputer.apply_imputation(data=pd.DataFrame(x_val)).to_numpy()
            x_test = self.imputer.apply_imputation(data=pd.DataFrame(x_test)).to_numpy()
        else:
            x_train = np.nan_to_num(x_train)
            x_val = np.nan_to_num(x_val)
            x_test = np.nan_to_num(x_test)

        # Calculate NaNs after imputation
        train_nan_after = np.isnan(x_train).sum()
        val_nan_after = np.isnan(x_val).sum()
        test_nan_after = np.isnan(x_test).sum()
        nans_after_imputation = train_nan_after + val_nan_after + test_nan_after
        if nans_after_imputation != 0:
            raise ValueError(
                "No NaNs are expected after imputation, yet there "
                f"are {nans_after_imputation} NaNs present."
            )

        seq_x_train, seq_x_val, seq_x_test = time_series_to_sequence(
            data=x_train,
            val_data=x_val,
            test_data=x_test,
            context_window=context_window,
        )

        if id_iter is not None:
            self.data[id_iter] = (x_train, x_val, x_test)
            self.x_train[id_iter] = seq_x_train
            self.x_val[id_iter] = seq_x_val
            self.x_test[id_iter] = seq_x_test
            if self._use_mask:
                self.mask_train[id_iter] = seq_mask_train
                self.mask_val[id_iter] = seq_mask_val
                self.mask_test[id_iter] = seq_mask_test
        else:
            self.data = (seq_x_train, seq_x_val, seq_x_test)
            self.x_train = seq_x_train
            self.x_val = seq_x_val
            self.x_test = seq_x_test
            if self._use_mask:
                self.custom_mask = (seq_mask_train, seq_mask_val, seq_mask_test)
                self.mask_train = seq_mask_train
                self.mask_val = seq_mask_val
                self.mask_test = seq_mask_test

        return True

    def concatenate_by_id(self) -> None:
        """
        Concatenate datasets by ID.
        This method combines the training, validation, and test datasets
        for each ID into a single dataset.
        It also concatenates the masks if they are used.

        :return: None
        :rtype: None
        """
        self._data = {}
        self.x_train = {}
        self.x_val = {}
        self.x_test = {}
        self.mask_train = {}
        self.mask_val = {}
        self.mask_test = {}
        self.length_datasets = {}
        for id_iter, d in self.id_data_dict.items():
            self.prepare_datasets(
                data=d,
                context_window=self._context_window,
                normalize=self._normalize,
                id_iter=id_iter,
            )
            self.length_datasets[id_iter] = {
                "train": len(self.x_train[id_iter]),
                "validation": len(self.x_val[id_iter]),
                "test": len(self.x_test[id_iter]),
            }

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
        if self._use_mask:
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

    @staticmethod
    def masked_weighted_mse(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        time_step_to_check: Union[int, List[int]],
        feature_to_check: Union[int, List[int]],
        feature_weights: Optional[tf.Tensor] = None,
        mask: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Compute Mean Squared Error (MSE) with optional masking and feature weights.

        :param y_true: Ground truth values with shape (batch_size, seq_length, num_features)
        :type y_true: tf.Tensor
        :param y_pred: Predicted values with shape (batch_size, seq_length, num_features)
        :type y_pred: tf.Tensor
        :param time_step_to_check: Time step to check
        :type time_step_to_check: Union[int, List[int]]
        :param feature_to_check: Feature to check
        :type feature_to_check: Union[int, List[int]]
        :param feature_weights: Feature weights
        :type feature_weights: Optional[tf.Tensor]
        :param mask: Optional binary mask with shape (batch_size, seq_length, num_features)
                    1 for observed values, 0 for missing values
        :type mask: Optional[tf.Tensor]
        :return: Masked and weighted MSE loss value
        :rtype: tf.Tensor
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Apply mask if provided
        if mask is not None:
            mask = tf.cast(mask, tf.float32)

            # Select the same time steps and features from the mask as we're using from the data
            # First select the time steps
            mask_selected = tf.gather(
                mask,
                (
                    [time_step_to_check]
                    if isinstance(time_step_to_check, int)
                    else time_step_to_check
                ),
                axis=1,
            )
            # Then select the features
            mask_selected = tf.gather(
                mask_selected,
                (
                    [feature_to_check]
                    if isinstance(feature_to_check, int)
                    else feature_to_check
                ),
                axis=2,
            )
            # Apply the mask to both true and predicted values
            y_true = tf.where(mask_selected > 0, y_true, tf.zeros_like(y_true))
            y_pred = tf.where(mask_selected > 0, y_pred, tf.zeros_like(y_pred))

        squared_error = tf.square(y_true - y_pred)

        # Apply feature-specific weights if provided
        if feature_weights is not None:
            feature_weights = tf.convert_to_tensor(
                value=feature_weights, dtype=tf.float32
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
