import os
import pickle
from typing import Union, List, Tuple, Any, Optional, Dict

import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from keras import Sequential
from keras.src.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from mango.logging import get_configured_logger
from mango.processing.data_imputer import DataImputer
from tensorflow.keras.layers import Dense

from mango_time_series.models.modules import encoder, decoder
from mango_time_series.models.utils.plots import (
    plot_actual_and_reconstructed,
    plot_loss_history,
    plot_reconstruction_iterations,
)
from mango_time_series.models.utils.processing import (
    time_series_split,
    convert_data_to_numpy,
    apply_padding,
    denormalize_data,
    handle_id_columns,
)
from mango_time_series.models.utils.sequences import time_series_to_sequence

logger = get_configured_logger()


class AutoEncoder:
    """
    Autoencoder model

    This Autoencoder model can be highly configurable but is already set up so
    that quick training and profiling can be done.
    """

    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    TEST_SIZE = 0.1

    def __init__(self) -> None:
        """
        Initialize the Autoencoder model with default parameters.

        Initializes internal state variables including paths, model configuration,
        and normalization settings.

        :return: None
        :rtype: None
        """
        self.root_dir = os.path.abspath(os.getcwd())
        self._save_path = None
        self._form = "lstm"
        self.model = None
        self._normalization_method = None
        self.normalization_values = {}
        self.imputer = None
        self._normalize = False
        self._verbose = False
        self._shuffle_buffer_size = None
        self._x_train_no_shuffle = None

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

        self._form = value

    @property
    def time_step_to_check(self) -> Optional[Union[int, List[int]]]:
        """
        Get the time step indices to check during reconstruction.

        :return: Time step indices
        :rtype: Optional[Union[int, List[int]]]
        """
        if not hasattr(self, "_time_step_to_check"):
            return None
        if len(self._time_step_to_check) == 1:
            return self._time_step_to_check[0]
        return self._time_step_to_check

    @time_step_to_check.setter
    def time_step_to_check(self, value: Union[int, List[int]]) -> None:
        """
        Set the time step indices to check during reconstruction.

        :param value: Time step indices to check
        :type value: Union[int, List[int]]
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

        # Convert single integer to list
        if isinstance(value, int):
            value = [value]
        elif not isinstance(value, list):
            raise ValueError(
                "time_step_to_check must be an integer or list of integers"
            )

        # Validate all values are integers
        if not all(isinstance(t, int) for t in value):
            raise ValueError("All elements in time_step_to_check must be integers")

        # If context_window is set, validate indices are in range
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
        return getattr(self, "_context_window", None)

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
    def data(self) -> Optional[np.ndarray]:
        """
        Get the data used for training the model.

        :return: Data used for training the model
        :rtype: Optional[np.ndarray]
        """
        return getattr(self, "_data", None)

    @data.setter
    def data(self, value: Optional[np.ndarray]) -> None:
        """
        Set the data used for training the model.
        """
        # Validate that data is a single array or a tuple of three arrays
        if isinstance(value, tuple):
            if len(value) != 3:
                raise ValueError("Data must be a tuple with three numpy arrays")
        elif not isinstance(value, np.ndarray):
            raise ValueError(
                "Data must be a numpy array or a tuple with three numpy arrays"
            )

        self._data = value

    @property
    def features_name(self) -> Optional[List[str]]:
        """
        Get the features name used for training the model.
        """
        return getattr(self, "_features_name", None)

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
    def feature_to_check(self) -> Optional[Union[int, List[int]]]:
        """
        Get the feature index or indices to check during reconstruction.

        :return: Feature index or indices to check
        :rtype: Optional[Union[int, List[int]]]
        """
        return getattr(self, "_feature_to_check", None)

    @feature_to_check.setter
    def feature_to_check(self, value: Union[int, List[int]]) -> None:
        """
        Set the feature index or indices to check during reconstruction.

        :param value: Feature index or indices to check
        :type value: Union[int, List[int]]
        :return: None
        :rtype: None
        :raises ValueError: If value is not valid or if attempting to change after model is built
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change feature_to_check after model is built")

        if not isinstance(value, (int, list)):
            raise ValueError("feature_to_check must be an integer or list of integers")

        value = [value] if isinstance(value, int) else value

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
        """
        self._normalize = value

    @property
    def normalization_method(self) -> Optional[str]:
        """
        Get the normalization method.

        :return: Normalization method
        :rtype: Optional[str]
        """
        return getattr(self, "_normalization_method", None)

    @normalization_method.setter
    def normalization_method(self, value: str) -> None:
        """
        Set the normalization method.

        :param value: Normalization method
        :type value: str
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change normalization_method after model is built")

        if value not in ["minmax", "zscore"]:
            raise ValueError(
                "Invalid normalization method. Choose 'minmax' or 'zscore'."
            )

        self._normalization_method = value

    @property
    def hidden_dim(self) -> Optional[Union[int, List[int]]]:
        """
        Get the hidden dimensions.

        :return: Hidden dimensions
        :rtype: Optional[Union[int, List[int]]]
        """
        return getattr(self, "_hidden_dim", None)

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
        return getattr(self, "_bidirectional_encoder", False)

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
        return getattr(self, "_bidirectional_decoder", False)

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
        return getattr(self, "_activation_encoder", None)

    @activation_encoder.setter
    def activation_encoder(self, value: Optional[str]) -> None:
        """
        Set the activation function for the encoder.

        :param value: Activation function
        :type value: str
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change activation_encoder after model is built")

        self._activation_encoder = value

    @property
    def activation_decoder(self) -> Optional[str]:
        """
        Get the activation function for the decoder.

        :return: Activation function
        :rtype: Optional[str]
        """
        return getattr(self, "_activation_decoder", None)

    @activation_decoder.setter
    def activation_decoder(self, value: Optional[str]) -> None:
        """
        Set the activation function for the decoder.
        """
        if hasattr(self, "model") and self.model is not None:
            raise ValueError("Cannot change activation_decoder after model is built")

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
        Set the feature weights.
        """
        self._feature_weights = value

    @property
    def train_size(self) -> float:
        """
        Get the training set size proportion.

        :return: Training set size (0.0-1.0)
        :rtype: float
        """
        return getattr(self, "_train_size", self.TRAIN_SIZE)

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
        return getattr(self, "_val_size", self.VAL_SIZE)

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
        return getattr(self, "_test_size", self.TEST_SIZE)

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
        Get the number of layers.
        """
        return getattr(self, "_num_layers", len(self._hidden_dim))

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        """
        Set the number of layers.
        """
        self._num_layers = value

    @property
    def id_data(self) -> Optional[np.ndarray]:
        """
        Get the ID data.
        """
        return getattr(self, "_id_data", None)

    @id_data.setter
    def id_data(self, value: Optional[np.ndarray]) -> None:
        """
        Set the ID data.
        """
        self._id_data = value

    @property
    def id_data_dict(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get the ID data dictionary.
        """
        return getattr(self, "_id_data_dict", None)

    @id_data_dict.setter
    def id_data_dict(self, value: Optional[Dict[str, np.ndarray]]) -> None:
        """
        Set the ID data dictionary.
        """
        self._id_data_dict = value

    @property
    def id_data_mask(self) -> Optional[np.ndarray]:
        """
        Get the ID data mask.
        """
        return getattr(self, "_id_data_mask", None)

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
        return getattr(self, "_id_data_dict_mask", None)

    @id_data_dict_mask.setter
    def id_data_dict_mask(self, value: Optional[Dict[str, np.ndarray]]) -> None:
        """
        Set the ID data mask dictionary.
        """
        self._id_data_dict_mask = value

    @property
    def id_columns_indices(self) -> List[int]:
        """
        Get the indices of the ID columns.
        """
        return getattr(self, "_id_columns_indices", None)

    @id_columns_indices.setter
    def id_columns_indices(self, value: List[int]) -> None:
        """
        Set the indices of the ID columns.
        """
        self._id_columns_indices = value

    @property
    def use_mask(self) -> bool:
        """
        Get the use_mask flag.
        """
        return self._use_mask

    @use_mask.setter
    def use_mask(self, value: bool) -> None:
        """
        Set the use_mask flag.
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
        Get the custom mask.
        """
        return getattr(self, "_custom_mask", None)

    @custom_mask.setter
    def custom_mask(self, value: Optional[np.ndarray]) -> None:
        """
        Set the custom mask.
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
        Get the training mask.
        """
        return getattr(self, "_mask_train", None)

    @mask_train.setter
    def mask_train(self, value: np.ndarray) -> None:
        """
        Set the training mask.
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
        Get the validation mask.
        """
        return getattr(self, "_mask_val", None)

    @mask_val.setter
    def mask_val(self, value: np.ndarray) -> None:
        """
        Set the validation mask.
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
        Get the test mask.
        """
        return getattr(self, "_mask_test", None)

    @mask_test.setter
    def mask_test(self, value: np.ndarray) -> None:
        """
        Set the test mask.
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
        Get the imputer.
        """
        return self._imputer

    @imputer.setter
    def imputer(self, value: Optional[DataImputer]) -> None:
        """
        Set the imputer.
        """
        self._imputer = value

    @property
    def shuffle(self) -> bool:
        """
        Get the shuffle flag.
        """
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value: bool) -> None:
        """
        Set the shuffle flag.
        """
        self._shuffle = value

    @property
    def shuffle_buffer_size(self) -> Optional[int]:
        """
        Get the shuffle buffer size.
        """
        return self._shuffle_buffer_size

    @shuffle_buffer_size.setter
    def shuffle_buffer_size(self, value: Optional[int]) -> None:
        """
        Set the shuffle buffer size.
        """
        if value is not None:
            if not isinstance(value, int) or value <= 0:
                raise ValueError("shuffle_buffer_size must be a positive integer.")

        self._shuffle_buffer_size = value

    @property
    def x_train_no_shuffle(self) -> np.ndarray:
        """
        Get the x_train_no_shuffle.
        """
        if self._x_train_no_shuffle is None:
            return self.x_train
        return self._x_train_no_shuffle

    @x_train_no_shuffle.setter
    def x_train_no_shuffle(self, value: np.ndarray) -> None:
        """
        Set the x_train_no_shuffle.
        """
        self._x_train_no_shuffle = value

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

        :raises NotImplementedError: If form='dense' is specified
        :raises ValueError: If invalid parameters are provided
        :return: None
        :rtype: None
        """
        self.form = form
        self.save_path = save_path
        self.context_window = context_window
        self.time_step_to_check = time_step_to_check
        self.feature_to_check = feature_to_check
        self.normalize = normalize
        self.normalization_method = normalization_method
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
        self.data, extracted_feature_names = convert_data_to_numpy(data)
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
            handle_id_columns(
                self._data, id_columns, self._features_name, self._context_window
            )
        )

        if self._use_mask and custom_mask is not None:
            custom_mask, self.id_data_mask, self.id_data_dict_mask, _ = (
                handle_id_columns(
                    custom_mask,
                    id_columns,
                    self._features_name,
                    self._context_window,
                )
            )
            self.custom_mask = custom_mask

        if self.id_data_dict:
            self.concatenate_by_id()
        else:
            self.prepare_datasets(self._data, self._context_window, self._normalize)

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

        if self._verbose:
            logger.info(f"Model structure:\n{self.model.summary()}")

        self.model_optimizer = self._get_optimizer(optimizer)

        if self._use_mask:
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
            train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
            val_dataset = tf.data.Dataset.from_tensor_slices(self.x_val)
            test_dataset = tf.data.Dataset.from_tensor_slices(self.x_test)

        if shuffle:
            self.train_dataset = train_dataset.shuffle(
                buffer_size=self.shuffle_buffer_size
            )
        self.train_dataset = train_dataset.cache().batch(batch_size)
        self.val_dataset = val_dataset.cache().batch(batch_size)
        self.test_dataset = test_dataset.cache().batch(batch_size)

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
        :param checkpoint: Number of epochs to save a checkpoint
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

        @tf.function
        def train_step(x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
            """
            Training step for the model.

            :param x: Input data
            :type x: tf.Tensor
            :param mask: Optional binary mask for missing values
            :type mask: Optional[tf.Tensor]
            :return: Training loss value
            :rtype: tf.Tensor
            """
            with tf.GradientTape() as autoencoder_tape:
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
                train_loss = self.masked_weighted_mse(
                    y_true=x_real,
                    y_pred=x_pred,
                    feature_weights=self._feature_weights,
                    feature_to_check=self._feature_to_check,
                    time_step_to_check=self._time_step_to_check,
                    mask=mask,
                )

            autoencoder_gradient = autoencoder_tape.gradient(
                train_loss, self.model.trainable_variables
            )

            self.model_optimizer.apply_gradients(
                zip(autoencoder_gradient, self.model.trainable_variables)
            )

            return train_loss

        @tf.function
        def validation_step(
            x: tf.Tensor, mask: Optional[tf.Tensor] = None
        ) -> tf.Tensor:
            """
            Validation step for the model.

            :param x: Input data
            :type x: tf.Tensor
            :param mask: Optional binary mask for missing values
            :type mask: Optional[tf.Tensor]
            :return: Validation loss value
            :rtype: tf.Tensor
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
            val_loss = self.masked_weighted_mse(
                y_true=x_real,
                y_pred=x_pred,
                mask=mask,
                feature_weights=self._feature_weights,
                feature_to_check=self._feature_to_check,
                time_step_to_check=self._time_step_to_check,
            )

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
                if self._use_mask:
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
                if self._use_mask:
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
                    self.save(filename="best_model.pkl")
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping at epoch {epoch} | Best Validation Loss: {best_val_loss:.6f}"
                    )
                    break

            if epoch % self.checkpoint == 0:
                if self._verbose:
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"Training Loss: {avg_train_loss:.6f} | "
                        f"Validation Loss: {avg_val_loss:.6f}"
                    )

                self.save(filename=f"{epoch}.pkl")

                # Store the loss history in the model instance
        self.train_loss_history = train_loss_history
        self.val_loss_history = val_loss_history

        # Plot loss history
        plot_loss_history(
            train_loss=train_loss_history,
            val_loss=val_loss_history,
            save_path=os.path.join(self._save_path, "plots"),
        )

        self.save(filename=f"{self.last_epoch}.pkl")

    def reconstruct(self) -> bool:
        """
        Reconstruct the data using the trained model and plot the actual and reconstructed values.

        :return: True if reconstruction was successful
        :rtype: bool
        """
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

                if self._normalization_method == "minmax":
                    scale_min = norm_values["min_x"][self._feature_to_check]
                    scale_max = norm_values["max_x"][self._feature_to_check]

                    # Denormalize predictions
                    x_hat_train = x_hat_train * (scale_max - scale_min) + scale_min
                    x_hat_val = x_hat_val * (scale_max - scale_min) + scale_min
                    x_hat_test = x_hat_test * (scale_max - scale_min) + scale_min

                    # Denormalize original data
                    x_train_converted = (
                        x_train_converted * (scale_max - scale_min) + scale_min
                    )
                    x_val_converted = (
                        x_val_converted * (scale_max - scale_min) + scale_min
                    )
                    x_test_converted = (
                        x_test_converted * (scale_max - scale_min) + scale_min
                    )

                elif self._normalization_method == "zscore":
                    scale_mean = norm_values["mean_"][self._feature_to_check]
                    scale_std = norm_values["std_"][self._feature_to_check]

                    # Denormalize predictions
                    x_hat_train = x_hat_train * scale_std + scale_mean
                    x_hat_val = x_hat_val * scale_std + scale_mean
                    x_hat_test = x_hat_test * scale_std + scale_mean

                    # Denormalize original data
                    x_train_converted = x_train_converted * scale_std + scale_mean
                    x_val_converted = x_val_converted * scale_std + scale_mean
                    x_test_converted = x_test_converted * scale_std + scale_mean

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
                    val_length = self.length_datasets[id_key]["val"]
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

                    # Apply denormalization based on the normalization method
                    if self._normalization_method == "minmax":
                        scale_min = norm_values["min_x"][self._feature_to_check]
                        scale_max = norm_values["max_x"][self._feature_to_check]

                        # Denormalize predictions
                        id_x_hat_train = (
                            id_x_hat_train * (scale_max - scale_min) + scale_min
                        )
                        id_x_hat_val = (
                            id_x_hat_val * (scale_max - scale_min) + scale_min
                        )
                        id_x_hat_test = (
                            id_x_hat_test * (scale_max - scale_min) + scale_min
                        )

                        # Denormalize original data
                        id_x_train = id_x_train * (scale_max - scale_min) + scale_min
                        id_x_val = id_x_val * (scale_max - scale_min) + scale_min
                        id_x_test = id_x_test * (scale_max - scale_min) + scale_min

                    elif self._normalization_method == "zscore":
                        scale_mean = norm_values["mean_"][self._feature_to_check]
                        scale_std = norm_values["std_"][self._feature_to_check]

                        # Denormalize predictions
                        id_x_hat_train = id_x_hat_train * scale_std + scale_mean
                        id_x_hat_val = id_x_hat_val * scale_std + scale_mean
                        id_x_hat_test = id_x_hat_test * scale_std + scale_mean

                        # Denormalize original data
                        id_x_train = id_x_train * scale_std + scale_mean
                        id_x_val = id_x_val * scale_std + scale_mean
                        id_x_test = id_x_test * scale_std + scale_mean

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

        plot_actual_and_reconstructed(
            actual=x_converted,
            reconstructed=x_hat,
            save_path=os.path.join(self._save_path, "plots"),
            feature_labels=feature_labels,
            train_split=train_split,
            val_split=val_split,
            length_datasets=self.length_datasets if self.id_data_dict != {} else None,
        )

        return True

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
        feature_to_check: Union[int, List[int]] = 0,
        form: str = "lstm",
        hidden_dim: Union[int, List[int]] = None,
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
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        id_columns: Union[str, int, List[str], List[int], None] = None,
        epochs: int = 100,
        checkpoint: int = 10,
        use_early_stopping: bool = True,
        patience: int = 10,
        use_post_decoder_dense: bool = False,
    ) -> "AutoEncoder":
        """
        Build and train the Autoencoder model in a single step.

        This method combines the functionality of `build_model` and `train` methods,
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
        :param save_path: Path to save the reconstructed data plots
        :type save_path: Optional[str]
        :return: Dictionary with reconstructed data per ID (or "global" if no ID)
        :rtype: Dict[str, pd.DataFrame]
        :raises ValueError: If no model is loaded or if id_columns format is invalid
        """
        if self.model is None:
            raise ValueError(
                "No model loaded. Use `load_from_pickle()` before calling `reconstruct_new_data()`."
            )

        data, feature_names = convert_data_to_numpy(data)

        if id_columns is not None:
            if isinstance(id_columns, (str, int)):
                id_columns = [id_columns]

            if not isinstance(id_columns, list):
                raise ValueError("id_columns must be a list of strings or integers")

        features_names_to_check = (
            [feature_names[i] for i in self._feature_to_check]
            if feature_names
            else None
        )

        # Handle ID columns
        if id_columns is not None:
            data, _, id_data_dict, self.id_columns_indices = handle_id_columns(
                data, id_columns, feature_names, self.context_window
            )
        else:
            id_data_dict = {"global": data}

        reconstructed_results = {}

        if id_columns is not None:
            for id_iter, data_id in id_data_dict.items():
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

        # Set normalization parameters
        self.min_x = normalization_values.get("min_x", None)
        self.max_x = normalization_values.get("max_x", None)
        self.mean_ = normalization_values.get("mean_", None)
        self.std_ = normalization_values.get("std_", None)

        # Case 1: No NaNs - Simple prediction
        if not has_nans:
            if self._normalization_method:
                try:
                    data = self._normalize_data(data=data)
                except Exception as e:
                    raise ValueError(f"Error during normalization: {e}")

            data_seq = time_series_to_sequence(data, self._context_window)
            reconstructed_data = self.model.predict(data_seq)

            if self._normalization_method:
                reconstructed_data = denormalize_data(
                    reconstructed_data,
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

            padded_reconstructed = apply_padding(
                data[:, self._feature_to_check],
                reconstructed_data,
                self._context_window,
                self._time_step_to_check,
            )

            reconstructed_df = pd.DataFrame(padded_reconstructed, columns=feature_names)

            # Generate plot path based on ID
            plot_path = (
                os.path.join(save_path or self.root_dir, "plots", str(id_iter))
                if id_iter
                else os.path.join(save_path or self.root_dir, "plots")
            )

            # Plot actual vs reconstructed data
            plot_actual_and_reconstructed(
                actual=data_original[:, self._feature_to_check].T,
                reconstructed=padded_reconstructed.T,
                save_path=plot_path,
                feature_labels=feature_names,
                train_split=None,
                val_split=None,
                length_datasets=None,
            )

            return reconstructed_df

        # Case 2: With NaNs - Iterative reconstruction
        reconstruction_records = []
        reconstructed_iterations[0] = np.copy(data[:, self._feature_to_check])

        if self._normalization_method:
            try:
                data = self._normalize_data(data=data, id_iter=id_iter)
            except Exception as e:
                raise ValueError(f"Error during normalization for ID {id_iter}: {e}")

        # Iterative reconstruction loop
        for iter_num in range(1, iterations):
            # Handle missing values
            if self.imputer is not None:
                data = self.imputer.apply_imputation(pd.DataFrame(data)).to_numpy()
            else:
                data = np.nan_to_num(data, nan=0)

            # Generate sequence and predict
            data_seq = time_series_to_sequence(data, self._context_window)
            reconstructed_data = self.model.predict(data_seq)

            if self._normalization_method:
                reconstructed_data = denormalize_data(
                    reconstructed_data,
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
            padded_reconstructed = apply_padding(
                data[:, self._feature_to_check],
                reconstructed_data,
                self._context_window,
                self._time_step_to_check,
            )
            reconstructed_iterations[iter_num] = np.copy(padded_reconstructed)

            # Record reconstruction progress
            for i, j in zip(*np.where(nan_positions)):
                reconstruction_records.append(
                    {
                        "ID": id_iter if id_iter else "global",
                        "Column": j + 1,
                        "Timestep": i,
                        "Iteration": iter_num,
                        "Reconstructed value": padded_reconstructed[i, j],
                    }
                )

                # Update data with reconstructed values
                if self._normalization_method:
                    data[i, self._feature_to_check[j]] = self._normalize_data(
                        data=padded_reconstructed,
                        id_iter=id_iter,
                        feature_to_check_filter=True,
                    )[i, j]
                else:
                    data[i, self._feature_to_check[j]] = padded_reconstructed[i, j]

        # Final reconstruction step
        if self.imputer is not None:
            data = self.imputer.apply_imputation(pd.DataFrame(data)).to_numpy()
        else:
            data = np.nan_to_num(data, nan=0)

        data_seq = time_series_to_sequence(data, self._context_window)
        reconstructed_data_final = self.model.predict(data_seq)

        if self._normalization_method:
            reconstructed_data_final = denormalize_data(
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
                    self.std_[self._feature_to_check] if self.std_ is not None else None
                ),
            )

        padded_reconstructed_final = apply_padding(
            data[:, self._feature_to_check],
            reconstructed_data_final,
            self._context_window,
            self._time_step_to_check,
        )

        reconstructed_iterations[iterations] = np.copy(padded_reconstructed_final)

        # Record final reconstruction results
        for i, j in zip(*np.where(nan_positions)):
            reconstruction_records.append(
                {
                    "ID": id_iter if id_iter else "global",
                    "Column": j + 1,
                    "Timestep": i,
                    "Iteration": iterations,
                    "Reconstructed value": padded_reconstructed_final[i, j],
                }
            )

        reconstructed_df = pd.DataFrame(reconstructed_data_final, columns=feature_names)

        # Save reconstruction progress
        progress_df = pd.DataFrame(reconstruction_records)
        file_path = os.path.join(
            save_path if save_path else self.root_dir,
            "reconstruction_progress",
            f"{id_iter}_progress.xlsx" if id_iter else "global_progress.xlsx",
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        progress_df.to_excel(file_path, index=False)

        # Plot reconstruction iterations
        plot_reconstruction_iterations(
            original_data=data_original[:, self._feature_to_check].T,
            reconstructed_iterations={
                k: v.T for k, v in reconstructed_iterations.items()
            },
            save_path=os.path.join(save_path if save_path else self.root_dir, "plots"),
            feature_labels=feature_names,
            id_iter=id_iter,
        )

        return reconstructed_df

    def _normalize_data_for_training(
        self,
        x_train: np.ndarray,
        x_val: np.ndarray,
        x_test: np.ndarray,
        id_iter: Optional[Union[str, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalize training, validation and test data using the specified method.
        Computes and stores normalization parameters during training.

        :param x_train: Training data to normalize
        :type x_train: np.ndarray
        :param x_val: Validation data to normalize
        :type x_val: np.ndarray
        :param x_test: Test data to normalize
        :type x_test: np.ndarray
        :param id_iter: ID of the iteration for group-specific normalization
        :type id_iter: Optional[Union[str, int]]
        :return: Tuple containing normalized training, validation and test data
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        :note: The method stores normalization parameters either globally or per ID
               in the instance's normalization_values dictionary.
        """
        normalization_values = {}

        if self._normalization_method == "minmax":
            min_x = np.nanmin(x_train, axis=0)
            max_x = np.nanmax(x_train, axis=0)
            range_x = max_x - min_x
            x_train = (x_train - min_x) / range_x
            x_val = (x_val - min_x) / range_x
            x_test = (x_test - min_x) / range_x
            normalization_values = {"min_x": min_x, "max_x": max_x}
        elif self._normalization_method == "zscore":
            mean_ = np.nanmean(x_train, axis=0)
            std_ = np.nanstd(x_train, axis=0)
            x_train = (x_train - mean_) / std_
            x_val = (x_val - mean_) / std_
            x_test = (x_test - mean_) / std_
            normalization_values = {"mean_": mean_, "std_": std_}

        # Initialize normalization_values attribute if it doesn't exist
        if not hasattr(self, "normalization_values"):
            self.normalization_values = {}

        # Store normalization values by ID or globally
        if id_iter is not None:
            self.normalization_values[f"{id_iter}"] = normalization_values
        else:
            self.normalization_values["global"] = normalization_values

        return x_train, x_val, x_test

    def _normalize_data_for_prediction(
        self, data: np.ndarray, feature_to_check_filter: bool = False
    ) -> np.ndarray | None:
        """
        Normalize new data using stored normalization parameters.
        If parameters are not available, computes them from input data.

        :param data: New data to normalize
        :type data: np.ndarray
        :param feature_to_check_filter: Whether to filter features for checking
        :type feature_to_check_filter: bool
        :return: Normalized data
        :rtype: np.ndarray
        """
        if self._normalization_method == "minmax":
            if self.min_x is None or self.max_x is None:
                min_x = np.nanmin(data, axis=0)
                max_x = np.nanmax(data, axis=0)
                range_x = max_x - min_x
                self.min_x = min_x
                self.max_x = max_x
                return (data - min_x) / range_x
            else:
                if feature_to_check_filter:
                    range_x = (
                        self.max_x[self._feature_to_check]
                        - self.min_x[self._feature_to_check]
                    )
                    return (data - self.min_x[self._feature_to_check]) / range_x
                else:
                    range_x = self.max_x - self.min_x
                    return (data - self.min_x) / range_x

        elif self._normalization_method == "zscore":
            if self.mean_ is None or self.std_ is None:
                mean_ = np.nanmean(data, axis=0)
                std_ = np.nanstd(data, axis=0)
                self.mean_ = mean_
                self.std_ = std_
                return (data - mean_) / std_
            else:
                if feature_to_check_filter:
                    return (data - self.mean_[self._feature_to_check]) / self.std_[
                        self._feature_to_check
                    ]
                else:
                    return (data - self.mean_) / self.std_

    def _normalize_data(
        self,
        x_train: Optional[np.ndarray] = None,
        x_val: Optional[np.ndarray] = None,
        x_test: Optional[np.ndarray] = None,
        data: Optional[np.ndarray] = None,
        id_iter: Optional[Union[str, int]] = None,
        feature_to_check_filter: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None:
        """
        Normalize data using the specified method.
        Can be used for training (x_train, x_val, x_test) or prediction (data).

        :param x_train: Training data for training mode
        :type x_train: Optional[np.ndarray]
        :param x_val: Validation data for training mode
        :type x_val: Optional[np.ndarray]
        :param x_test: Test data for training mode
        :type x_test: Optional[np.ndarray]
        :param data: New data to normalize for prediction mode
        :type data: Optional[np.ndarray]
        :param id_iter: ID of the iteration for group-specific normalization
        :type id_iter: Optional[Union[str, int]]
        :param feature_to_check_filter: Whether to filter features for checking
        :type feature_to_check_filter: bool
        :return: Normalized data in training or prediction mode
        :rtype: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        :raises ValueError: If neither training nor prediction data is provided
        """
        # Training mode
        if x_train is not None and x_val is not None and x_test is not None:
            return self._normalize_data_for_training(
                x_train, x_val, x_test, id_iter=id_iter
            )

        # Prediction mode
        elif data is not None:
            return self._normalize_data_for_prediction(
                data, feature_to_check_filter=feature_to_check_filter
            )

        else:
            raise ValueError(
                "Provide either (x_train, x_val, x_test) for training or `data` for prediction."
            )

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
            x_train, x_val, x_test = time_series_split(
                data, self._train_size, self._val_size, self._test_size
            )
            data = tuple([x_train, x_val, x_test])
        else:
            if not isinstance(data, tuple) or len(data) != 3:
                raise ValueError(
                    "Data must be a numpy array or a tuple with three numpy arrays"
                )

        return self._prepare_dataset(data, context_window, normalize, id_iter=id_iter)

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
                    mask_train, mask_val, mask_test = time_series_split(
                        (
                            self.id_data_dict_mask[id_iter]
                            if id_iter is not None
                            else self._custom_mask
                        ),
                        self._train_size,
                        self._val_size,
                        self._test_size,
                    )

            seq_mask_train = time_series_to_sequence(mask_train, context_window)
            seq_mask_val = time_series_to_sequence(mask_val, context_window)
            seq_mask_test = time_series_to_sequence(mask_test, context_window)

        if normalize:
            x_train, x_val, x_test = self._normalize_data(
                x_train, x_val, x_test, id_iter=id_iter
            )

        if self._use_mask and self.imputer is not None:
            import pandas as pd

            x_train = self.imputer.apply_imputation(pd.DataFrame(x_train)).to_numpy()
            x_val = self.imputer.apply_imputation(pd.DataFrame(x_val)).to_numpy()
            x_test = self.imputer.apply_imputation(pd.DataFrame(x_test)).to_numpy()
        else:
            x_train = np.nan_to_num(x_train)
            x_val = np.nan_to_num(x_val)
            x_test = np.nan_to_num(x_test)

        seq_x_train = time_series_to_sequence(x_train, context_window)
        seq_x_val = time_series_to_sequence(x_val, context_window)
        seq_x_test = time_series_to_sequence(x_test, context_window)

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
                d, self._context_window, self._normalize, id_iter=id_iter
            )
            self.length_datasets[id_iter] = {
                "train": len(self.x_train[id_iter]),
                "val": len(self.x_val[id_iter]),
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
            feature_weights = tf.convert_to_tensor(feature_weights, dtype=tf.float32)
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

    @staticmethod
    def _get_optimizer(
        optimizer_name: str,
    ) -> Union[Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]:
        """
        Returns the optimizer based on the given name.

        :param optimizer_name: Name of the optimizer to use
        :type optimizer_name: str
        :return: The requested optimizer instance
        :rtype: Union[Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam]
        :raises ValueError: If optimizer_name is not a valid optimizer
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
