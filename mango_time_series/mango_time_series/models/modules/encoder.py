import logging as log

from keras import Input, Model
from keras.src.layers import LSTM, Dense
from typing import Union, List


logger = log.getLogger(__name__)


def encoder(type: str, **kwargs):
    """
    Encoder module for different models.
    It can be of different types: currently only Dense and LSTM encoders are implemented.

    :param type: type of encoder, currently only "dense" and "lstm" are supported
    :type type: str
    :param kwargs: keyword arguments for the encoder.
      This can vary depending on the underlying type of encoder.
      Go to specific documentation for more details
    :return: encoder model
    :rtype: keras.Model
    """
    if type == "dense":
        return _encoder_dense(**kwargs)
    elif type == "lstm":
        return _encoder_lstm(**kwargs)
    else:
        raise ValueError(f"Invalid encoder type: {type}")


def _encoder_dense(
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    verbose: bool = False,
) -> Model:
    """
    Dense encoder for LSTM autoencoder

    :param features: number of features in the input
    :type features: int
    :param hidden_dim: number of hidden dimensions in the dense layer. It can be a single integer (same for all layers)
      or a list of dimensions for each layer.
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: number of dense layers
    :type num_layers: int
    :param verbose: whether to print the model summary
    :type verbose: bool
    :return: dense encoder model
    :rtype: keras.Model
    """

    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * num_layers
    elif isinstance(hidden_dim, list):
        if len(hidden_dim) != num_layers:
            raise ValueError("The length of hidden_dim must match the number of layers")
    else:
        raise ValueError("hidden_dim must be an integer or a list of integers")

    input_layer = Input((features,))

    for _ in range(num_layers):
        layer = Dense(hidden_dim[_])(input_layer if _ == 0 else layer)

    model = Model(input_layer, layer, name="encoder")

    if verbose:
        logger.info(model.summary())

    return model


def _encoder_lstm(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    verbose: bool = False,
) -> Model:
    """
    LSTM encoder for LSTM autoencoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the input
    :type features: int
    :param hidden_dim: number of hidden dimensions in the LSTM. It can be a single integer (same for all layers)
      or a list of dimensions for each layer.
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: number of LSTM layers
    :type num_layers: int
    :param verbose: whether to print the model summary
    :type verbose: bool
    :return: LSTM encoder model with the specified parameters
    :rtype: keras.Model
    """

    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * num_layers
    elif isinstance(hidden_dim, list):
        if len(hidden_dim) != num_layers:
            raise ValueError("The length of hidden_dim must match the number of layers")
    else:
        raise ValueError("hidden_dim must be an integer or a list of integers")

    input_layer = Input((context_window, features))

    for _ in range(num_layers):
        layer = LSTM(hidden_dim, return_sequences=True)(
            input_layer if _ == 0 else layer
        )
    dense = Dense(hidden_dim)(layer)
    model = Model(input_layer, dense, name="encoder")
    if verbose:
        logger.info(model.summary())
    return model
