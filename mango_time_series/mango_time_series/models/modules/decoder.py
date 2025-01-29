import logging as log
from typing import Union, List

from keras import Input, Model
from keras.src.layers import LSTM, Dense, Flatten


logger = log.getLogger(__name__)


def decoder(type: str, **kwargs):
    """
    Decoder module for different models.
    It can be of different types: currently only Dense and LSTM decoders are implemented.

    :param type: type of decoder, currently only "dense" and "lstm" are supported
    :type type: str
    :param kwargs: keyword arguments for the decoder.
      This can vary depending on the underlying type of decoder.
      Go to specific documentation for more details
    :return: decoder model
    :rtype: keras.Model
    """
    if type == "dense":
        return _decoder_dense(**kwargs)
    elif type == "lstm":
        return _decoder_lstm(**kwargs)
    else:
        raise ValueError(f"Invalid decoder type: {type}")


def _decoder_dense(
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    verbose: bool = False,
) -> Model:
    """
    Dense decoder

    :param features: number of features in the output
    :type features: int
    :param hidden_dim: number of hidden dimensions in the dense layer. It can be a single integer (same for all layers)
      or a list of dimensions for each layer.
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: number of dense layers
    :type num_layers: int
    :param verbose: whether to print the model summary
    :type verbose: bool
    :return: dense decoder model
    :rtype: keras.Model
    """
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * num_layers
    elif isinstance(hidden_dim, list):
        if len(hidden_dim) != num_layers:
            raise ValueError("The length of hidden_dim must match the number of layers")
    else:
        raise ValueError("hidden_dim must be an integer or a list of integers")

    input_layer = Input((hidden_dim[0],))

    for i in range(num_layers):
        layer = Dense(hidden_dim[i])(input_layer if i == 0 else layer)

    output_layer = Dense(features)(layer)
    model = Model(input_layer, output_layer, name="decoder")

    if verbose:
        logger.info(model.summary())

    return model


def _decoder_lstm(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    verbose: bool = False,
) -> Model:
    """
    LSTM decoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the output
    :type features: int
    :param hidden_dim: number of hidden dimensions in the LSTM. It can be a single integer (same for all layers)
      or a list of dimensions for each layer.
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: number of LSTM layers
    :type num_layers: int
    :param verbose: whether to print the model summary
    :type verbose: bool
    :return: LSTM decoder model
    :rtype: keras.Model
    """
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * num_layers
    elif isinstance(hidden_dim, list):
        if len(hidden_dim) != num_layers:
            raise ValueError("The length of hidden_dim must match the number of layers")
    else:
        raise ValueError("hidden_dim must be an integer or a list of integers")

    input_layer = Input((context_window, hidden_dim[0]))

    for i in range(num_layers):
        layer = LSTM(
            hidden_dim[i],
            return_sequences=True,
        )(input_layer if i == 0 else layer)

    flatten = Flatten()(layer)
    output_layer = Dense(features)(flatten)
    model = Model(input_layer, output_layer, name="decoder")

    if verbose:
        logger.info(model.summary())

    return model
