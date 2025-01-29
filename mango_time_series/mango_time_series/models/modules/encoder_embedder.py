import logging as log
from typing import Union, List

from keras import Input, Model
from keras.src.layers import LSTM


logger = log.getLogger(__name__)


def encoder_embedder(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int = 1,
    verbose: bool = False,
) -> Model:
    """
    Encoder embedder module that creates an LSTM-based embedding model.

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
    :return: LSTM encoder embedder model with the specified parameters
    :rtype: keras.Model
    """
    if num_layers == 0:
        raise ValueError("Number of layers must be greater than 0")

    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * num_layers
    elif isinstance(hidden_dim, list):
        if len(hidden_dim) != num_layers:
            raise ValueError("The length of hidden_dim must match the number of layers")
    else:
        raise ValueError("hidden_dim must be an integer or a list of integers")

    input_layer = Input((context_window, features))

    for i in range(num_layers):
        layer = LSTM(
            hidden_dim[i],
            return_sequences=True,
        )(input_layer if i == 0 else layer)

    model = Model(input_layer, layer, name="encoder_embedder")

    if verbose:
        logger.info(model.summary())

    return model
