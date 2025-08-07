import logging as log
from typing import Union, List

from keras import Input, Model
from keras.src.layers import LSTM, Dense, Flatten, SimpleRNN, GRU, Bidirectional

logger = log.getLogger(__name__)


def decoder(form: str, **kwargs):
    """
    Decoder module for different models.
    It can be of different types: Dense, RNN, GRU and LSTM decoders.

    :param form: type of decoder, one of "dense", "rnn", "gru" or "lstm"
    :type form: str
    :param kwargs: keyword arguments for the decoder.
      This can vary depending on the underlying type of decoder.
      Go to specific documentation for more details
    :return: decoder model
    :rtype: keras.Model
    """
    if form == "dense":
        return _decoder_dense(**kwargs)
    elif form == "lstm":
        return _decoder_lstm(**kwargs)
    elif form == "rnn":
        return _decoder_rnn(**kwargs)
    elif form == "gru":
        return _decoder_gru(**kwargs)
    else:
        raise ValueError(f"Invalid decoder type: {form}")


def _decoder_dense(
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    activation: str = None,
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
    :param activation: activation function for the dense layer
    :type activation: str
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

    hidden_dim = hidden_dim[::-1]

    input_layer = Input((hidden_dim[0],))

    for i in range(num_layers):
        layer = Dense(hidden_dim[i], activation=activation)(
            input_layer if i == 0 else layer
        )

    output_layer = Dense(features, activation=activation)(layer)
    model = Model(input_layer, output_layer, name="dense_decoder")

    if verbose:
        logger.info(model.summary())

    return model


def _decoder_lstm(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    use_bidirectional: bool = False,
    activation: str = None,
    verbose: bool = False,
) -> Model:
    """
    LSTM decoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the output
    :type features: int
    :param hidden_dim: number of hidden dimensions in the LSTM layer. It can be a single integer (same for all layers)
      or a list of dimensions for each layer.
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: number of LSTM layers
    :type num_layers: int
    :param use_bidirectional: whether to use bidirectional LSTM
    :type use_bidirectional: bool
    :param activation: activation function for the dense layer
    :type activation: str
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

    hidden_dim = hidden_dim[::-1]

    input_layer = Input((context_window, hidden_dim[0]))

    for i in range(num_layers):
        lstm_layer = LSTM(
            hidden_dim[i],
            return_sequences=True,
        )

        if use_bidirectional:
            layer = Bidirectional(lstm_layer)(input_layer if i == 0 else layer)
        else:
            layer = lstm_layer(input_layer if i == 0 else layer)

    flatten = Flatten()(layer)
    output_layer = Dense(features, activation=activation)(flatten)
    model = Model(input_layer, output_layer, name="lstm_decoder")

    if verbose:
        logger.info(model.summary())

    return model


def _decoder_gru(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    use_bidirectional: bool = False,
    activation: str = None,
    verbose: bool = False,
) -> Model:
    """
    GRU decoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the output
    :type features: int
    :param hidden_dim: number of hidden dimensions in the GRU layer. It can be a single integer (same for all layers)
      or a list of dimensions for each layer.
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: number of GRU layers
    :type num_layers: int
    :param use_bidirectional: whether to use bidirectional GRU
    :type use_bidirectional: bool
    :param activation: activation function for the dense layer
    :type activation: str
    :param verbose: whether to print the model summary
    :type verbose: bool
    :return: GRU decoder model
    :rtype: keras.Model
    """
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * num_layers
    elif isinstance(hidden_dim, list):
        if len(hidden_dim) != num_layers:
            raise ValueError("The length of hidden_dim must match the number of layers")
    else:
        raise ValueError("hidden_dim must be an integer or a list of integers")

    hidden_dim = hidden_dim[::-1]

    input_layer = Input((context_window, hidden_dim[0]))

    for i in range(num_layers):
        gru_layer = GRU(
            hidden_dim[i],
            return_sequences=True,
        )
        if use_bidirectional:
            layer = Bidirectional(gru_layer)(input_layer if i == 0 else layer)
        else:
            layer = gru_layer(input_layer if i == 0 else layer)

    flatten = Flatten()(layer)
    output_layer = Dense(features, activation=activation)(flatten)
    model = Model(input_layer, output_layer, name="gru_decoder")

    if verbose:
        logger.info(model.summary())

    return model


def _decoder_rnn(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    use_bidirectional: bool = False,
    activation: str = None,
    verbose: bool = False,
) -> Model:
    """
    RNN decoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the output
    :type features: int
    :param hidden_dim: number of hidden dimensions in the RNN layer. It can be a single integer (same for all layers)
      or a list of dimensions for each layer.
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: number of RNN layers
    :type num_layers: int
    :param use_bidirectional: whether to use bidirectional RNN
    :type use_bidirectional: bool
    :param activation: activation function for the dense layer
    :type activation: str
    :param verbose: whether to print the model summary
    :type verbose: bool
    :return: RNN decoder model
    :rtype: keras.Model
    """
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * num_layers
    elif isinstance(hidden_dim, list):
        if len(hidden_dim) != num_layers:
            raise ValueError("The length of hidden_dim must match the number of layers")
    else:
        raise ValueError("hidden_dim must be an integer or a list of integers")

    hidden_dim = hidden_dim[::-1]

    input_layer = Input((context_window, hidden_dim[0]))

    for i in range(num_layers):
        rnn_layer = SimpleRNN(
            hidden_dim[i],
            return_sequences=True,
        )
        if use_bidirectional:
            layer = Bidirectional(rnn_layer)(input_layer if i == 0 else layer)
        else:
            layer = rnn_layer(input_layer if i == 0 else layer)

    flatten = Flatten()(layer)
    output_layer = Dense(features, activation=activation)(flatten)
    model = Model(input_layer, output_layer, name="rnn_decoder")

    if verbose:
        logger.info(model.summary())

    return model
