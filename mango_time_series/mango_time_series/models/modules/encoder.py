import logging as log
from typing import Union, List

from keras import Input, Model
from keras.src.layers import LSTM, Dense, SimpleRNN, GRU, Bidirectional

logger = log.getLogger(__name__)


def encoder(form: str, **kwargs):
    """
    Encoder module for different models.
    It can be of different types: Dense, RNN, GRU and LSTM encoders.

    :param form: type of encoder, one of "dense", "rnn", "gru" or "lstm"
    :type form: str
    :param kwargs: keyword arguments for the encoder.
      This can vary depending on the underlying type of encoder.
      Go to specific documentation for more details
    :return: encoder model
    :rtype: keras.Model
    """
    if form == "dense":
        return _encoder_dense(**kwargs)
    elif form == "lstm":
        return _encoder_lstm(**kwargs)
    elif form == "rnn":
        return _encoder_rnn(**kwargs)
    elif form == "gru":
        return _encoder_gru(**kwargs)
    else:
        raise ValueError(f"Invalid encoder type: {form}")


def _encoder_dense(
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    activation: str = None,
    verbose: bool = False,
) -> Model:
    """
    Dense encoder

    :param features: number of features in the input
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

    for i in range(num_layers):
        layer = Dense(hidden_dim[i], activation=activation)(
            input_layer if i == 0 else layer
        )

    model = Model(input_layer, layer, name="dense_encoder")

    if verbose:
        logger.info(model.summary())

    return model


def _encoder_lstm(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    use_bidirectional: bool = False,
    activation: str = None,
    verbose: bool = False,
) -> Model:
    """
    LSTM encoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the input
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
    :return: LSTM encoder model
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

    for i in range(num_layers):
        lstm_layer = LSTM(hidden_dim[i], return_sequences=True)

        if use_bidirectional:
            layer = Bidirectional(lstm_layer)(input_layer if i == 0 else layer)
        else:
            layer = lstm_layer(input_layer if i == 0 else layer)

    dense = Dense(hidden_dim[-1], activation=activation)(layer)
    model = Model(input_layer, dense, name="lstm_encoder")

    if verbose:
        logger.info(model.summary())

    return model


def _encoder_gru(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    use_bidirectional: bool = False,
    activation: str = None,
    verbose: bool = False,
) -> Model:
    """
    GRU encoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the input
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
    :return: GRU encoder model
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

    for i in range(num_layers):
        gru_layer = GRU(
            hidden_dim[i],
            return_sequences=True,
        )
        if use_bidirectional:
            layer = Bidirectional(gru_layer)(input_layer if i == 0 else layer)

        else:
            layer = gru_layer(input_layer if i == 0 else layer)

    dense = Dense(hidden_dim[-1], activation=activation)(layer)
    model = Model(input_layer, dense, name="gru_encoder")

    if verbose:
        logger.info(model.summary())

    return model


def _encoder_rnn(
    context_window: int,
    features: int,
    hidden_dim: Union[int, List[int]],
    num_layers: int,
    use_bidirectional: bool = False,
    activation: str = None,
    verbose: bool = False,
) -> Model:
    """
    RNN encoder

    :param context_window: number of timesteps in the input
    :type context_window: int
    :param features: number of features in the input
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
    :return: RNN encoder model
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

    for i in range(num_layers):
        rnn_layer = SimpleRNN(
            hidden_dim[i],
            return_sequences=True,
        )
        if use_bidirectional:
            layer = Bidirectional(rnn_layer)(input_layer if i == 0 else layer)
        else:
            layer = rnn_layer(input_layer if i == 0 else layer)

    dense = Dense(hidden_dim[-1], activation=activation)(layer)
    model = Model(input_layer, dense, name="rnn_encoder")

    if verbose:
        logger.info(model.summary())

    return model
