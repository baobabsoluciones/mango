from typing import Union, List

from keras import Input, Model
from keras.src.layers import LSTM, Dense, SimpleRNN, GRU, Bidirectional
from mango_autoencoder.logging import get_configured_logger

logger = get_configured_logger()


def encoder(form: str, **kwargs):
    """
    Create an encoder model for different neural network architectures.

    Factory function that creates encoder models of various types including
    Dense, RNN, GRU and LSTM encoders. The specific parameters depend on
    the encoder type selected.

    :param form: Type of encoder architecture
    :type form: str
    :param kwargs: Keyword arguments specific to the encoder type
    :type kwargs: dict
    :return: Configured encoder model
    :rtype: keras.Model
    :raises ValueError: If an invalid encoder form is specified

    Example:
        >>> # Create LSTM encoder
        >>> lstm_enc = encoder(
        ...     form="lstm",
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Create Dense encoder
        >>> dense_enc = encoder(
        ...     form="dense",
        ...     features=10,
        ...     hidden_dim=[128, 64],
        ...     num_layers=2
        ... )

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
    Create a dense (fully-connected) encoder model.

    Builds a feedforward neural network encoder with configurable layers
    and dimensions. Each layer is a dense layer with optional activation.

    :param features: Number of input features
    :type features: int
    :param hidden_dim: Hidden layer dimensions. Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of dense layers to create
    :type num_layers: int
    :param activation: Activation function for dense layers
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled dense encoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Uniform layer dimensions
        >>> encoder = _encoder_dense(features=10, hidden_dim=64, num_layers=3)
        >>> # Custom layer dimensions
        >>> encoder = _encoder_dense(features=10, hidden_dim=[128, 64, 32], num_layers=3)

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
    Create an LSTM-based encoder model for sequence data.

    Builds a Long Short-Term Memory encoder that processes sequential data
    with configurable layers, dimensions, and bidirectional processing.
    Includes a final dense layer with optional activation.

    :param context_window: Number of timesteps in input sequences
    :type context_window: int
    :param features: Number of features per timestep
    :type features: int
    :param hidden_dim: LSTM layer dimensions. Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of LSTM layers to stack
    :type num_layers: int
    :param use_bidirectional: Whether to use bidirectional LSTM layers
    :type use_bidirectional: bool
    :param activation: Activation function for final dense layer
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled LSTM encoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Standard LSTM encoder
        >>> encoder = _encoder_lstm(
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Bidirectional LSTM with custom dimensions
        >>> encoder = _encoder_lstm(
        ...     context_window=20,
        ...     features=3,
        ...     hidden_dim=[128, 64],
        ...     num_layers=2,
        ...     use_bidirectional=True
        ... )

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
    Create a GRU-based encoder model for sequence data.

    Builds a Gated Recurrent Unit encoder that processes sequential data
    with configurable layers, dimensions, and bidirectional processing.
    Includes a final dense layer with optional activation.

    :param context_window: Number of timesteps in input sequences
    :type context_window: int
    :param features: Number of features per timestep
    :type features: int
    :param hidden_dim: GRU layer dimensions. Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of GRU layers to stack
    :type num_layers: int
    :param use_bidirectional: Whether to use bidirectional GRU layers
    :type use_bidirectional: bool
    :param activation: Activation function for final dense layer
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled GRU encoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Standard GRU encoder
        >>> encoder = _encoder_gru(
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Bidirectional GRU with custom dimensions
        >>> encoder = _encoder_gru(
        ...     context_window=20,
        ...     features=3,
        ...     hidden_dim=[128, 64],
        ...     num_layers=2,
        ...     use_bidirectional=True
        ... )

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
    Create an RNN-based encoder model for sequence data.

    Builds a Simple Recurrent Neural Network encoder that processes sequential data
    with configurable layers, dimensions, and bidirectional processing.
    Includes a final dense layer with optional activation.

    :param context_window: Number of timesteps in input sequences
    :type context_window: int
    :param features: Number of features per timestep
    :type features: int
    :param hidden_dim: RNN layer dimensions. Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of RNN layers to stack
    :type num_layers: int
    :param use_bidirectional: Whether to use bidirectional RNN layers
    :type use_bidirectional: bool
    :param activation: Activation function for final dense layer
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled RNN encoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Standard RNN encoder
        >>> encoder = _encoder_rnn(
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Bidirectional RNN with custom dimensions
        >>> encoder = _encoder_rnn(
        ...     context_window=20,
        ...     features=3,
        ...     hidden_dim=[128, 64],
        ...     num_layers=2,
        ...     use_bidirectional=True
        ... )

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
