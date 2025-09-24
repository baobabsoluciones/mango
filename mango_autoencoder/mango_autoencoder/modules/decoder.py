from typing import Union, List

from keras import Input, Model
from keras.src.layers import LSTM, Dense, Flatten, SimpleRNN, GRU, Bidirectional
from mango_autoencoder.logging import get_configured_logger

logger = get_configured_logger()


def decoder(form: str, **kwargs):
    """
    Create a decoder model for different neural network architectures.

    Factory function that creates decoder models of various types including
    Dense, RNN, GRU and LSTM decoders. The decoder reconstructs the original
    input from the encoded representation.

    :param form: Type of decoder architecture
    :type form: str
    :param kwargs: Keyword arguments specific to the decoder type
    :type kwargs: dict
    :return: Configured decoder model
    :rtype: keras.Model
    :raises ValueError: If an invalid decoder form is specified

    Example:
        >>> # Create LSTM decoder
        >>> lstm_dec = decoder(
        ...     form="lstm",
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Create Dense decoder
        >>> dense_dec = decoder(
        ...     form="dense",
        ...     features=10,
        ...     hidden_dim=[64, 128],
        ...     num_layers=2
        ... )

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
    Create a dense (fully-connected) decoder model.

    Builds a feedforward neural network decoder that reconstructs the original
    input from encoded representations. Layer dimensions are reversed from
    encoder configuration to create a symmetric architecture.

    :param features: Number of output features to reconstruct
    :type features: int
    :param hidden_dim: Hidden layer dimensions (reversed internally). Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of dense layers to create
    :type num_layers: int
    :param activation: Activation function for dense layers
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled dense decoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Uniform layer dimensions (reversed internally)
        >>> decoder = _decoder_dense(features=10, hidden_dim=64, num_layers=3)
        >>> # Custom layer dimensions (will be reversed: [32, 64, 128])
        >>> decoder = _decoder_dense(features=10, hidden_dim=[128, 64, 32], num_layers=3)

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
    Create an LSTM-based decoder model for sequence data reconstruction.

    Builds a Long Short-Term Memory decoder that reconstructs original sequences
    from encoded representations. Layer dimensions are reversed from encoder
    configuration and includes flattening and dense output layers.

    :param context_window: Number of timesteps in input sequences
    :type context_window: int
    :param features: Number of output features to reconstruct
    :type features: int
    :param hidden_dim: LSTM layer dimensions (reversed internally). Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of LSTM layers to stack
    :type num_layers: int
    :param use_bidirectional: Whether to use bidirectional LSTM layers
    :type use_bidirectional: bool
    :param activation: Activation function for final dense layer
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled LSTM decoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Standard LSTM decoder
        >>> decoder = _decoder_lstm(
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Bidirectional LSTM with custom dimensions (will be reversed)
        >>> decoder = _decoder_lstm(
        ...     context_window=20,
        ...     features=3,
        ...     hidden_dim=[64, 128],
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
    Create a GRU-based decoder model for sequence data reconstruction.

    Builds a Gated Recurrent Unit decoder that reconstructs original sequences
    from encoded representations. Layer dimensions are reversed from encoder
    configuration and includes flattening and dense output layers.

    :param context_window: Number of timesteps in input sequences
    :type context_window: int
    :param features: Number of output features to reconstruct
    :type features: int
    :param hidden_dim: GRU layer dimensions (reversed internally). Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of GRU layers to stack
    :type num_layers: int
    :param use_bidirectional: Whether to use bidirectional GRU layers
    :type use_bidirectional: bool
    :param activation: Activation function for final dense layer
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled GRU decoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Standard GRU decoder
        >>> decoder = _decoder_gru(
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Bidirectional GRU with custom dimensions (will be reversed)
        >>> decoder = _decoder_gru(
        ...     context_window=20,
        ...     features=3,
        ...     hidden_dim=[64, 128],
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
    Create an RNN-based decoder model for sequence data reconstruction.

    Builds a Simple Recurrent Neural Network decoder that reconstructs original
    sequences from encoded representations. Layer dimensions are reversed from
    encoder configuration and includes flattening and dense output layers.

    :param context_window: Number of timesteps in input sequences
    :type context_window: int
    :param features: Number of output features to reconstruct
    :type features: int
    :param hidden_dim: RNN layer dimensions (reversed internally). Single int for uniform layers or list for custom dimensions
    :type hidden_dim: Union[int, List[int]]
    :param num_layers: Number of RNN layers to stack
    :type num_layers: int
    :param use_bidirectional: Whether to use bidirectional RNN layers
    :type use_bidirectional: bool
    :param activation: Activation function for final dense layer
    :type activation: str, optional
    :param verbose: Whether to print model summary
    :type verbose: bool
    :return: Compiled RNN decoder model
    :rtype: keras.Model
    :raises ValueError: If hidden_dim list length doesn't match num_layers

    Example:
        >>> # Standard RNN decoder
        >>> decoder = _decoder_rnn(
        ...     context_window=10,
        ...     features=5,
        ...     hidden_dim=64,
        ...     num_layers=2
        ... )
        >>> # Bidirectional RNN with custom dimensions (will be reversed)
        >>> decoder = _decoder_rnn(
        ...     context_window=20,
        ...     features=3,
        ...     hidden_dim=[64, 128],
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
