from keras.src.losses import MeanSquaredError


def mean_squared_error(y_true, y_pred):
    """
    Mean squared error loss function.

    Readily prepared to be a tensorflow function
    """
    loss = MeanSquaredError()
    return loss(y_true, y_pred)
