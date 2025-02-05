import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def plot_actual_and_reconstructed(actual, reconstructed, split_size=0.7):
    # Generate a plotly figure with two subplots, one with the actual data and the other with the reconstructed data
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Actual", "Reconstructed"))
    # Add the actual line plot
    fig.add_trace(go.Scatter(y=actual[0], mode="lines"), row=1, col=1)

    # split reconstructed based on split_size
    split_point = round(split_size * reconstructed.shape[1])
    reconstructed_train = reconstructed[:, :split_point]
    reconstructed_test = reconstructed[:, split_point:]

    # Add zeros to each one so it has the original length
    zeros_train = np.zeros((1, actual.shape[1] - reconstructed_train.shape[1]))
    zeros_test = np.zeros((1, actual.shape[1] - reconstructed_test.shape[1]))

    reconstructed_train = np.concatenate((reconstructed_train, zeros_train), axis=1)
    reconstructed_test = np.concatenate((zeros_test, reconstructed_test), axis=1)

    # Add the reconstructed line plot
    fig.add_trace(go.Scatter(y=reconstructed_train[0], mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(y=reconstructed_test[0], mode="lines"), row=2, col=1)

    # fig.add_trace(go.Scatter(y=reconstructed[0], mode="lines"), row=2, col=1)
    fig.show()

    # Now let's plot the actual and reconstructed time series in the same graph to compare
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual[0], mode="lines", name="Actual"))
    fig.add_trace(
        go.Scatter(y=reconstructed_train[0], mode="lines", name="Reconstructed - train")
    )
    fig.add_trace(
        go.Scatter(y=reconstructed_test[0], mode="lines", name="Reconstructed - test")
    )
    fig.show()
