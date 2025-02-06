import os
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from typing import List, Optional

pio.renderers.default = "browser"


def plot_loss_history(
    train_loss: List[float],
    val_loss: List[float],
    save_path: str,
):
    """
    Plot training and validation loss history.

    :param train_loss: list of training loss values per epoch
    :type train_loss: List[float]
    :param val_loss: list of validation loss values per epoch
    :type val_loss: List[float]
    :param save_path: path to folder where plot will be saved
    :type save_path: str
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Create figure
    fig = go.Figure()

    # Add traces for training and validation loss
    fig.add_trace(
        go.Scatter(
            y=train_loss, mode="lines", name="Training Loss", line=dict(color="blue")
        )
    )
    fig.add_trace(
        go.Scatter(
            y=val_loss, mode="lines", name="Validation Loss", line=dict(color="red")
        )
    )

    # Update layout
    fig.update_layout(
        title="Training and Validation Loss History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        showlegend=True,
        hovermode="x unified",
    )

    # Save plot
    loss_path = os.path.join(save_path, "loss_history.html")
    fig.write_html(loss_path)


def plot_actual_and_reconstructed(
    actual: np.ndarray,
    reconstructed: np.ndarray,
    save_path: str,
    feature_labels: Optional[List[str]] = None,
    split_size: float = 0.7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Plot actual vs reconstructed values for each feature and save to specified folder.

    :param actual: numpy array of shape (F,N) where F is number of features and N is observations
    :type actual: np.ndarray
    :param reconstructed: numpy array of shape (F,N) where F is number of features and N is observations
    :type reconstructed: np.ndarray
    :param save_path: path to folder where plots will be saved
    :type save_path: str
    :param feature_labels: optional list of labels for each feature
    :type feature_labels: Optional[List[str]]
    :param split_size: proportion of data used for training (validation will be 10% and test the rest)
    :type split_size: float
    :param start_date: optional start date for the plot
    :type start_date: Optional[str]
    :param end_date: optional end date for the plot
    :type end_date: Optional[str]
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    num_features = actual.shape[0]

    # If no feature labels provided, use feature indices
    if feature_labels is None:
        feature_labels = [f"feature_{i}" for i in range(num_features)]
    elif len(feature_labels) != num_features:
        raise ValueError("Number of feature labels must match number of features")

    # Calculate split points
    train_split = round((split_size - 0.1) * reconstructed.shape[1])
    val_split = round(split_size * reconstructed.shape[1])

    # Split reconstructed data into train, validation and test
    reconstructed_train = reconstructed[:, :train_split]
    reconstructed_val = reconstructed[:, train_split:val_split]
    reconstructed_test = reconstructed[:, val_split:]

    # Calculate the length of zeros needed for each section
    train_len = train_split
    val_len = val_split - train_split
    test_len = reconstructed.shape[1] - val_split

    # Create zero arrays for padding
    # For padding after train
    zeros_after_train = np.zeros((reconstructed.shape[0], val_len + test_len))
    # For padding before val
    zeros_before_val = np.zeros((reconstructed.shape[0], train_len))
    # For padding after val
    zeros_after_val = np.zeros((reconstructed.shape[0], test_len))
    # For padding before test
    zeros_before_test = np.zeros((reconstructed.shape[0], train_len + val_len))

    # Pad each split with zeros in the correct positions
    reconstructed_train_padded = np.concatenate(
        (reconstructed_train, zeros_after_train), axis=1
    )
    reconstructed_val_padded = np.concatenate(
        (zeros_before_val, reconstructed_val, zeros_after_val), axis=1
    )
    reconstructed_test_padded = np.concatenate(
        (zeros_before_test, reconstructed_test), axis=1
    )

    for feature, label in enumerate(feature_labels):
        # First plot: Separate actual and reconstructed
        fig_separate = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"Actual - {label}",
                f"Reconstructed - {label}",
            ),
        )

        # Add the actual line plot
        fig_separate.add_trace(
            go.Scatter(y=actual[feature], mode="lines", name="Actual"), row=1, col=1
        )

        # Add the reconstructed line plots
        fig_separate.add_trace(
            go.Scatter(
                y=reconstructed_train_padded[feature], mode="lines", name="Train"
            ),
            row=2,
            col=1,
        )
        fig_separate.add_trace(
            go.Scatter(
                y=reconstructed_val_padded[feature], mode="lines", name="Validation"
            ),
            row=2,
            col=1,
        )
        fig_separate.add_trace(
            go.Scatter(y=reconstructed_test_padded[feature], mode="lines", name="Test"),
            row=2,
            col=1,
        )

        fig_separate.update_layout(title=f"{label} - Separate Views", showlegend=True)

        # Save separate view plot
        separate_path = os.path.join(save_path, f"{label}_separate.html")
        fig_separate.write_html(separate_path)

        # Second plot: Overlapped actual and reconstructed
        fig_overlap = go.Figure()
        fig_overlap.add_trace(
            go.Scatter(y=actual[feature], mode="lines", name="Actual")
        )
        fig_overlap.add_trace(
            go.Scatter(
                y=reconstructed_train_padded[feature],
                mode="lines",
                name="Reconstructed - Train",
            )
        )
        fig_overlap.add_trace(
            go.Scatter(
                y=reconstructed_val_padded[feature],
                mode="lines",
                name="Reconstructed - Validation",
            )
        )
        fig_overlap.add_trace(
            go.Scatter(
                y=reconstructed_test_padded[feature],
                mode="lines",
                name="Reconstructed - Test",
            )
        )

        fig_overlap.update_layout(
            title=f"{label} - Overlapped View",
            xaxis_title="Time Step",
            yaxis_title="Value",
            showlegend=True,
        )

        # Save overlapped view plot
        overlap_path = os.path.join(save_path, f"{label}_overlap.html")
        fig_overlap.write_html(overlap_path)

    fig_all = go.Figure()

    # Add traces for each feature - both actual and reconstructed
    for feature, label in enumerate(feature_labels):
        # Add actual values
        fig_all.add_trace(
            go.Scatter(
                y=actual[feature],
                mode="lines",
                name=f"{label} - Actual",
                line=dict(dash="solid"),
            )
        )
        # Add reconstructed values
        fig_all.add_trace(
            go.Scatter(
                y=reconstructed[feature],
                mode="lines",
                name=f"{label} - Reconstructed",
                line=dict(dash="dash"),
            )
        )

    # Update layout
    fig_all.update_layout(
        title="All Features - Actual vs Reconstructed",
        xaxis_title="Time Step",
        yaxis_title="Value",
        showlegend=True,
        hovermode="x unified",
    )

    # Save combined plot
    combined_path = os.path.join(save_path, "all_features_actual_vs_reconstructed.html")
    fig_all.write_html(combined_path)
