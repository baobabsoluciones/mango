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
    train_split: Optional[int] = None,
    val_split: Optional[int] = None,
    length_datasets: Optional[dict] = None,
):
    """
    Plot actual vs reconstructed values for each feature and save to specified folder.
    Can separate plots by data ID if id_data is provided.

    :param actual: numpy array of shape (F,N) where F is number of features and N is observations
    :type actual: np.ndarray
    :param reconstructed: numpy array of shape (F,N) where F is number of features and N is observations
    :type reconstructed: np.ndarray
    :param save_path: path to folder where plots will be saved
    :type save_path: str
    :param feature_labels: optional list of labels for each feature
    :type feature_labels: Optional[List[str]]
    :param train_split: index position where training data ends (exclusive)
    :type train_split: Optional[int]
    :param val_split: index position where validation data ends (exclusive)
    :type val_split: Optional[int]
    :param length_datasets: dictionary with the length of the datasets for each id
    :type length_datasets: Optional[dict]
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    num_features = actual.shape[0]

    # If no feature labels provided, use feature indices
    if feature_labels is None:
        feature_labels = [f"feature_{i}" for i in range(num_features)]
    elif len(feature_labels) != num_features:
        raise ValueError("Number of feature labels must match number of features")

    # Set default split points if not provided
    data_length = reconstructed.shape[1]
    if train_split is None:
        train_split = round(0.6 * data_length)
    if val_split is None:
        val_split = round(0.7 * data_length)

    # Ensure splits are valid
    if not (0 < train_split < val_split <= data_length):
        raise ValueError(
            f"Invalid split points: train_split={train_split}, val_split={val_split}, data_length={data_length}"
        )

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

    # Actual data
    actual_train = actual[:, :train_split]
    actual_val = actual[:, train_split:val_split]
    actual_test = actual[:, val_split:]

    # If ID data is provided, process per ID
    if length_datasets is not None:
        # Create a separate directory for ID-based plots
        id_save_path = os.path.join(save_path, "by_id")
        os.makedirs(id_save_path, exist_ok=True)

        # Process each ID separately
        for id_value in sorted(length_datasets.keys()):
            # Get the length of the datasets for this id
            train_len = length_datasets[id_value]["train"]
            val_len = length_datasets[id_value]["val"]
            test_len = length_datasets[id_value]["test"]

            # Extract data for this ID
            id_actual_train = actual_train[:, :train_len]
            id_actual_val = actual_val[:, :val_len]
            id_actual_test = actual_test[:, :test_len]

            # Remove from actual_train, actual_val and actual_test the data for this id
            actual_train = actual_train[:, train_len:]
            actual_val = actual_val[:, val_len:]
            actual_test = actual_test[:, test_len:]

            id_reconstructed_train = reconstructed_train[:, :train_len]
            id_reconstructed_val = reconstructed_val[:, :val_len]
            id_reconstructed_test = reconstructed_test[:, :test_len]

            # Create zero arrays for padding
            zeros_after_train = np.zeros((reconstructed.shape[0], val_len + test_len))
            zeros_before_val = np.zeros((reconstructed.shape[0], train_len))
            zeros_after_val = np.zeros((reconstructed.shape[0], test_len))
            zeros_before_test = np.zeros((reconstructed.shape[0], train_len + val_len))

            # Pad each split with zeros in the correct positions
            id_reconstructed_train_padded = np.concatenate(
                (id_reconstructed_train, zeros_after_train), axis=1
            )
            id_reconstructed_val_padded = np.concatenate(
                (zeros_before_val, id_reconstructed_val, zeros_after_val), axis=1
            )
            id_reconstructed_test_padded = np.concatenate(
                (zeros_before_test, id_reconstructed_test), axis=1
            )

            # Remove from reconstructed_train, reconstructed_val and reconstructed_test the data for this id
            reconstructed_train = reconstructed_train[:, train_len:]
            reconstructed_val = reconstructed_val[:, val_len:]
            reconstructed_test = reconstructed_test[:, test_len:]

            # Concat the dataset and plot
            id_actual = np.concatenate(
                (id_actual_train, id_actual_val, id_actual_test), axis=1
            )

            # Plot for each feature for this ID
            for feature, label in enumerate(feature_labels):
                # Plot for this ID and feature
                fig_id = go.Figure()

                # Add actual values
                fig_id.add_trace(
                    go.Scatter(
                        y=id_actual[feature],
                        mode="lines",
                        name="Actual",
                        line=dict(color="blue"),
                    )
                )

                # Define masks for train, validation, and test
                train_mask = np.arange(id_actual_train.shape[1])
                val_mask = np.arange(
                    id_actual_train.shape[1],
                    id_actual_train.shape[1] + id_actual_val.shape[1],
                )
                test_mask = np.arange(
                    id_actual_train.shape[1] + id_actual_val.shape[1],
                    id_actual_train.shape[1]
                    + id_actual_val.shape[1]
                    + id_actual_test.shape[1],
                )

                # Add reconstructed values with split coloring
                if id_reconstructed_train_padded.shape[1] > 0:
                    fig_id.add_trace(
                        go.Scatter(
                            y=id_reconstructed_train_padded[feature],
                            mode="lines",
                            name="Reconstructed - Train",
                            line=dict(color="green"),
                        )
                    )

                if id_reconstructed_val_padded.shape[1] > id_actual_train.shape[1]:
                    fig_id.add_trace(
                        go.Scatter(
                            y=id_reconstructed_val_padded[feature],
                            mode="lines",
                            name="Reconstructed - Validation",
                            line=dict(color="orange"),
                        )
                    )

                if (
                    id_reconstructed_test_padded.shape[1]
                    > id_actual_train.shape[1] + id_actual_val.shape[1]
                ):
                    fig_id.add_trace(
                        go.Scatter(
                            y=id_reconstructed_test_padded[feature],
                            mode="lines",
                            name="Reconstructed - Test",
                            line=dict(color="red"),
                        )
                    )

                # Update layout
                fig_id.update_layout(
                    title=f"ID {id_value} - {label}",
                    xaxis_title="Time Step",
                    yaxis_title="Value",
                    showlegend=True,
                    hovermode="x unified",
                )

                # Save ID-specific plot
                id_plot_path = os.path.join(
                    save_path, "by_id", f"{label}_id_{id_value}.html"
                )
                fig_id.write_html(id_plot_path)

            # Plot all features for this ID
            fig_all_id = go.Figure()
            for feature, label in enumerate(feature_labels):
                fig_all_id.add_trace(
                    go.Scatter(y=id_actual[feature], mode="lines", name=label)
                )
                fig_all_id.add_trace(
                    go.Scatter(
                        y=id_reconstructed_train_padded[feature],
                        mode="lines",
                        line=dict(dash="dash"),
                        name=f"Reconstructed - Train ({label})",
                    )
                )
                fig_all_id.add_trace(
                    go.Scatter(
                        y=id_reconstructed_val_padded[feature],
                        mode="lines",
                        line=dict(dash="dash"),
                        name=f"Reconstructed - Validation ({label})",
                    )
                )
                fig_all_id.add_trace(
                    go.Scatter(
                        y=id_reconstructed_test_padded[feature],
                        mode="lines",
                        line=dict(dash="dash"),
                        name=f"Reconstructed - Test ({label})",
                    )
                )

                fig_all_id.update_layout(
                    title=f"ID {id_value} - All Features",
                    xaxis_title="Time Step",
                    yaxis_title="Value",
                    showlegend=True,
                    hovermode="x unified",
                )

                id_plot_path = os.path.join(
                    save_path, "by_id", f"all_features_id_{id_value}.html"
                )
                fig_all_id.write_html(id_plot_path)

    else:
        # Continue with original plotting for the full dataset
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
                go.Scatter(
                    y=reconstructed_test_padded[feature], mode="lines", name="Test"
                ),
                row=2,
                col=1,
            )

            fig_separate.update_layout(
                title=f"{label} - Separate Views", showlegend=True
            )

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
        combined_path = os.path.join(
            save_path, "all_features_actual_vs_reconstructed.html"
        )
        fig_all.write_html(combined_path)
