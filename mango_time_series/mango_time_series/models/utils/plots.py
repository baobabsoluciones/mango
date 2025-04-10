import os
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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
    df_actual: pd.DataFrame,
    df_reconstructed: pd.DataFrame,
    save_path: str,
    feature_labels: Optional[List[str]] = None,
):
    """
    Plot actual vs reconstructed values for each feature and save to specified folder.
    Can separate plots by data ID if id_data is provided.

    :param df_actual: DataFrame containing actual values
    :type df_actual: pd.DataFrame
    :param df_reconstructed: DataFrame containing reconstructed values
    :type df_reconstructed: pd.DataFrame
    :param save_path: path to folder where plots will be saved
    :type save_path: str
    :param feature_labels: optional list of labels for each feature
    :type feature_labels: Optional[List[str]]
    """
    os.makedirs(save_path, exist_ok=True)

    # Check if we have IDs in the data
    has_ids = "id" in df_actual.columns

    # Check if we have dataset splits
    has_splits = "dataset" in df_actual.columns

    if not has_splits:
        # Simple case: just actual vs reconstructed without splits
        for feature_name in feature_labels:
            feature_df_actual = df_actual.copy()
            feature_df_actual["value"] = df_actual[feature_name]

            fig = go.Figure()

            # Add actual values
            fig.add_trace(
                go.Scatter(
                    y=feature_df_actual["value"],
                    mode="lines",
                    name="Actual",
                    line=dict(color="blue"),
                )
            )

            # Add reconstructed values
            feature_df_reconstructed = df_reconstructed.copy()
            feature_df_reconstructed["value"] = feature_df_reconstructed[feature_name]
            fig.add_trace(
                go.Scatter(
                    y=feature_df_reconstructed["value"],
                    mode="lines",
                    name="Reconstructed",
                    line=dict(dash="dash"),
                )
            )

            # Update layout
            fig.update_layout(
                title=f"{feature_name} - Actual vs Reconstructed",
                xaxis_title="Time Step",
                yaxis_title="Value",
                showlegend=True,
            )

            # Save plot
            plot_path = os.path.join(save_path, f"{feature_name}_new_data.html")
            fig.write_html(plot_path)

        return

    if has_ids:
        # Create plots for each ID and feature
        for id_value in sorted(df_actual["id"].unique()):
            # Create a separate directory for ID-based plots
            id_save_path = os.path.join(save_path, id_value)
            os.makedirs(id_save_path, exist_ok=True)

            # Filter data for this ID
            id_df_actual = df_actual[df_actual["id"] == id_value]
            id_df_reconstructed = df_reconstructed[df_reconstructed["id"] == id_value]

            # Plot for each feature
            for feature_name in feature_labels:
                feature_df_actual = id_df_actual[
                    id_df_actual["feature"] == feature_name
                ]
                feature_df_reconstructed = id_df_reconstructed[
                    id_df_reconstructed["feature"] == feature_name
                ]

                # Create figure
                fig = go.Figure()

                # Add actual values
                fig.add_trace(
                    go.Scatter(
                        x=feature_df_actual["time_step"],
                        y=feature_df_actual["value"],
                        mode="lines",
                        name="Actual",
                        line=dict(color="blue"),
                    )
                )

                # Add reconstructed values for each dataset
                for dataset in ["train", "validation", "test"]:
                    dataset_df_reconstructed = feature_df_reconstructed[
                        (feature_df_reconstructed["dataset"] == dataset)
                    ]

                    if not dataset_df_reconstructed.empty:
                        color = (
                            "green"
                            if dataset == "train"
                            else "orange" if dataset == "validation" else "red"
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=dataset_df_reconstructed["time_step"],
                                y=dataset_df_reconstructed["value"],
                                mode="lines",
                                name=f"Reconstructed - {dataset.capitalize()}",
                                line=dict(color=color),
                            )
                        )

                # Update layout
                fig.update_layout(
                    title=f"ID {id_value} - {feature_name}",
                    xaxis_title="Time Step",
                    yaxis_title="Value",
                    showlegend=True,
                    hovermode="x unified",
                )

                # Save ID-specific plot
                id_plot_path = os.path.join(id_save_path, f"{feature_name}.html")
                fig.write_html(id_plot_path)

            # Plot all features for this ID
            fig_all = go.Figure()

            for feature_name in feature_labels:
                feature_df_actual = id_df_actual[
                    id_df_actual["feature"] == feature_name
                ]

                feature_df_reconstructed = id_df_reconstructed[
                    id_df_reconstructed["feature"] == feature_name
                ]

                fig_all.add_trace(
                    go.Scatter(
                        x=feature_df_actual["time_step"],
                        y=feature_df_actual["value"],
                        mode="lines",
                        name=f"{feature_name} - Actual",
                    )
                )

                # Add reconstructed values for each dataset
                for dataset in ["train", "validation", "test"]:
                    dataset_df_reconstructed = feature_df_reconstructed[
                        feature_df_reconstructed["dataset"] == dataset
                    ]

                    if not dataset_df_reconstructed.empty:
                        color = (
                            "green"
                            if dataset == "train"
                            else "orange" if dataset == "validation" else "red"
                        )
                        fig_all.add_trace(
                            go.Scatter(
                                x=dataset_df_reconstructed["time_step"],
                                y=dataset_df_reconstructed["value"],
                                mode="lines",
                                name=f"{feature_name} - {dataset.capitalize()}",
                                line=dict(dash="dash", color=color),
                            )
                        )

            # Update layout
            fig_all.update_layout(
                title=f"ID {id_value} - All Features",
                xaxis_title="Time Step",
                yaxis_title="Value",
                showlegend=True,
                hovermode="x unified",
            )

            # Save all features plot
            id_plot_path = os.path.join(id_save_path, "all_features.html")
            fig_all.write_html(id_plot_path)

    else:
        # Create plots for each feature
        for feature_name in feature_labels:
            feature_df_actual = df_actual[df_actual["feature"] == feature_name]
            feature_df_reconstructed = df_reconstructed[
                df_reconstructed["feature"] == feature_name
            ]

            # First plot: Separate actual and reconstructed
            fig_separate = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    f"Actual - {feature_name}",
                    f"Reconstructed - {feature_name}",
                ),
            )

            # Add the actual line plot
            fig_separate.add_trace(
                go.Scatter(
                    x=feature_df_actual["time_step"],
                    y=feature_df_actual["value"],
                    mode="lines",
                    name="Actual",
                ),
                row=1,
                col=1,
            )

            # Add the reconstructed line plots for each dataset
            for dataset in ["train", "validation", "test"]:
                dataset_df_reconstructed = feature_df_reconstructed[
                    feature_df_reconstructed["dataset"] == dataset
                ]

                if not dataset_df_reconstructed.empty:
                    color = (
                        "green"
                        if dataset == "train"
                        else "orange" if dataset == "validation" else "red"
                    )
                    fig_separate.add_trace(
                        go.Scatter(
                            x=dataset_df_reconstructed["time_step"],
                            y=dataset_df_reconstructed["value"],
                            mode="lines",
                            name=dataset.capitalize(),
                            line=dict(color=color),
                        ),
                        row=2,
                        col=1,
                    )

            fig_separate.update_layout(
                title=f"{feature_name} - Separate Views", showlegend=True
            )

            # Save separate view plot
            separate_path = os.path.join(save_path, f"{feature_name}_separate.html")
            fig_separate.write_html(separate_path)

            # Second plot: Overlapped actual and reconstructed
            fig_overlap = go.Figure()

            # Add actual values
            fig_overlap.add_trace(
                go.Scatter(
                    x=feature_df_actual["time_step"],
                    y=feature_df_actual["value"],
                    mode="lines",
                    name="Actual",
                )
            )

            # Add reconstructed values for each dataset
            for dataset in ["train", "validation", "test"]:
                dataset_df_reconstructed = feature_df_reconstructed[
                    feature_df_reconstructed["dataset"] == dataset
                ]

                if not dataset_df_reconstructed.empty:
                    color = (
                        "green"
                        if dataset == "train"
                        else "orange" if dataset == "validation" else "red"
                    )
                    fig_overlap.add_trace(
                        go.Scatter(
                            x=dataset_df_reconstructed["time_step"],
                            y=dataset_df_reconstructed["value"],
                            mode="lines",
                            name=f"Reconstructed - {dataset.capitalize()}",
                            line=dict(color=color),
                        )
                    )

            fig_overlap.update_layout(
                title=f"{feature_name} - Overlapped View",
                xaxis_title="Time Step",
                yaxis_title="Value",
                showlegend=True,
            )

            # Save overlapped view plot
            overlap_path = os.path.join(save_path, f"{feature_name}_overlap.html")
            fig_overlap.write_html(overlap_path)

        # Create a combined plot for all features
        fig_all = go.Figure()

        # Add traces for each feature - both actual and reconstructed
        for feature_name in feature_labels:
            feature_df_actual = df_actual[df_actual["feature"] == feature_name]
            feature_df_reconstructed = df_reconstructed[
                df_reconstructed["feature"] == feature_name
            ]

            # Add actual values
            fig_all.add_trace(
                go.Scatter(
                    x=feature_df_actual["time_step"],
                    y=feature_df_actual["value"],
                    mode="lines",
                    name=f"{feature_name} - Actual",
                    line=dict(dash="solid"),
                )
            )

            # Add reconstructed values for each dataset
            for dataset in ["train", "validation", "test"]:
                dataset_df_reconstructed = feature_df_reconstructed[
                    feature_df_reconstructed["dataset"] == dataset
                ]

                if not dataset_df_reconstructed.empty:
                    color = (
                        "green"
                        if dataset == "train"
                        else "orange" if dataset == "validation" else "red"
                    )
                    fig_all.add_trace(
                        go.Scatter(
                            x=dataset_df_reconstructed["time_step"],
                            y=dataset_df_reconstructed["value"],
                            mode="lines",
                            name=f"{feature_name} - {dataset.capitalize()}",
                            line=dict(dash="dash", color=color),
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


def plot_reconstruction_iterations(
    original_data: np.ndarray,
    reconstructed_iterations: dict,
    save_path: str,
    feature_labels: Optional[List[str]] = None,
    id_iter: Optional[str] = None,
):
    """
    Plots the original data with missing values, the first full reconstruction,
    and the iterative reconstruction of NaN values.

    :param original_data: 2D numpy array (features x timesteps) with the original data.
    :param reconstructed_iterations: Dictionary {iteration: 2D numpy array}
                                     containing reconstructions per iteration.
    :param save_path: Path to save the plots.
    :param feature_labels: Optional list of labels for each feature.
    :param id_iter: Optional ID to distinguish plots if working with multiple IDs.
    """
    os.makedirs(save_path, exist_ok=True)

    num_features, num_timesteps = original_data.shape
    max_iterations = max(reconstructed_iterations.keys())

    if feature_labels is None:
        feature_labels = [f"Feature {i}" for i in range(num_features)]

    for feature_idx in range(num_features):
        fig = go.Figure()

        # Original data (with NaNs hidden from visualization)
        original_values = original_data[feature_idx]
        nan_mask = np.isnan(original_values)
        fig.add_trace(
            go.Scatter(
                y=np.where(nan_mask, None, original_values),
                mode="lines",
                name=f"Original data {id_iter}" if id_iter else "Original data",
                line=dict(color="black", width=2, dash="solid"),
            )
        )

        # Iterative NaN reconstructions
        colors = [
            "red",
            "orange",
            "green",
            "purple",
            "pink",
            "brown",
            "cyan",
            "magenta",
        ]
        for iter_num in range(1, max_iterations):
            reconstructed_values = np.copy(original_values)
            nan_x, nan_y = [], []

            for t in range(num_timesteps):
                if np.isnan(original_data[feature_idx, t]):
                    reconstructed_values[t] = reconstructed_iterations[iter_num][
                        feature_idx, t
                    ]
                    nan_x.append(t)
                    nan_y.append(reconstructed_iterations[iter_num][feature_idx, t])

            fig.add_trace(
                go.Scatter(
                    y=reconstructed_values,
                    mode="lines",
                    name=(
                        f"Iteration {iter_num} {id_iter}"
                        if id_iter
                        else f"Iteration {iter_num}"
                    ),
                    line=dict(
                        color=colors[(iter_num - 2) % len(colors)],
                        width=1.5,
                        dash="dot",
                    ),
                )
            )

            # Add scatter points for reconstructed NaN values
            fig.add_trace(
                go.Scatter(
                    x=nan_x,
                    y=nan_y,
                    mode="markers",
                    name=(
                        f"Reconstructed NaNs (Iter {iter_num}) {id_iter}"
                        if id_iter
                        else f"Reconstructed NaNs (Iter {iter_num})"
                    ),
                    marker=dict(
                        color=colors[(iter_num - 2) % len(colors)],
                        size=6,
                        symbol="circle",
                    ),
                )
            )

        # Final iteration (Full dataset reconstruction)
        fig.add_trace(
            go.Scatter(
                y=reconstructed_iterations[max_iterations][feature_idx],
                mode="lines",
                name=(
                    f"Final reconstruction {id_iter}"
                    if id_iter
                    else "Final reconstruction"
                ),
                line=dict(color="darkblue", width=2.5, dash="solid"),
            )
        )

        # Layout and save
        fig.update_layout(
            title=(
                f"Feature: {feature_labels[feature_idx]} - Reconstruction Progress {id_iter}"
                if id_iter
                else f"Feature: {feature_labels[feature_idx]} - Reconstruction Progress"
            ),
            xaxis_title="Time Step",
            yaxis_title="Value",
            showlegend=True,
        )

        plot_filename = f"{feature_labels[feature_idx]}_iterations.html"
        plot_dir = os.path.join(save_path, str(id_iter)) if id_iter else save_path
        os.makedirs(plot_dir, exist_ok=True)

        plot_path = os.path.join(plot_dir, plot_filename)
        fig.write_html(plot_path)
