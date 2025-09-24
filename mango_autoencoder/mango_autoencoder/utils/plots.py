import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from mango_autoencoder.logging import get_configured_logger
from plotly.subplots import make_subplots

pio.renderers.default = "browser"

logger = get_configured_logger()


def plot_loss_history(
    train_loss: List[float],
    val_loss: List[float],
    save_path: str,
):
    """
    Plot training and validation loss history over epochs.

    Creates an interactive Plotly line chart showing the progression of
    training and validation losses during model training. The plot is
    saved as an HTML file in the specified directory.

    :param train_loss: Training loss values for each epoch
    :type train_loss: List[float]
    :param val_loss: Validation loss values for each epoch
    :type val_loss: List[float]
    :param save_path: Directory path where the plot HTML file will be saved
    :type save_path: str
    :return: None
    :rtype: None

    Example:
        >>> train_losses = [0.5, 0.3, 0.2, 0.15, 0.1]
        >>> val_losses = [0.6, 0.4, 0.25, 0.18, 0.12]
        >>> plot_loss_history(train_losses, val_losses, "./plots")
        # Saves loss_history.html in ./plots/ directory

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

    Creates comprehensive visualizations comparing original data with autoencoder
    reconstructions. Supports different data structures including ID-based data
    and dataset splits (train/validation/test). Generates multiple plot types
    including separate views, overlapped views, and combined feature plots.

    :param df_actual: DataFrame containing actual/original values
    :type df_actual: pd.DataFrame
    :param df_reconstructed: DataFrame containing reconstructed values from autoencoder
    :type df_reconstructed: pd.DataFrame
    :param save_path: Directory path where plots will be saved as HTML files
    :type save_path: str
    :param feature_labels: Optional list of labels for each feature column
    :type feature_labels: Optional[List[str]]
    :return: None
    :rtype: None

    Example:
        >>> import pandas as pd
        >>> actual_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> reconstructed_df = pd.DataFrame({"feature1": [1.1, 1.9, 3.1], "feature2": [3.9, 5.2, 5.8]})
        >>> plot_actual_and_reconstructed(actual_df, reconstructed_df, "./plots", ["sensor1", "sensor2"])

    """
    os.makedirs(save_path, exist_ok=True)

    # Check if we have IDs in the data
    has_ids = "id" in df_actual.columns

    # Check if we have dataset splits
    has_splits = "data_split" in df_actual.columns

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
                        (feature_df_reconstructed["data_split"] == dataset)
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
                        feature_df_reconstructed["data_split"] == dataset
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
                    feature_df_reconstructed["data_split"] == dataset
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
                    feature_df_reconstructed["data_split"] == dataset
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
                    feature_df_reconstructed["data_split"] == dataset
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
    Plot the original data with missing values and iterative reconstruction progress.

    Creates detailed visualizations showing the progression of NaN value reconstruction
    across multiple iterations. Displays original data (with NaNs), intermediate
    reconstruction iterations, and final reconstruction results for each feature.

    :param original_data: 2D numpy array (features x timesteps) with the original data including NaNs
    :type original_data: np.ndarray
    :param reconstructed_iterations: Dictionary mapping iteration numbers to 2D numpy arrays containing reconstructions
    :type reconstructed_iterations: dict
    :param save_path: Directory path where plots will be saved as HTML files
    :type save_path: str
    :param feature_labels: Optional list of labels for each feature
    :type feature_labels: Optional[List[str]]
    :param id_iter: Optional identifier to distinguish plots when working with multiple datasets
    :type id_iter: Optional[str]
    :return: None
    :rtype: None

    Example:
        >>> import numpy as np
        >>> original = np.array([[1, np.nan, 3], [4, 5, np.nan]])
        >>> iterations = {1: np.array([[1, 2, 3], [4, 5, 6]]), 2: np.array([[1, 2.1, 3], [4, 5, 5.9]])}
        >>> plot_reconstruction_iterations(original, iterations, "./plots", ["feature1", "feature2"])

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


def create_error_analysis_dashboard(
    error_df: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "error_analysis_dashboard.html",
    show: bool = True,
    height: int = 1000,
    width: int = 1200,
    template: str = "plotly_white",
) -> go.Figure:
    """
    Create an interactive dashboard for comprehensive error analysis with multiple plots.

    Generates a multi-panel dashboard containing bar plots of mean errors by feature,
    box plots showing error distributions, and correlation heatmaps between features.
    Provides comprehensive visualization for understanding reconstruction error patterns.

    :param error_df: DataFrame containing reconstruction error data (samples x features)
    :type error_df: pd.DataFrame
    :param save_path: Optional directory path to save the dashboard HTML file
    :type save_path: Optional[str]
    :param filename: Name of the HTML file to save
    :type filename: str
    :param show: Whether to display the dashboard in browser
    :type show: bool
    :param height: Height of the figure in pixels
    :type height: int
    :param width: Width of the figure in pixels
    :type width: int
    :param template: Plotly template for styling (e.g., 'plotly_white', 'ggplot2')
    :type template: str
    :return: Plotly figure object containing the dashboard
    :rtype: go.Figure

    Example:
        >>> import pandas as pd
        >>> error_data = pd.DataFrame({"feature1": [0.1, 0.2, 0.3], "feature2": [0.2, 0.1, 0.4]})
        >>> dashboard = create_error_analysis_dashboard(error_data, save_path="./plots")

    """
    try:
        # Create a subplot figure with 3 plots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Mean Error by Feature",
                "Error Distribution by Feature",
                "Error Correlation Between Features",
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "heatmap", "colspan": 2}, None],
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )

        # Mean error by column - barplot
        mean_errors = error_df.mean().reset_index()
        mean_errors.columns = ["Feature", "Mean Error"]
        fig.add_trace(
            go.Bar(
                x=mean_errors["Feature"], y=mean_errors["Mean Error"], name="Mean Error"
            ),
            row=1,
            col=1,
        )

        # Error distribution - boxplot
        for col in error_df.columns:
            fig.add_trace(
                go.Box(y=error_df[col], name=col, boxpoints="outliers"), row=1, col=2
            )

        # Heatmap of correlation between errors
        corr = error_df.corr()
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1,
            ),
            row=2,
            col=1,
        )

        # Update layout
        fig.update_layout(
            height=height,
            width=width,
            showlegend=False,
            title_text="Error Analysis Dashboard",
            template=template,
        )

        # Update axes labels
        fig.update_xaxes(title_text="Feature", row=1, col=1)
        fig.update_yaxes(title_text="Mean Error", row=1, col=1)
        fig.update_xaxes(title_text="Feature", row=1, col=2)
        fig.update_yaxes(title_text="Error Value", row=1, col=2)
        fig.update_xaxes(title_text="Feature", row=2, col=1)
        fig.update_yaxes(title_text="Feature", row=2, col=1)

        if save_path:
            output_dir = Path(save_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_dir / filename)
            logger.info(f"Error analysis dashboard saved to {output_dir / filename}")

        if show:
            fig.show()

        return fig

    except Exception as e:
        logger.error(f"Error creating error analysis dashboard: {str(e)}")
        raise


def boxplot_reconstruction_error(
    reconstruction_error_df: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "reconstruction_error_boxplot.html",
    show: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
    template: str = "plotly_white",
    xaxis_tickangle: int = -45,
    color_palette: Optional[List[str]] = None,
) -> go.Figure:
    """
    Generate and optionally save a boxplot for reconstruction error analysis using Plotly.

    Creates interactive boxplots showing the distribution of reconstruction errors
    across features and optionally across dataset splits (train/validation/test).
    Provides statistical insights into error patterns and outliers.

    :param reconstruction_error_df: DataFrame with reconstruction error values (samples x features)
    :type reconstruction_error_df: pd.DataFrame
    :param save_path: Optional directory path to save the plot HTML file
    :type save_path: Optional[str]
    :param filename: Name of the HTML file to save
    :type filename: str
    :param show: Whether to display the plot in browser
    :type show: bool
    :param height: Height of the figure in pixels (None for auto-sizing)
    :type height: Optional[int]
    :param width: Width of the figure in pixels (None for auto-sizing)
    :type width: Optional[int]
    :param template: Plotly template for styling (e.g., 'plotly_white', 'ggplot2')
    :type template: str
    :param xaxis_tickangle: Angle for x-axis labels in degrees
    :type xaxis_tickangle: int
    :param color_palette: Optional list of colors for data splits
    :type color_palette: Optional[List[str]]
    :return: Plotly figure object containing the boxplot
    :rtype: go.Figure

    Example:
        >>> import pandas as pd
        >>> error_df = pd.DataFrame({
        ...     "feature1": [0.1, 0.2, 0.3, 0.4],
        ...     "feature2": [0.2, 0.1, 0.4, 0.3],
        ...     "data_split": ["train", "train", "val", "val"]
        ... })
        >>> boxplot = boxplot_reconstruction_error(error_df, save_path="./plots")

    """
    if reconstruction_error_df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    try:
        if "data_split" in reconstruction_error_df.columns:
            # Melt the DataFrame
            melted_df = reconstruction_error_df.melt(
                id_vars=["data_split"], var_name="sensor", value_name="AE_error"
            )
            # Create the boxplot using Plotly
            fig = px.box(
                melted_df,
                x="sensor",
                y="AE_error",
                color="data_split",
                labels={
                    "sensor": "",
                    "AE_error": "Reconstruction Error (Autoencoder - Actual)",
                    "data_split": "Dataset Split",
                },
                template=template,
                color_discrete_map=color_palette,
            )

        else:
            melted_df = reconstruction_error_df.melt(
                var_name="sensor", value_name="AE_error"
            )
            fig = px.box(
                melted_df,
                x="sensor",
                y="AE_error",
                color="sensor",
                labels={
                    "sensor": "",
                    "AE_error": "Reconstruction Error (Autoencoder - Actual)",
                },
                template=template,
            )
            if color_palette is None:
                uniform_color = "#636EFA"
            else:
                uniform_color = color_palette[0]
            fig.for_each_trace(
                lambda t: t.update(marker_color=uniform_color, line_color=uniform_color)
            )

        if "data_split" in reconstruction_error_df.columns:
            showlegend = True
        else:
            showlegend = False

        # Update layout for better visualization and responsiveness
        fig.update_layout(
            showlegend=showlegend,
            xaxis_tickangle=xaxis_tickangle,
            title=dict(
                text="Autoencoder Reconstruction Error", x=0.5, xanchor="center"
            ),
            margin=dict(l=50, r=50, t=100, b=100),
            autosize=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white"),
            # Make the plot responsive to container size
            height=height,
            width=width,
            # Ensure the plot fills the container
            uirevision=True,
            # Improve responsiveness
            xaxis=dict(
                automargin=True,
                showgrid=True,
                gridcolor="lightgray",
            ),
            yaxis=dict(
                automargin=True,
                showgrid=True,
                gridcolor="lightgray",
            ),
        )

        fig.update_traces(
            hovertemplate=None,
            marker=dict(size=3),
        )

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            full_path = path / filename
            # Save with responsive configuration
            fig.write_html(
                full_path,
                include_plotlyjs=True,
                full_html=True,
                config={"responsive": True},
            )
            logger.info(f"Reconstruction error boxplot saved to {full_path}")

        if show:
            logger.info("Displaying reconstruction error boxplot.")
            fig.show()

        return fig

    except Exception as e:
        logger.error(f"Error creating reconstruction error boxplot: {str(e)}")
        raise


def create_actual_vs_reconstructed_plot(
    df_actual: pd.DataFrame,
    df_reconstructed: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "actual_vs_reconstructed.html",
    show: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None,
    template: str = "plotly_white",
) -> go.Figure:
    """
    Create an interactive plot comparing actual and reconstructed values.

    Generates a comprehensive line plot showing the comparison between original
    data and autoencoder reconstructions across all features. Combines data
    with type indicators for clear visualization of reconstruction quality.

    :param df_actual: DataFrame containing actual/original values
    :type df_actual: pd.DataFrame
    :param df_reconstructed: DataFrame containing reconstructed values from autoencoder
    :type df_reconstructed: pd.DataFrame
    :param save_path: Optional directory path to save the plot HTML file
    :type save_path: Optional[str]
    :param filename: Name of the HTML file to save
    :type filename: str
    :param show: Whether to display the plot in browser
    :type show: bool
    :param height: Height of the figure in pixels
    :type height: Optional[int]
    :param width: Width of the figure in pixels
    :type width: Optional[int]
    :param template: Plotly template for styling (e.g., 'plotly_white', 'ggplot2')
    :type template: str
    :return: Plotly figure object containing the comparison plot
    :rtype: go.Figure

    Example:
        >>> import pandas as pd
        >>> actual_df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> reconstructed_df = pd.DataFrame({"feature1": [1.1, 1.9, 3.1], "feature2": [3.9, 5.2, 5.8]})
        >>> plot = create_actual_vs_reconstructed_plot(actual_df, reconstructed_df, save_path="./plots")

    """
    try:
        # Combine actual and reconstructed data
        df_combined = pd.concat([df_actual, df_reconstructed])

        # Create the plot
        fig = px.line(
            df_combined,
            x=df_combined.index,
            y=df_combined.columns[:-1],  # Exclude the 'type' column
            color="type",
            title="Actual vs Reconstructed Values",
            labels={"value": "Value", "index": "Time Step", "type": "Data Type"},
            template=template,
        )

        # Update layout
        fig.update_layout(
            height=height,
            width=width,
            showlegend=True,
            legend_title="Data Type",
            xaxis_title="Time Step",
            yaxis_title="Value",
        )

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            full_path = path / filename
            fig.write_html(full_path)
            logger.info(f"Actual vs reconstructed plot saved to {full_path}")

        if show:
            fig.show()

        return fig

    except Exception as e:
        logger.error(f"Error creating actual vs reconstructed plot: {str(e)}")
        raise


def plot_corrected_data(
    actual_data_df: pd.DataFrame,
    autoencoder_output_df: pd.DataFrame,
    anomaly_mask: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "corrected_data_plot.html",
    show: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
    template: str = "plotly_white",
    color_palette: Optional[List[str]] = None,
) -> go.Figure:
    """
    Plot original sensor data, autoencoder reconstruction, and corrected (replaced) anomaly points.

    Creates comprehensive visualizations showing the data correction process by displaying
    original data, autoencoder reconstructions, and specifically highlighting points
    that were identified as anomalies and replaced with reconstructed values.

    :param actual_data_df: Original sensor data including the context window
    :type actual_data_df: pd.DataFrame
    :param autoencoder_output_df: Autoencoder output after context window removal
    :type autoencoder_output_df: pd.DataFrame
    :param anomaly_mask: DataFrame of boolean values indicating where values were categorized as anomalies
    :type anomaly_mask: pd.DataFrame
    :param save_path: Optional directory path to save the plot as an HTML file
    :type save_path: Optional[str]
    :param filename: Filename to use if saving the plot
    :type filename: str
    :param show: Whether to display the plot in a browser window
    :type show: bool
    :param height: Optional height of the figure in pixels
    :type height: Optional[int]
    :param width: Optional width of the figure in pixels
    :type width: Optional[int]
    :param template: Plotly template for figure styling (e.g., 'plotly_white', 'ggplot2')
    :type template: str
    :param color_palette: Optional list of colors to use for plotting each sensor
    :type color_palette: Optional[List[str]]
    :return: Plotly figure object with actual, reconstructed, and corrected data traces
    :rtype: go.Figure
    :raises ValueError: If input DataFrames have mismatched lengths, columns, or invalid structure
    :raises Exception: If an error occurs during plotting or file saving

    Example:
        >>> import pandas as pd
        >>> actual_df = pd.DataFrame({"sensor1": [1, 2, 3, 4], "sensor2": [5, 6, 7, 8]})
        >>> reconstructed_df = pd.DataFrame({"sensor1": [1.1, 1.9, 3.1, 3.9], "sensor2": [5.2, 5.8, 7.1, 7.9]})
        >>> mask_df = pd.DataFrame({"sensor1": [False, True, False, True], "sensor2": [True, False, True, False]})
        >>> plot = plot_corrected_data(actual_df, reconstructed_df, mask_df, save_path="./plots")

    """

    # Define context offset based on the length of the actual vs outencoder outut
    context_offset = len(actual_data_df.index) - len(autoencoder_output_df.index)
    expected_autoencoder_length = len(actual_data_df) - context_offset

    if context_offset < 0:
        raise ValueError("context_offset must be a positive integer")
    if len(autoencoder_output_df) != expected_autoencoder_length:
        raise ValueError(
            f"Autoencoder output rows {len(autoencoder_output_df)} do not match expected length"
            f"{expected_autoencoder_length} (actual data rows {len(actual_data_df)} minus context offset {context_offset})"
        )
    if len(anomaly_mask) != len(actual_data_df) - context_offset:
        raise ValueError(
            f"Anomaly mask rows {len(anomaly_mask)} do not match actual data rows {len(actual_data_df)} with context offset {context_offset}"
        )
    if not autoencoder_output_df.columns.equals(actual_data_df.columns):
        raise ValueError(
            f"Autoencoder output columns ({autoencoder_output_df.columns})"
            f" do match actual data columns ({actual_data_df.columns})"
        )
    anomaly_cols = [col for col in anomaly_mask.columns if col != "data_split"]
    actual_cols = list(actual_data_df.columns)
    if set(anomaly_cols) != set(actual_cols):
        raise ValueError(
            f"Anomaly mask columns ({anomaly_mask.columns}) "
            f"do not match actual data columns ({actual_data_df.columns})"
        )

    try:

        fig = go.Figure()
        sensor_columns = actual_data_df.columns
        if color_palette is None:
            color_palette = pc.qualitative.Plotly

        for i, col in enumerate(sensor_columns):
            color = color_palette[i % len(color_palette)]

            # Plot actual sensor data
            fig.add_trace(
                go.Scatter(
                    x=actual_data_df.index,
                    y=actual_data_df[col],
                    mode="lines",
                    name=f"{col} - Original",
                    line=dict(color=color, width=1),
                    opacity=0.4,
                    # legendgroup=col,
                    # legendgrouptitle={"text": col},
                    showlegend=True,
                )
            )

            # Plot autoencoder reconstructed sensor data
            autoencoder_idx = autoencoder_output_df.index
            shifted_idx = autoencoder_idx + context_offset
            fig.add_trace(
                go.Scatter(
                    x=shifted_idx,
                    y=autoencoder_output_df[col],
                    mode="lines",
                    name=f"{col} - Autoencoder Output",
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.8,
                    # legendgroup=col,
                    showlegend=True,
                )
            )

            # Plot replaced points based on anomaly mask
            mask = anomaly_mask[col]
            replaced_idx = autoencoder_output_df.index[mask] + context_offset
            fig.add_trace(
                go.Scatter(
                    x=replaced_idx,
                    y=autoencoder_output_df.loc[mask, col],
                    mode="markers",
                    name=f"{col} - Replaced",
                    marker=dict(size=6, symbol="x", color=color),
                    # legendgroup=col,
                    showlegend=True,
                )
            )

        fig.update_layout(
            title=dict(text="Autoencoder Corrected Data", x=0.5, xanchor="center"),
            xaxis_title="Time Step",
            yaxis_title="Value",
            showlegend=True,
            hovermode="x unified",
            height=height,
            width=width,
            template=template,
        )

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            full_path = path / filename
            fig.write_html(full_path)
            logger.info(f"Corrected data plot saved to {full_path}")

        if show:
            fig.show()

        return fig
    except Exception as e:
        logger.error(f"Error creating corrected data plot: {str(e)}")
        raise


def plot_anomaly_proportions(
    anomaly_mask: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "anomaly_proportions_plot.html",
    show: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
    template: str = "plotly_white",
    color_palette: Optional[List[str]] = None,
    xaxis_tickangle: int = -45,
) -> go.Figure:
    """
    Generate a bar chart showing anomaly proportions by sensor and data split.

    Creates interactive bar charts displaying the proportion of anomalies for each
    sensor across different dataset splits (train/validation/test). Anomaly proportions
    are calculated as the number of anomalies divided by the total number of observations
    for each sensor, providing insights into data quality patterns.

    :param anomaly_mask: DataFrame containing boolean anomaly mask (True = anomaly) and a 'data_split' column
    :type anomaly_mask: pd.DataFrame
    :param save_path: Optional directory path to save the output HTML plot
    :type save_path: Optional[str]
    :param filename: Filename for the saved plot (HTML format)
    :type filename: str
    :param show: Whether to display the plot interactively in browser
    :type show: bool
    :param height: Optional height of the figure in pixels
    :type height: Optional[int]
    :param width: Optional width of the figure in pixels
    :type width: Optional[int]
    :param template: Plotly layout template to use (e.g., 'plotly_white', 'ggplot2')
    :type template: str
    :param color_palette: Optional list of color hex codes or names to use per data split
    :type color_palette: Optional[List[str]]
    :param xaxis_tickangle: Angle for x-axis labels in degrees
    :type xaxis_tickangle: int
    :return: Plotly Figure object containing the bar chart
    :rtype: go.Figure
    :raises ValueError: If required columns are missing or data is invalid
    :raises Exception: If plot creation or file saving fails

    Example:
        >>> import pandas as pd
        >>> mask_df = pd.DataFrame({
        ...     "sensor1": [True, False, True, False],
        ...     "sensor2": [False, True, False, True],
        ...     "data_split": ["train", "train", "val", "val"]
        ... })
        >>> plot = plot_anomaly_proportions(mask_df, save_path="./plots")

    """
    if "data_split" not in anomaly_mask.columns:
        raise ValueError("Anomaly mask must contain a 'data_split' column.")

    sensor_columns = [col for col in anomaly_mask.columns if col != "data_split"]
    data_splits = anomaly_mask["data_split"].dropna().unique()

    if not sensor_columns:
        raise ValueError("Anomaly mask must contain sensor columns.")
    if len(data_splits) == 0:
        raise ValueError("Anomaly mask must contain at least one unique data split.")

    try:

        fig = go.Figure()

        if color_palette is None:
            color_palette = pc.qualitative.Plotly

        # Go through each data split and calculate proportions of anomalies per sensor
        for i, split in enumerate(data_splits):
            color = color_palette[i % len(color_palette)]
            split_mask = anomaly_mask[anomaly_mask["data_split"] == split]
            proportions = (
                split_mask[sensor_columns].sum() / split_mask[sensor_columns].count()
            )
            fig.add_trace(
                go.Bar(
                    x=sensor_columns,
                    y=proportions,
                    name=split,
                    marker=dict(color=color),
                )
            )

        fig.update_layout(
            title=dict(
                text="Anomaly Proportions",
                x=0.5,
                xanchor="center",
            ),
            yaxis_title="Anomaly Proportion (Given a Sensor and Data Split)",
            showlegend=True,
            legend_title="Dataset Split",
            hovermode="x unified",
            height=height,
            width=width,
            template=template,
            xaxis_tickangle=xaxis_tickangle,
        )

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            full_path = path / filename
            fig.write_html(full_path)
            logger.info(f"Anomaly proportions plot saved to {full_path}")

        if show:
            fig.show()

        return fig

    except Exception as e:
        logger.error(f"Error creating corrected data plot: {str(e)}")
        raise
