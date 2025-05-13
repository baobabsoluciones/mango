import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from pathlib import Path


def calculate_reconstruction_error(x_converted, x_hat):
    """
    Calculate the reconstruction error matrix.

    :param x_converted: Original data
    :param x_hat: Reconstructed data
    :return: Matrix of absolute differences between original and reconstructed data
    """
    return np.abs(x_converted - x_hat)

def analyze_error_by_columns(error_matrix, column_names=None, save_path=None):
    """
    Analyze reconstruction error distribution by columns with interactive Plotly visualizations.

    :param error_matrix: Matrix of reconstruction errors
    :param column_names: List of column names for the features
    :param save_path: Path to save the generated plots
    :return: DataFrame with error data
    """
    if column_names is None:
        column_names = [f"Feature_{i}" for i in range(error_matrix.shape[1])]

    error_df = pd.DataFrame(error_matrix, columns=column_names)

    # Mean error by column - barplot
    mean_errors = error_df.mean().reset_index()
    mean_errors.columns = ["Feature", "Mean Error"]
    fig1 = px.bar(
        mean_errors, x="Feature", y="Mean Error", title="Mean Error by Feature"
    )
    fig1.update_layout(xaxis_tickangle=-90)

    # Error distribution - boxplot
    fig2 = px.box(error_df, title="Error Distribution by Feature")

    # Heatmap of correlation between errors
    corr = error_df.corr()
    fig3 = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        title="Error Correlation Between Features",
    )

    if save_path:
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig1.write_html(output_dir / "mean_error_barplot.html")
        fig2.write_html(output_dir / "error_boxplot.html")
        fig3.write_html(output_dir / "error_correlation.html")

    fig1.show()
    fig2.show()
    fig3.show()

    return error_df
