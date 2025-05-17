import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from pathlib import Path
from mango_time_series.models.utils.processing import time_series_split
import matplotlib.pyplot as plt
import seaborn as sns


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


def reconstruction_error(
    actual_data_df: pd.DataFrame,
    autoencoder_output_df: pd.DataFrame,
    context_window: int = 10,
    TRAIN_SIZE: float = 0.8,
    VAL_SIZE: float = 0.1,
    TEST_SIZE: float = 0.1,
    save_path: str = None,
    filename: str = "reconstruction_error.csv",
):
    """
    Calculate and optionally save reconstruction error between actual sensor data and autoencoder output.

    :param actual_data_df: Original sensor data
    :param autoencoder_output_df: Autoencoder output
    :param context_window: Number of initial rows used as context by the AE
    :param TRAIN_SIZE: Proportion of training data
    :param VAL_SIZE: Proportion of validation data
    :param TEST_SIZE: Proportion of test data
    :param save_path: Optional directory to save the output CSV
    :param filename: Filename for saved CSV
    :return: DataFrame with reconstruction error and associated data_split
    """
    # Checks
    if len(autoencoder_output_df) != len(actual_data_df):
        raise ValueError(
            f"Autoencoder output rows {len(autoencoder_output_df)} do not match actual data rows {len(actual_data_df)}"
        )
    if not np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1.0):
        raise ValueError(
            f"TRAIN_SIZE + VAL_SIZE + TEST_SIZE must equal 1, but got {TRAIN_SIZE + VAL_SIZE + TEST_SIZE}"
        )

    # Drop first context_window rows from actual data and autoencoder output
    context_offset = context_window - 1
    actual_data_df = actual_data_df.iloc[context_offset:].reset_index(drop=True)
    autoencoder_output_df = autoencoder_output_df.iloc[context_offset:].reset_index(
        drop=True
    )

    # Drop the 'type'=reconstructed column
    if "type" in autoencoder_output_df.columns:
        autoencoder_output_df = autoencoder_output_df.drop(columns=["type"])

    # Define data splits ( train, validation, test) for autoencoder output
    autoencoder_train_df, autoencoder_val_df, autoencoder_test_df = time_series_split(
        autoencoder_output_df, TRAIN_SIZE, VAL_SIZE, TEST_SIZE
    )
    autoencoder_train_df["data_split"] = "train"
    autoencoder_val_df["data_split"] = "validation"
    autoencoder_test_df["data_split"] = "test"

    # Combine dataframes
    autoencoder_output_df = pd.concat(
        [autoencoder_train_df, autoencoder_val_df, autoencoder_test_df], axis=0
    ).reset_index(drop=True)

    # Order data_split column
    autoencoder_output_df["data_split"] = pd.Categorical(
        autoencoder_output_df["data_split"],
        categories=["train", "validation", "test"],
        ordered=True,
    )

    # Generate reconstruction error DataFrame
    reconstruction_error_df = (
        autoencoder_output_df.drop(columns=["data_split"]) - actual_data_df
    )
    # Add data_split column
    reconstruction_error_df["data_split"] = autoencoder_output_df["data_split"]
    # Rename columns
    reconstruction_error_df.rename(
        columns={
            col: col.split("-")[0] + "_AE_error"
            for col in reconstruction_error_df.columns
            if col != "data_split"
        },
        inplace=True,
    )

    if save_path:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / filename
        reconstruction_error_df.to_csv(file_path, index=False)
        print(f"Reconstruction error saved to {file_path}")

    return reconstruction_error_df


def reconstruction_error_summary(
    reconstruction_error_df: pd.DataFrame,
    save_path: str = None,
    filename: str = "reconstruction_error_summary.csv",
):
    """
    Generate and optionally save summary statistics (mean and std) for reconstruction error
    grouped by data split.

    :param reconstruction_error_df: DataFrame with reconstruction error and 'data_split' column
    :param save_path: Optional path to save the summary CSV
    :param filename: Filename to use for the saved CSV
    :return: Summary statistics MultiIndex column DataFrame
    """
    summary_stats = reconstruction_error_df.groupby("data_split").agg(["mean", "std"])

    # Save if a path is provided
    if save_path:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / filename
        summary_stats.to_csv(full_path)
        print(f"Reconstruction error summary saved to {full_path}")

    return summary_stats


def reconstruction_error_boxplot(
    reconstruction_error_df: pd.DataFrame,
    save_path: str = None,
    filename: str = "reconstruction_error_boxplot.png",
):
    """
    Generate and optionally save a boxplot for reconstruction error.

    :param reconstruction_error_df: DataFrame with reconstruction error values and 'data_split'
    :param save_path: Optional path to save the plot
    :param filename: Filename to use if saving the plot
    :return: The matplotlib plt object
    """
    # Melt the Dataframe
    melted_df = reconstruction_error_df.melt(
        id_vars=["data_split"], var_name="sensor", value_name="AE_error"
    )
    # Remove "_AE_error" from sensor names for cleaner names
    melted_df["sensor"] = melted_df["sensor"].str.replace("_AE_error", "", regex=False)

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_df, x="sensor", y="AE_error", hue="data_split", fliersize=1)
    plt.title("Autoencoder Reconstruction Error")
    plt.xlabel("")
    plt.ylabel("Reconstruction Error (Autoencoder - Actual)")
    plt.tight_layout()

    # Save if a path is provided
    if save_path:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / filename
        plt.savefig(full_path)
        print(f"Reconstruction error boxplot saved to {full_path}")

    return plt


def std_error_threshold(
    reconstruction_error_df: pd.DataFrame,
    std_threshold: float = 3.0,
    save_path: str = None,
    anomaly_mask_filename: str = "std_anomaly_mask.csv",
    anomaly_proportions_filename: str = "std_anomaly_proportions.csv",
):
    """
    Identify anomalies using a standard deviation threshold over reconstruction error.
    Considers all time series data for a given sensor.

    :param reconstruction_error_df: DataFrame with AE reconstruction errors and 'data_split'
    :param std_threshold: Threshold in terms of standard deviations from the mean error
    :param save_path: Optional path to save outputs
    :param anomaly_mask_filename: CSV filename for the boolean anomaly mask
    :param anomaly_proportions_filename: CSV filename for anomaly rate summary
    :return: DataFrame boolean mask of anomalies (True for anomalies, False otherwise)
    """
    # Calculate mean and std error for each sensor column
    sensor_columns = [
        col for col in reconstruction_error_df.columns if col != "data_split"
    ]
    mean_errors = reconstruction_error_df[sensor_columns].mean()
    std_errors = reconstruction_error_df[sensor_columns].std()

    # Create sensor based anomaly mask (showing which data points are outside the std threshold)
    anomaly_mask = pd.DataFrame(
        data=False, index=reconstruction_error_df.index, columns=sensor_columns
    )
    for col in sensor_columns:
        anomaly_mask[col] = np.abs(reconstruction_error_df[col] - mean_errors[col]) > (
            std_threshold * std_errors[col]
        )
    anomaly_mask["data_split"] = reconstruction_error_df["data_split"]

    # Calculate anomalies per sensor and proportion of anomalies
    anomaly_counts = anomaly_mask.groupby("data_split")[sensor_columns].sum()
    total_counts = anomaly_mask.groupby("data_split")[sensor_columns].count()
    anomaly_proportions = anomaly_counts / total_counts

    # Rename anomaly proportions columns
    anomaly_proportions.rename(
        columns={
            col: col + "_prop"
            for col in anomaly_proportions.columns
            if "_AE_error" in col
        },
        inplace=True,
    )

    # Save if a path is provided
    if save_path:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        mask_full_path = path / anomaly_mask_filename
        prop_full_path = path / anomaly_proportions_filename
        anomaly_mask.to_csv(mask_full_path)
        anomaly_proportions.to_csv(prop_full_path)
        print(f"Anomaly mask saved to {mask_full_path}")
        print(f"Anomaly proportions saved to {prop_full_path}")

    return anomaly_mask


def corrected_data(
    actual_data_df: pd.DataFrame,
    autoencoder_output_df: pd.DataFrame,
    anomaly_mask: pd.DataFrame,
    context_window: int = 10,
    save_path: str = None,
    filename: str = "corrected_data.csv",
):
    """
    Replace anomalous values in the original sensor data with autoencoder reconstructed values.

    :param actual_data_df: Original sensor data
    :param autoencoder_output_df: Autoencoder output
    :param anomaly_mask: DataFrame of boolean values indicating where to apply corrections
    :param context_window: Number of rows at the start of the data where AE does not have output
    :param save_path: Optional path to save the corrected data as CSV
    :param filename: Filename to use if saving the corrected data
    :return: DataFrame with corrected sensor data
    """
    # Drop first context_window rows from actual data and AE output and save them
    context_offset = context_window - 1
    initial_rows_df = actual_data_df.iloc[:context_offset].copy()
    actual_data_df = actual_data_df.iloc[context_offset:].reset_index(drop=True)
    autoencoder_output_df = autoencoder_output_df.iloc[context_offset:].reset_index(
        drop=True
    )

    # Create corrected dataset where anomalies are replaced with AE output
    corrected_data_df = actual_data_df.copy()
    for corrected_data_col in corrected_data_df.columns:
        anomaly_mask_col = corrected_data_col.split("-")[0] + "_AE_error"
        corrected_data_df.loc[anomaly_mask[anomaly_mask_col], corrected_data_col] = (
            autoencoder_output_df.loc[
                anomaly_mask[anomaly_mask_col], corrected_data_col
            ]
        )
        num_replaced = anomaly_mask[anomaly_mask_col].sum()
        print(f"{num_replaced} anomalies corrected in column {corrected_data_col}")

    corrected_data_df = pd.concat(
        [initial_rows_df, corrected_data_df], axis=0
    ).reset_index(drop=True)

    # Save if a path is provided
    if save_path:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / filename
        corrected_data_df.to_csv(full_path, index=False)
        print(f"Corrected data saved to {full_path}")

    return corrected_data_df
