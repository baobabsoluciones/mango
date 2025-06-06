from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import f_oneway

from mango.logging import get_configured_logger
from mango_time_series.models.utils.plots import create_error_analysis_dashboard
from mango_time_series.models.utils.processing import time_series_split

logger = get_configured_logger()


def calculate_reconstruction_error(
    x_converted: np.ndarray, x_hat: np.ndarray
) -> np.ndarray:
    """
    Calculate the reconstruction error matrix between original and reconstructed data.

    :param x_converted: Original data array
    :type x_converted: np.ndarray
    :param x_hat: Reconstructed data array
    :type x_hat: np.ndarray
    :return: Matrix of absolute differences between original and reconstructed data
    :rtype: np.ndarray
    :raises ValueError: If input arrays have different shapes
    """
    if x_converted.shape != x_hat.shape:
        raise ValueError(
            f"Input arrays must have the same shape. Got {x_converted.shape} and {x_hat.shape}"
        )
    return np.abs(x_converted - x_hat)


def analyze_error_by_columns(
    error_matrix: np.ndarray,
    column_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> pd.DataFrame:
    """
    Analyze reconstruction error distribution by columns with interactive Plotly visualizations.

    :param error_matrix: Matrix of reconstruction errors
    :type error_matrix: np.ndarray
    :param column_names: List of column names for the features
    :type column_names: Optional[List[str]]
    :param save_path: Path to save the generated plots
    :type save_path: Optional[str]
    :param show: Whether to display the plots
    :type show: bool
    :return: DataFrame with error data
    :rtype: pd.DataFrame
    :raises ValueError: If error_matrix is empty or if column_names length doesn't match error_matrix columns
    """
    if error_matrix.size == 0:
        raise ValueError("Error matrix cannot be empty")

    if column_names is None:
        column_names = [f"Feature_{i}" for i in range(error_matrix.shape[1])]
    elif len(column_names) != error_matrix.shape[1]:
        raise ValueError(
            f"Number of column names ({len(column_names)}) must match number of columns in error matrix ({error_matrix.shape[1]})"
        )

    error_df = pd.DataFrame(error_matrix, columns=column_names)

    try:
        create_error_analysis_dashboard(
            error_df=error_df, save_path=save_path, show=show
        )
        return error_df

    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        raise


def reconstruction_error(
    actual_data_df: pd.DataFrame,
    autoencoder_output_df: pd.DataFrame,
    context_window: int = 10,
    time_step_to_check: Union[int, List[int]] = [9],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    save_path: Optional[str] = None,
    filename: str = "reconstruction_error.csv",
) -> pd.DataFrame:
    """
    Calculate and optionally save reconstruction error between actual sensor data and autoencoder output.

    :param actual_data_df: Original sensor data
    :type actual_data_df: pd.DataFrame
    :param autoencoder_output_df: Autoencoder output
    :type autoencoder_output_df: pd.DataFrame
    :param context_window: Number of initial rows used as context by the AE
    :type context_window: int
    :param time_step_to_check: Time step to predict within the window
    :type time_step_to_check: Union[int, List[int]]
    :param train_size: Proportion of training data
    :type train_size: float
    :param val_size: Proportion of validation data
    :type val_size: float
    :param test_size: Proportion of test data
    :type test_size: float
    :param save_path: Optional directory to save the output CSV
    :type save_path: Optional[str]
    :param filename: Filename for saved CSV
    :type filename: str
    :return: DataFrame with reconstruction error and associated data_split
    :rtype: pd.DataFrame
    :raises ValueError: If input DataFrames have different lengths or if split sizes don't sum to 1
    """
    # Validate input parameters
    context_offset = context_window - 1
    expected_autoencoder_length = len(actual_data_df) - context_offset

    if len(autoencoder_output_df) != expected_autoencoder_length:
        raise ValueError(
            f"Autoencoder output rows {len(autoencoder_output_df)} do not match expected length "
            f"{expected_autoencoder_length} (actual data rows {len(actual_data_df)} minus context offset {context_offset})"
        )
    if isinstance(time_step_to_check, list):
        time_step_to_check = time_step_to_check[0]
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError(
            f"train_size + val_size + test_size must equal 1, but got {train_size + val_size + test_size}"
        )
    if time_step_to_check < 0 or time_step_to_check > context_window - 1:
        raise ValueError(
            f"time_step_to_check must be between 0 and {context_window - 1}, "
            f"but got {time_step_to_check}"
        )
    if context_window < 1:
        raise ValueError("context_window must be a positive integer")
    if not autoencoder_output_df.columns.equals(actual_data_df.columns):
        raise ValueError(
            f"Autoencoder output columns ({autoencoder_output_df.columns})"
            f" do not match actual data columns ({actual_data_df.columns})"
        )

    try:
        actual_data_df = actual_data_df.iloc[
            time_step_to_check : time_step_to_check + len(autoencoder_output_df)
        ].reset_index(drop=True)

        # Drop the 'type'=reconstructed column
        if "type" in autoencoder_output_df.columns:
            autoencoder_output_df = autoencoder_output_df.drop(columns=["type"])

        # Define data splits ( train, validation, test) for autoencoder output
        autoencoder_train_df, autoencoder_val_df, autoencoder_test_df = (
            time_series_split(autoencoder_output_df, train_size, val_size, test_size)
        )

        autoencoder_train_df = autoencoder_train_df.copy()
        autoencoder_val_df = autoencoder_val_df.copy()
        autoencoder_test_df = autoencoder_test_df.copy()

        autoencoder_train_df.loc[:, "data_split"] = "train"
        autoencoder_val_df.loc[:, "data_split"] = "validation"
        autoencoder_test_df.loc[:, "data_split"] = "test"

        # Combine dataframes
        autoencoder_output_df = pd.concat(
            [autoencoder_train_df, autoencoder_val_df, autoencoder_test_df], axis=0
        ).reset_index(drop=True)

        autoencoder_output_df.loc[:, "data_split"] = pd.Categorical(
            autoencoder_output_df["data_split"],
            categories=["train", "validation", "test"],
            ordered=True,
        )

        # Generate reconstruction error DataFrame
        reconstruction_error_df = (
            autoencoder_output_df.drop(columns=["data_split"]) - actual_data_df
        )
        reconstruction_error_df.loc[:, "data_split"] = autoencoder_output_df[
            "data_split"
        ]

        # Warn if some sensors have higher reconstruction error than others
        sensor_columns = [
            col for col in reconstruction_error_df.columns if col != "data_split"
        ]
        mean_errors = reconstruction_error_df[sensor_columns].abs().mean()
        median_mean_error = mean_errors.median()
        threshold = 3 * median_mean_error
        high_error_sensors = mean_errors[mean_errors > threshold]
        if not high_error_sensors.empty:
            logger.warning(
                "Sensors with high reconstruction error compared to others:\n"
                + high_error_sensors.sort_values(ascending=False).to_string(
                    float_format="%.4f"
                )
            )

        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            file_path = path / filename
            reconstruction_error_df.to_csv(file_path, index=False, float_format="%.4f")
            logger.info(f"Reconstruction error saved to {file_path}")

        return reconstruction_error_df

    except Exception as e:
        logger.error(f"Error calculating reconstruction error: {str(e)}")
        raise


def anova_reconstruction_error(
    reconstruction_error_df: pd.DataFrame,
    f_stat_threshold: float = 300.0,
) -> pd.DataFrame:
    """
    Perform one-way ANOVA to test if reconstruction errors vary across data splits for each sensor.

    :param reconstruction_error_df: DataFrame with reconstruction error and 'data_split' column
    :type reconstruction_error_df: pd.DataFrame
    :param F_threshold: Minimum F-statistic to consider the variability practically significant
    :type F_threshold: float
    :return: DataFrame with F-statistics and p-values per sensor
    :rtype: pd.DataFrame
    """
    try:
        results = []
        sensor_columns = [
            col for col in reconstruction_error_df.columns if col != "data_split"
        ]

        # Loop through each sensor column and perform one-way ANOVA across data splits
        for sensor in sensor_columns:
            group = reconstruction_error_df.groupby("data_split")[sensor].apply(list)

            # Skip sensors with less than 2 data splits
            if len(group) < 2:
                logger.warning(
                    f"Not enough data splits to calculate variability for sensor {sensor}. Skipping."
                )
                continue

            # Perform one-way ANOVA
            f_stat, p_val = f_oneway(*group)
            results.append(
                {
                    "sensor": sensor,
                    "F_statistic": f_stat,
                    "p_value": p_val,
                }
            )

            # Log if F-statistic exceeds threshold
            if f_stat > f_stat_threshold:
                logger.warning(
                    f"Sensor {sensor} has a high F-statistic ({f_stat:.4f}) indicating significant variability in reconstruction error across data splits."
                )

        results_df = pd.DataFrame(results)
        return results_df

    except Exception as e:
        logger.error(
            f"Error computing one-way ANOVA tests across sensor data splits: {str(e)}"
        )
        raise


def reconstruction_error_summary(
    reconstruction_error_df: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "reconstruction_error_summary.csv",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate and optionally save summary statistics (mean and std) for reconstruction error
    grouped by data split.

    :param reconstruction_error_df: DataFrame with reconstruction error and 'data_split' column
    :type reconstruction_error_df: pd.DataFrame
    :param save_path: Optional path to save the summary CSV
    :type save_path: Optional[str]
    :param filename: Filename to use for the saved CSV
    :type filename: str
    :param threshold: Relative threshold for flagging large differences in mean/std across splits
    :type threshold: float
    :return: Summary statistics MultiIndex column DataFrame
    :rtype: pd.DataFrame
    :raises ValueError: If required columns are missing or if data is empty
    """
    if reconstruction_error_df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    if "data_split" not in reconstruction_error_df.columns:
        raise ValueError("Input DataFrame must contain 'data_split' column")

    try:
        summary_stats = reconstruction_error_df.groupby("data_split").agg(
            ["mean", "std"]
        )

        summary_stats = summary_stats.T
        summary_stats.index.names = ["sensor", "statistic"]
        summary_stats = summary_stats.unstack(level=1)
        summary_stats.columns = [
            f"{split}_{stat}" for split, stat in summary_stats.columns
        ]

        # Reorder columns to group by statistic (mean, then std)
        split_order = ["train", "validation", "test"]
        column_order = [
            f"{split}_{stat}" for stat in ["mean", "std"] for split in split_order
        ]
        summary_stats = summary_stats[
            [col for col in column_order if col in summary_stats.columns]
        ]

        mean_columns = [col for col in summary_stats.columns if col.endswith("_mean")]
        std_columns = [col for col in summary_stats.columns if col.endswith("_std")]

        for sensor, row in summary_stats.iterrows():
            mean_values = row[mean_columns].dropna()
            std_values = row[std_columns].dropna()
            if len(mean_values) >= 2:
                diff_mean = mean_values.max() - mean_values.min()
                if diff_mean / mean_values.max() > threshold:
                    logger.warning(
                        f"Sensor {sensor} has a high difference in mean reconstruction error across splits."
                    )
            if len(std_values) >= 2:
                diff_std = std_values.max() - std_values.min()
                if diff_std / std_values.max() > threshold:
                    logger.warning(
                        f"Sensor {sensor} has a high difference in std reconstruction error across splits."
                    )

        # Save if a path is provided
        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            full_path = path / filename
            summary_stats.to_csv(full_path, index=True, float_format="%.4f")
            logger.info(f"Reconstruction error summary saved to {full_path}")

        return summary_stats

    except Exception as e:
        logger.error(f"Error generating reconstruction error summary: {str(e)}")
        raise


def std_error_threshold(
    reconstruction_error_df: pd.DataFrame,
    std_threshold: float = 3.0,
    save_path: Optional[str] = None,
    anomaly_mask_filename: str = "std_anomaly_mask.csv",
    anomaly_proportions_filename: str = "std_anomaly_proportions.csv",
) -> pd.DataFrame:
    """
    Identify anomalies using a standard deviation threshold over reconstruction error.
    Considers all time series data for a given sensor.

    :param reconstruction_error_df: DataFrame with AE reconstruction errors and 'data_split'
    :type reconstruction_error_df: pd.DataFrame
    :param std_threshold: Threshold in terms of standard deviations from the mean error
    :type std_threshold: float
    :param save_path: Optional path to save outputs
    :type save_path: Optional[str]
    :param anomaly_mask_filename: CSV filename for the boolean anomaly mask
    :type anomaly_mask_filename: str
    :param anomaly_proportions_filename: CSV filename for anomaly rate summary
    :type anomaly_proportions_filename: str
    :return: DataFrame boolean mask of anomalies (True for anomalies, False otherwise)
    :rtype: pd.DataFrame
    :raises ValueError: If required columns are missing, if data is empty, or if std_threshold is negative
    """
    if reconstruction_error_df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    if "data_split" not in reconstruction_error_df.columns:
        raise ValueError("Input DataFrame must contain 'data_split' column")
    if std_threshold < 0:
        raise ValueError("std_threshold must be non-negative")

    try:
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
            anomaly_mask[col] = np.abs(
                reconstruction_error_df[col] - mean_errors[col]
            ) > (std_threshold * std_errors[col])
        anomaly_mask["data_split"] = reconstruction_error_df["data_split"]

        # Calculate anomalies per sensor and proportion of anomalies
        anomaly_counts = anomaly_mask.groupby("data_split")[sensor_columns].sum()
        total_counts = anomaly_mask.groupby("data_split")[sensor_columns].count()
        anomaly_proportions = anomaly_counts / total_counts
        high_anomaly_sensors = anomaly_proportions.mean().sort_values(ascending=False)
        logger.info(
            "Sensors with highest anomaly proportions (# anomalies / sensor data count):\n"
            + high_anomaly_sensors.head().to_string(float_format="%.4f")
        )

        # Save if a path is provided
        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            mask_full_path = path / anomaly_mask_filename
            prop_full_path = path / anomaly_proportions_filename
            anomaly_mask.to_csv(mask_full_path, index=False, float_format="%.4f")
            anomaly_proportions.to_csv(prop_full_path, index=True, float_format="%.4f")
            logger.info(f"Anomaly mask saved to {mask_full_path}")
            logger.info(f"Anomaly proportions saved to {prop_full_path}")

        return anomaly_mask

    except Exception as e:
        logger.error(f"Error calculating std error threshold: {str(e)}")
        raise


def corrected_data(
    actual_data_df: pd.DataFrame,
    autoencoder_output_df: pd.DataFrame,
    anomaly_mask: pd.DataFrame,
    context_window: int = 10,
    save_path: Optional[str] = None,
    filename: str = "corrected_data.csv",
) -> pd.DataFrame:
    """
    Replace anomalous values in the original sensor data with autoencoder reconstructed values.

    :param actual_data_df: Original sensor data
    :type actual_data_df: pd.DataFrame
    :param autoencoder_output_df: Autoencoder output
    :type autoencoder_output_df: pd.DataFrame
    :param anomaly_mask: DataFrame of boolean values indicating where to apply corrections
    :type anomaly_mask: pd.DataFrame
    :param context_window: Number of rows at the start of the data where AE does not have output
    :type context_window: int
    :param save_path: Optional path to save the corrected data as CSV
    :type save_path: Optional[str]
    :param filename: Filename to use if saving the corrected data
    :type filename: str
    :return: DataFrame with corrected sensor data
    :rtype: pd.DataFrame
    :raises ValueError: If input DataFrames have different lengths or if context_window is invalid
    """
    # Validate input parameters
    context_offset = context_window - 1
    expected_autoencoder_length = len(actual_data_df) - context_offset

    if len(autoencoder_output_df) != expected_autoencoder_length:
        raise ValueError(
            f"Autoencoder output rows {len(autoencoder_output_df)} do not match expected length"
            f"{expected_autoencoder_length} (actual data rows {len(actual_data_df)} minus context offset {context_offset})"
        )
    if len(anomaly_mask) != len(actual_data_df) - context_offset:
        raise ValueError(
            f"Anomaly mask rows {len(anomaly_mask)} do not match actual data rows {len(actual_data_df)} with context offset {context_offset}"
        )
    if context_window < 1:
        raise ValueError("context_window must be a positive integer")
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
        # Save initial rows and adjust actual_data_df and anomaly_mask to match autoencoder_output_df
        initial_rows_df = actual_data_df.iloc[:context_offset].copy()
        actual_data_df = actual_data_df.iloc[context_offset:].reset_index(drop=True)

        # Create corrected dataset where anomalies are replaced with AE output
        corrected_data_df = actual_data_df.copy()
        for corrected_data_col in corrected_data_df.columns:
            corrected_data_df.loc[
                anomaly_mask[corrected_data_col], corrected_data_col
            ] = autoencoder_output_df.loc[
                anomaly_mask[corrected_data_col], corrected_data_col
            ]
            num_replaced = anomaly_mask[corrected_data_col].sum()
            logger.info(
                f"{num_replaced} anomalies corrected in column {corrected_data_col}"
            )

        corrected_data_df = pd.concat(
            [initial_rows_df, corrected_data_df], axis=0
        ).reset_index(drop=True)

        # Save if a path is provided
        if save_path:
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            full_path = path / filename
            corrected_data_df.to_csv(full_path, index=False, float_format="%.4f")
            logger.info(f"Corrected data saved to {full_path}")

        return corrected_data_df

    except Exception as e:
        logger.error(f"Error correcting data: {str(e)}")
        raise
