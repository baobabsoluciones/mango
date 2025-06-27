from typing import List, Optional

import numpy as np
import pandas as pd

from mango.logging import get_configured_logger
from mango_time_series.models.utils.plots import create_error_analysis_dashboard
from mango_time_series.models.utils.processing import save_csv

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
    threshold_factor: int = 3,
    save_path: Optional[str] = None,
    filename: str = "reconstruction_error.csv",
) -> pd.DataFrame:
    """
    Calculate and optionally save reconstruction error between actual data and autoencoder output.

    :param actual_data_df: Original data
    :type actual_data_df: pd.DataFrame
    :param autoencoder_output_df: Autoencoder output
    :type autoencoder_output_df: pd.DataFrame
    :param threshold_factor: Multiplier for median reconstruction error to flag high-error features
    :type threshold_factor: int
    :param save_path: Optional directory to save the output CSV
    :type save_path: Optional[str]
    :param filename: Filename for saved CSV
    :type filename: str
    :return: DataFrame with reconstruction error
    :rtype: pd.DataFrame
    :raises ValueError: If input DataFrames have different lengths or columns
    """
    # Validate input parameters
    if len(autoencoder_output_df) != len(actual_data_df):
        raise ValueError(
            f"Autoencoder output rows ({len(autoencoder_output_df)}) "
            f"do not match actual data rows ({len(actual_data_df)})"
        )
    if not autoencoder_output_df.columns.equals(actual_data_df.columns):
        raise ValueError(
            f"Autoencoder output columns ({autoencoder_output_df.columns}) "
            f"do not match actual data columns ({actual_data_df.columns})"
        )

    try:
        feature_columns = [
            col for col in autoencoder_output_df.columns if col != "data_split"
        ]
        # Generate reconstruction error DataFrame
        reconstruction_error_df = (
            autoencoder_output_df[feature_columns] - actual_data_df[feature_columns]
        )

        # Add back data_split if it was originally present
        if "data_split" in actual_data_df.columns:
            reconstruction_error_df.insert(
                0, "data_split", actual_data_df["data_split"]
            )

        # Warn if some features have higher reconstruction error than others
        mean_errors = reconstruction_error_df[feature_columns].abs().mean()
        median_mean_error = mean_errors.median()
        threshold = threshold_factor * median_mean_error
        high_error_features = mean_errors[mean_errors > threshold]
        if not high_error_features.empty:
            logger.warning(
                "Features with high reconstruction error compared to others:\n"
                + high_error_features.sort_values(ascending=False).to_string(
                    float_format="%.4f"
                )
            )

        # Save if a path is provided
        if save_path:
            save_csv(
                data=reconstruction_error_df,
                save_path=save_path,
                filename=filename,
            )

        return reconstruction_error_df

    except Exception as e:
        logger.error(f"Error calculating reconstruction error: {str(e)}")
        raise


def reconstruction_error_summary(
    reconstruction_error_df: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "reconstruction_error_summary.csv",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate and optionally save summary statistics (mean and std) for reconstruction error
    grouped by data split (if provided).

    :param reconstruction_error_df: DataFrame with reconstruction error
    :type reconstruction_error_df: pd.DataFrame
    :param save_path: Optional path to save the summary CSV
    :type save_path: Optional[str]
    :param filename: Filename to use for the saved CSV
    :type filename: str
    :param threshold: Relative threshold for flagging large differences in mean/std across splits
    :type threshold: float
    :return: Summary statistics MultiIndex column DataFrame
    :rtype: pd.DataFrame
    :raises ValueError: If data is empty
    """
    if reconstruction_error_df.empty:
        raise ValueError("Input DataFrame cannot be empty")

    try:
        if "data_split" in reconstruction_error_df.columns:
            split_order = ["train", "validation", "test"]
            if set(split_order) != set(reconstruction_error_df["data_split"].unique()):
                raise ValueError(
                    f"data_split in reconstruction_error_df must be train, validation, test."
                )
            summary_stats = reconstruction_error_df.groupby("data_split").agg(
                ["mean", "std"]
            )
            summary_stats = summary_stats.T.unstack(level=1)
            summary_stats.index.rename("feature", inplace=True)
            summary_stats.columns = [
                f"{split}_{stat}" for split, stat in summary_stats.columns
            ]

            # Reorder columns: mean then std, across train, val, test
            column_order = [
                f"{split}_{stat}" for stat in ["mean", "std"] for split in split_order
            ]
            summary_stats = summary_stats[column_order]

            mean_columns = [
                col for col in summary_stats.columns if col.endswith("_mean")
            ]
            std_columns = [col for col in summary_stats.columns if col.endswith("_std")]

            for feature, row in summary_stats.iterrows():
                mean_values = row[mean_columns].dropna()
                std_values = row[std_columns].dropna()
                if len(mean_values) >= 2 and len(std_values) >= 2:
                    diff_mean = mean_values.max() - mean_values.min()
                    max_mean = mean_values.max()
                    if max_mean > 0 and (diff_mean / max_mean) > threshold:
                        logger.warning(
                            f"{feature} relative mean error range across splits ({diff_mean:.3f}) "
                            f"exceeds threshold ({threshold:.0%}) of maximum ({max_mean:.3f})"
                        )
                    diff_std = std_values.max() - std_values.min()
                    max_std = std_values.max()
                    if max_std > 0 and (diff_std / max_std) > threshold:
                        logger.warning(
                            f"{feature} relative std error range across splits ({diff_std:.3f}) "
                            f"exceeds threshold ({threshold:.0%}) of maximum ({max_std:.3f})"
                        )
        else:
            summary_stats = reconstruction_error_df.agg(["mean", "std"])
            summary_stats = summary_stats.T
            summary_stats.index.rename("feature", inplace=True)

        # Save if a path is provided
        if save_path:
            save_csv(
                data=summary_stats,
                save_path=save_path,
                filename=filename,
                save_index=True,
            )

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
    Considers all time series data for a given feature.

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
    :return: Boolean DataFrame mask (True = anomaly, False = normal)
    :rtype: pd.DataFrame
    :raises ValueError: If required columns are missing, if data is empty, or if std_threshold is negative
    """
    if reconstruction_error_df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    if "data_split" not in reconstruction_error_df.columns:
        raise ValueError("Input DataFrame must contain 'data_split' column")
    if std_threshold <= 0:
        raise ValueError("std_threshold must be greater than 0")
    if reconstruction_error_df.isna().any().any():
        logger.warning(
            "Missing values in reconstruction_error_df used in std_error_threshold()."
        )

    try:
        # Calculate mean and std error for each feature column
        feature_columns = [
            col for col in reconstruction_error_df.columns if col != "data_split"
        ]
        mean_errors = reconstruction_error_df[feature_columns].mean()
        std_errors = reconstruction_error_df[feature_columns].std()

        # Create feature based anomaly mask (showing which data points are outside the std threshold)
        anomaly_mask = pd.DataFrame(
            np.abs(reconstruction_error_df[feature_columns] - mean_errors)
            > (std_threshold * std_errors),
            index=reconstruction_error_df.index,
            columns=feature_columns,
        )

        # Add back data_split column
        anomaly_mask.insert(0, "data_split", reconstruction_error_df["data_split"])

        # Calculate anomalies per feature and proportion of anomalies
        anomaly_groups = anomaly_mask.groupby("data_split")[feature_columns]
        anomaly_counts = anomaly_groups.sum()
        total_counts = anomaly_groups.count()
        anomaly_proportions = anomaly_counts / total_counts
        high_anomaly_features = anomaly_proportions.mean().sort_values(ascending=False)
        logger.info(
            "Features with highest anomaly proportions (# anomalies / feature data count):\n"
            + high_anomaly_features.head().to_string(float_format="%.4f")
        )

        # Save if a path is provided
        if save_path:
            save_csv(
                data=anomaly_mask,
                save_path=save_path,
                filename=anomaly_mask_filename,
            )
            save_csv(
                data=anomaly_proportions,
                save_path=save_path,
                filename=anomaly_proportions_filename,
            )

        return anomaly_mask

    except Exception as e:
        logger.error(f"Error calculating std error threshold: {str(e)}")
        raise


def corrected_data(
    actual_data_df: pd.DataFrame,
    autoencoder_output_df: pd.DataFrame,
    anomaly_mask: pd.DataFrame,
    save_path: Optional[str] = None,
    filename: str = "corrected_data.csv",
) -> pd.DataFrame:
    """
    Replace anomalous values in the original data with autoencoder reconstructed values.

    :param actual_data_df: Original data
    :type actual_data_df: pd.DataFrame
    :param autoencoder_output_df: Autoencoder output
    :type autoencoder_output_df: pd.DataFrame
    :param anomaly_mask: DataFrame of boolean values indicating where to apply corrections
    :type anomaly_mask: pd.DataFrame
    :param save_path: Optional path to save the corrected data as CSV
    :type save_path: Optional[str]
    :param filename: Filename to use if saving the corrected data
    :type filename: str
    :return: DataFrame with corrected data
    :rtype: pd.DataFrame
    :raises ValueError: If input DataFrames have different lengths or columns
    """
    # Validate input parameters
    if len(autoencoder_output_df) != len(actual_data_df):
        raise ValueError(
            f"Autoencoder output length ({len(autoencoder_output_df)}) "
            f"does not match actual data length ({len(actual_data_df)})."
        )
    if len(anomaly_mask) != len(actual_data_df):
        raise ValueError(
            f"Anomaly mask length ({len(anomaly_mask)}) "
            f"does not match actual data length ({len(actual_data_df)})."
        )
    feature_cols = [col for col in autoencoder_output_df.columns if col != "data_split"]
    if list(feature_cols) != list(actual_data_df.columns):
        raise ValueError(
            f"Autoencoder output feature columns ({feature_cols}) "
            f"do not match actual data columns ({list(actual_data_df.columns)})"
        )
    if not autoencoder_output_df.columns.equals(anomaly_mask.columns):
        raise ValueError(
            f"Anomaly mask columns ({anomaly_mask.columns}) "
            f"do not match autoencoder output columns ({autoencoder_output_df.columns})"
        )

    try:
        # Create corrected dataset where anomalies are replaced with AE output
        corrected_data_df = actual_data_df.copy()
        corrected_data_df[feature_cols] = actual_data_df[feature_cols].where(
            ~anomaly_mask[feature_cols], autoencoder_output_df[feature_cols]
        )

        replaced_counts = anomaly_mask[feature_cols].sum()
        for col, count in replaced_counts.items():
            logger.info(f"{count} anomalies corrected in column {col}")

        # Save if a path is provided
        if save_path:
            save_csv(
                data=corrected_data_df,
                save_path=save_path,
                filename=filename,
            )

        return corrected_data_df

    except Exception as e:
        logger.error(f"Error correcting data: {str(e)}")
        raise
