from typing import List, Optional

import mango_autoencoder.utils.processing as processing
import numpy as np
import pandas as pd
from mango_autoencoder.logging import get_configured_logger
from mango_autoencoder.utils.plots import create_error_analysis_dashboard
from scipy.stats import f_oneway

logger = get_configured_logger()


def calculate_reconstruction_error(
    x_converted: np.ndarray, x_hat: np.ndarray
) -> np.ndarray:
    """
    Calculate the reconstruction error matrix between original and reconstructed data.

    Computes the absolute difference between original data and autoencoder
    reconstructed data. This is a fundamental metric for evaluating
    autoencoder performance and identifying reconstruction quality.

    :param x_converted: Original input data array
    :type x_converted: np.ndarray
    :param x_hat: Reconstructed data array from autoencoder
    :type x_hat: np.ndarray
    :return: Matrix of absolute differences between original and reconstructed data
    :rtype: np.ndarray
    :raises ValueError: If input arrays have different shapes

    Example:
        >>> original = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> reconstructed = np.array([[1.1, 1.9], [2.8, 4.2]])
        >>> error = calculate_reconstruction_error(original, reconstructed)
        >>> print(error)
        [[0.1 0.1]
         [0.2 0.2]]

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

    Creates comprehensive error analysis dashboards showing error distributions,
    statistics, and visualizations for each feature column. Generates interactive
    Plotly charts for detailed error analysis.

    :param error_matrix: Matrix of reconstruction errors (samples x features)
    :type error_matrix: np.ndarray
    :param column_names: List of feature names for columns
    :type column_names: Optional[List[str]]
    :param save_path: Directory path to save generated plots
    :type save_path: Optional[str]
    :param show: Whether to display plots in browser
    :type show: bool
    :return: DataFrame with error data and column names
    :rtype: pd.DataFrame
    :raises ValueError: If error_matrix is empty or if column_names length doesn't match error_matrix columns

    Example:
        >>> error_matrix = np.array([[0.1, 0.2], [0.3, 0.1], [0.2, 0.4]])
        >>> column_names = ["feature_1", "feature_2"]
        >>> error_df = analyze_error_by_columns(
        ...     error_matrix,
        ...     column_names,
        ...     save_path="./plots"
        ... )

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
    split_column: Optional[str] = "data_split",
    save_path: Optional[str] = None,
    filename: str = "reconstruction_error.csv",
) -> pd.DataFrame:
    """
    Calculate and optionally save reconstruction error between actual data and autoencoder output.

    Computes the difference between original data and autoencoder reconstructed data,
    with optional data split handling and high-error feature detection. Provides
    warnings for features with unusually high reconstruction errors.

    :param actual_data_df: Original input data DataFrame
    :type actual_data_df: pd.DataFrame
    :param autoencoder_output_df: Reconstructed data DataFrame from autoencoder
    :type autoencoder_output_df: pd.DataFrame
    :param threshold_factor: Multiplier for median reconstruction error to flag high-error features
    :type threshold_factor: int
    :param split_column: Optional name of column that defines data split (train/val/test)
    :type split_column: Optional[str]
    :param save_path: Optional directory path to save the output CSV file
    :type save_path: Optional[str]
    :param filename: Filename for the saved CSV file
    :type filename: str
    :return: DataFrame with reconstruction error values
    :rtype: pd.DataFrame
    :raises ValueError: If input DataFrames have different index or columns

    Example:
        >>> import pandas as pd
        >>> actual = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> reconstructed = pd.DataFrame({"feature1": [1.1, 1.9, 3.1], "feature2": [3.9, 5.2, 5.8]})
        >>> error_df = reconstruction_error(actual, reconstructed, save_path="./results")

    """
    # Validate input parameters
    if not actual_data_df.index.equals(autoencoder_output_df.index):
        raise ValueError(
            f"Autoencoder output index ({autoencoder_output_df.index}) "
            f"does not match actual data index ({actual_data_df.index})"
        )
    if not autoencoder_output_df.columns.equals(actual_data_df.columns):
        raise ValueError(
            f"Autoencoder output columns ({autoencoder_output_df.columns}) "
            f"do not match actual data columns ({actual_data_df.columns})"
        )
    if split_column is not None and split_column not in actual_data_df.columns:
        raise ValueError(f"{split_column} not found in actual_data_df")

    try:
        # If split_column is None, all columns are used for error calculation
        feature_columns = [
            col for col in autoencoder_output_df.columns if col != split_column
        ]
        # Generate reconstruction error DataFrame
        reconstruction_error_df = (
            autoencoder_output_df[feature_columns] - actual_data_df[feature_columns]
        )

        # Add split_column if it was originally present
        if split_column in actual_data_df.columns:
            reconstruction_error_df.insert(
                0, split_column, actual_data_df[split_column]
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
            processing.save_csv(
                data=reconstruction_error_df,
                save_path=save_path,
                filename=filename,
            )

        return reconstruction_error_df

    except Exception as e:
        logger.error(f"Error calculating reconstruction error: {str(e)}")
        raise


def anova_reconstruction_error(
    reconstruction_error_df: pd.DataFrame,
    p_value_threshold: Optional[float] = 0.05,
    F_stat_threshold: Optional[float] = None,
    split_column: str = "data_split",
) -> pd.DataFrame:
    """
    Perform one-way ANOVA to test if reconstruction errors vary across split_column for each feature.

    Conducts statistical analysis to determine if reconstruction errors differ
    significantly between data splits (train/validation/test) for each feature.
    Uses F-test to assess variance differences and provides warnings for
    significant differences.

    :param reconstruction_error_df: DataFrame with reconstruction error and split_column
    :type reconstruction_error_df: pd.DataFrame
    :param p_value_threshold: Maximum p-value to trigger logger warning (significance level)
    :type p_value_threshold: float
    :param F_stat_threshold: Minimum F-statistic to trigger logger warning
    :type F_stat_threshold: float
    :param split_column: Name of column that defines data split categories
    :type split_column: str
    :return: DataFrame with F-statistics and p-values per feature
    :rtype: pd.DataFrame
    :raises ValueError: If split_column not found or has insufficient categories

    Example:
        >>> error_df = pd.DataFrame({
        ...     "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        ...     "feature2": [0.2, 0.3, 0.4, 0.5, 0.6],
        ...     "data_split": ["train", "train", "val", "val", "test"]
        ... })
        >>> anova_results = anova_reconstruction_error(error_df, p_value_threshold=0.01)

    """
    if split_column not in reconstruction_error_df.columns:
        raise ValueError(
            f"Anova calculation requires reconstruction_error_df to have {split_column}."
        )
    if reconstruction_error_df[split_column].nunique() <= 2:
        raise ValueError(f"{split_column} should have at least 2 categories.")

    try:
        results = []
        feature_columns = [
            col for col in reconstruction_error_df.columns if col != split_column
        ]

        # Loop through each feature column and perform one-way ANOVA across split_column
        groups = reconstruction_error_df.groupby(split_column)
        for feature in feature_columns:
            group_i = groups[feature].apply(list)

            # Perform one-way ANOVA
            f_stat, p_val = f_oneway(*group_i)
            results.append(
                {
                    "feature": feature,
                    "F_statistic": f_stat,
                    "p_value": p_val,
                }
            )

            # Log if statistics exceed threshold
            if f_stat and F_stat_threshold and f_stat > F_stat_threshold:
                logger.warning(
                    f"{feature}: F_statistic ({f_stat:.4f}) exceeds threshold ({F_stat_threshold})"
                )
            if p_val and p_value_threshold and p_val < p_value_threshold:
                logger.warning(
                    f"{feature}: p_value ({p_val:.4f}) is below threshold ({p_value_threshold})"
                )

        return pd.DataFrame(results)

    except Exception as e:
        logger.error(
            f"Error computing one-way ANOVA tests across feature split_column: {str(e)}"
        )
        raise


def reconstruction_error_summary(
    reconstruction_error_df: pd.DataFrame,
    split_column: Optional[str] = "data_split",
    split_order: Optional[List[str]] = ["train", "validation", "test"],
    save_path: Optional[str] = None,
    filename: str = "reconstruction_error_summary.csv",
) -> pd.DataFrame:
    """
    Generate and optionally save summary statistics for reconstruction error
    grouped by split_column (if provided and present in the DataFrame).

    Creates comprehensive summary statistics including absolute mean and standard
    deviation for reconstruction errors, optionally grouped by data splits.
    Provides organized output with consistent column ordering.

    :param reconstruction_error_df: DataFrame with reconstruction error data
    :type reconstruction_error_df: pd.DataFrame
    :param split_column: Optional name of column that defines data split categories
    :type split_column: Optional[str]
    :param split_order: Optional order of split categories for consistent output
    :type split_order: Optional[List[str]]
    :param save_path: Optional directory path to save the summary CSV file
    :type save_path: Optional[str]
    :param filename: Filename to use for the saved CSV file
    :type filename: str
    :return: Summary statistics DataFrame with mean and std error metrics
    :rtype: pd.DataFrame
    :raises ValueError: If data is empty or split_order doesn't match actual splits

    Example:
        >>> error_df = pd.DataFrame({
        ...     "feature1": [0.1, 0.2, 0.3, 0.4],
        ...     "feature2": [0.2, 0.3, 0.4, 0.5],
        ...     "data_split": ["train", "train", "val", "val"]
        ... })
        >>> summary = reconstruction_error_summary(error_df, save_path="./results")

    """
    if reconstruction_error_df.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    if split_column is not None and split_column not in reconstruction_error_df.columns:
        raise ValueError(
            f"{split_column} is not None and is not in "
            f"reconstruction_error_df columns ({reconstruction_error_df.columns})"
        )

    try:
        # Define list of statistics and their corresponding names
        stat_list = [processing.abs_mean, "std"]
        stat_list_names = [s if isinstance(s, str) else s.__name__ for s in stat_list]

        # Calculate summary statistics for split column if present
        if split_column in reconstruction_error_df.columns:
            unique_splits = reconstruction_error_df[split_column].unique()
            if set(split_order) != set(unique_splits):
                raise ValueError(
                    f"split_order set ({split_order}) must be equal to "
                    f"reconstruction_error_df[{split_column}] set ({unique_splits})"
                )
            summary_stats = reconstruction_error_df.groupby(split_column).agg(stat_list)
            summary_stats = summary_stats.T.unstack(level=1)
            summary_stats.index.name = "feature"
            summary_stats.columns = [
                f"{split}_{stat}_error" for split, stat in summary_stats.columns
            ]

            # Reorder columns: per split column show statistics
            column_order = [
                f"{split}_{stat}_error"
                for stat in stat_list_names
                for split in split_order
            ]
            summary_stats = summary_stats[column_order]

        # Calculate cumulative summary statistics if split column is not present
        else:
            summary_stats = reconstruction_error_df.agg(stat_list)
            summary_stats = summary_stats.T
            summary_stats.index.name = "feature"
            summary_stats.columns = [
                f"{split_stat}_error" for split_stat in summary_stats.columns
            ]

        # Save if a path is provided
        if save_path:
            processing.save_csv(
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

    Detects anomalous data points by comparing reconstruction errors to statistical
    thresholds based on standard deviations from the mean. Creates boolean masks
    indicating anomaly locations and provides summary statistics of anomaly rates.

    :param reconstruction_error_df: DataFrame with autoencoder reconstruction errors and 'data_split' column
    :type reconstruction_error_df: pd.DataFrame
    :param std_threshold: Threshold in terms of standard deviations from the mean error
    :type std_threshold: float
    :param save_path: Optional directory path to save output files
    :type save_path: Optional[str]
    :param anomaly_mask_filename: CSV filename for the boolean anomaly mask
    :type anomaly_mask_filename: str
    :param anomaly_proportions_filename: CSV filename for anomaly rate summary
    :type anomaly_proportions_filename: str
    :return: Boolean DataFrame mask (True = anomaly, False = normal)
    :rtype: pd.DataFrame
    :raises ValueError: If required columns are missing, if data is empty, or if std_threshold is negative

    Example:
        >>> error_df = pd.DataFrame({
        ...     "feature1": [0.1, 0.2, 0.3, 0.4, 0.5],
        ...     "feature2": [0.2, 0.3, 0.4, 0.5, 0.6],
        ...     "data_split": ["train", "train", "val", "val", "test"]
        ... })
        >>> anomaly_mask = std_error_threshold(error_df, std_threshold=2.0, save_path="./results")

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
            processing.save_csv(
                data=anomaly_mask,
                save_path=save_path,
                filename=anomaly_mask_filename,
            )
            processing.save_csv(
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

    Creates a corrected dataset by replacing identified anomalous values with
    their autoencoder reconstructions. Uses a boolean mask to determine which
    values should be replaced and provides logging of correction counts.

    :param actual_data_df: Original input data DataFrame
    :type actual_data_df: pd.DataFrame
    :param autoencoder_output_df: Reconstructed data DataFrame from autoencoder
    :type autoencoder_output_df: pd.DataFrame
    :param anomaly_mask: Boolean DataFrame indicating where to apply corrections (True = replace)
    :type anomaly_mask: pd.DataFrame
    :param save_path: Optional directory path to save the corrected data as CSV
    :type save_path: Optional[str]
    :param filename: Filename to use if saving the corrected data
    :type filename: str
    :return: DataFrame with corrected data (anomalies replaced with reconstructions)
    :rtype: pd.DataFrame
    :raises ValueError: If input DataFrames have different lengths or columns

    Example:
        >>> import pandas as pd
        >>> actual = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        >>> reconstructed = pd.DataFrame({"feature1": [1.1, 1.9, 3.1], "feature2": [3.9, 5.2, 5.8]})
        >>> mask = pd.DataFrame({"feature1": [False, True, False], "feature2": [True, False, False]})
        >>> corrected = corrected_data(actual, reconstructed, mask, save_path="./results")

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
            processing.save_csv(
                data=corrected_data_df,
                save_path=save_path,
                filename=filename,
            )

        return corrected_data_df

    except Exception as e:
        logger.error(f"Error correcting data: {str(e)}")
        raise
