from typing import Tuple

import numpy as np
import statsmodels.api as sm
from mango_time_series.logging import get_configured_logger
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.stats.diagnostic import het_breuschpagan

logger = get_configured_logger()

try:
    import pandas as pd
    import polars as pl
except ImportError:
    pd = None
    pl = None


def get_optimal_lambda(series: np.ndarray) -> float:
    """
    Calculate the optimal Box-Cox lambda parameter for transformation.

    Uses the boxcox_normmax function to find the lambda value that maximizes
    the normality of the transformed data. Automatically handles negative values
    by shifting the series to ensure all values are positive before transformation.

    :param series: Time series data to find optimal lambda for
    :type series: numpy.ndarray
    :return: Optimal lambda value for Box-Cox transformation
    :rtype: float

    Note:
        If the series contains negative values, it is automatically shifted
        to ensure all values are positive before calculating lambda.
    """
    min_value = min(series)
    if min_value < 0:
        series = series - min_value + 1

    optimal_lambda = boxcox_normmax(x=series)
    return optimal_lambda


def apply_boxcox_with_lambda(series: np.ndarray, lambda_value: float) -> np.ndarray:
    """
    Apply Box-Cox transformation using a specified lambda value.

    Transforms the time series data using the Box-Cox power transformation
    with the provided lambda parameter. Automatically handles negative values
    by shifting the series to ensure all values are positive before transformation.

    :param series: Time series data to transform
    :type series: numpy.ndarray
    :param lambda_value: Lambda parameter for Box-Cox transformation
    :type lambda_value: float
    :return: Transformed time series data
    :rtype: numpy.ndarray

    Note:
        If the series contains negative values, it is automatically shifted
        to ensure all values are positive before applying the transformation.
    """
    min_value = min(series)
    if min_value < 0:
        series = series - min_value + 1

    transformed_series = boxcox(x=series, lmbda=lambda_value)
    return transformed_series


def detect_and_transform_heteroscedasticity(
    series: np.ndarray,
) -> Tuple[np.ndarray, float or None]:
    """
    Detect heteroscedasticity and apply Box-Cox transformation if needed.

    Performs the Breusch-Pagan test to detect heteroscedasticity (non-constant variance)
    in the time series. If heteroscedasticity is detected (p-value < 0.05), applies
    Box-Cox transformation to stabilize the variance. Returns the original series
    if no transformation is needed or if the series contains non-positive values.

    :param series: Time series data to analyze and potentially transform
    :type series: numpy.ndarray
    :return: Tuple containing (transformed_series, lambda_value)
        - transformed_series: Original or transformed time series
        - lambda_value: Lambda used for transformation, or None if no transformation applied
    :rtype: tuple[numpy.ndarray, float or None]

    Raises:
        ValueError: If the time series contains only one data point

    Note:
        - Series with zeros or negative values are not transformed
        - Uses Breusch-Pagan test with significance level of 0.05
        - Logs the test results and transformation decisions
    """
    if len(series) <= 1:
        raise ValueError(
            "The time series must contain more than one data point for the test"
        )

    if np.any(series <= 0):
        logger.warning(
            "Series contains zeros or negative values. Skipping Box-Cox transformation."
        )
        return series, None

    # Breusch-Pagan test
    trend = np.arange(len(series))
    X = sm.add_constant(trend)
    model = sm.OLS(series, X).fit()
    residuals = model.resid
    bp_test = het_breuschpagan(resid=residuals, exog_het=X)
    if isinstance(bp_test, tuple):
        p_val = bp_test[1]
    else:
        p_val = bp_test

    if p_val < 0.05:
        logger.info(
            f"Heteroscedasticity detected via Breusch-Pagan test (p-value = {p_val:.4f})."
        )
        optimal_lambda = get_optimal_lambda(series=series)
        transformed_series = apply_boxcox_with_lambda(
            series=series, lambda_value=optimal_lambda
        )
        return transformed_series, optimal_lambda
    else:
        logger.info(
            f"No significant heteroscedasticity detected via Breusch-Pagan test (p-value = {p_val:.4f})."
        )
        return series, None
