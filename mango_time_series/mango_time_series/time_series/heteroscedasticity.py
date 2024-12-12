from typing import Tuple

import numpy as np
import statsmodels.api as sm
from mango.logging.logger import get_basic_logger
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.stats.diagnostic import het_breuschpagan

logger = get_basic_logger()

try:
    import pandas as pd
    import polars as pl
except ImportError:
    pd = None
    pl = None


def get_optimal_lambda(series: np.ndarray) -> float:
    """
    Calculate the optimal Box-Cox lambda using the boxcox_normmax function.
    Finds the lambda that maximizes the normality of the transformed data.

    :param series: A numpy array representing the time series data.
    :return: The optimal lambda value for the Box-Cox transformation.
    """
    min_value = min(series)
    if min_value < 0:
        series = series - min_value + 1

    optimal_lambda = boxcox_normmax(x=series)
    return optimal_lambda


def apply_boxcox_with_lambda(series: np.ndarray, lambda_value: float) -> np.ndarray:
    """
    Apply Box-Cox transformation using a specified lambda value.

    :param series: A numpy array representing the time series data to be transformed.
    :param lambda_value: The lambda value to use for the Box-Cox transformation.
    :return: A numpy array of the transformed time series.
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
    Detect heteroscedasticity in a time series using the Breusch-Pagan test and apply Box-Cox transformation if detected.

    :param series: A numpy array representing the time series data.
    :return: A tuple with the transformed series (or original if no transformation is applied) and the lambda value (or None).
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
