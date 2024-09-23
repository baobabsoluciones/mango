from typing import Tuple

import numpy as np
import statsmodels.api as sm
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.stats.diagnostic import het_breuschpagan

from mango_base.mango.logging.logger import get_basic_logger

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

    optimal_lambda = boxcox_normmax(series)
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

    transformed_series = boxcox(series, lmbda=lambda_value)
    return transformed_series


def detect_and_transform_heteroscedasticity(
    df: pl.DataFrame, target_column: str
) -> Tuple[np.ndarray, float or None]:
    """
    Detect heteroscedasticity in a time series using the Breusch-Pagan test and apply Box-Cox transformation if detected.

    :param df: A Polars dataframe containing the time series data.
    :param target_column: The name of the target column containing the time series values.
    :return: A tuple with the transformed series (or original if no transformation is applied) and the lambda value (or None).
    """
    df_pandas = df.to_pandas()

    if target_column not in df_pandas.columns:
        raise ValueError(f"Column '{target_column}' not found in DataFrame")

    series = df_pandas[target_column].values

    if len(series) <= 1:
        raise ValueError(
            "The time series must contain more than one data point for the test"
        )

    # Prepare for Breusch-Pagan test
    trend = np.arange(len(series))
    X = sm.add_constant(trend)
    model = sm.OLS(series, X).fit()
    residuals = model.resid
    bp_test = het_breuschpagan(residuals, X)
    if isinstance(bp_test, tuple):
        p_val = bp_test[1]  # p-value from Breusch-Pagan test
    else:
        p_val = bp_test

    if p_val < 0.05:
        logger.info(
            f"Heteroscedasticity detected via Breusch-Pagan test (p-value = {p_val:.4f})."
        )
        # Apply Box-Cox transformation if heteroscedasticity is detected
        optimal_lambda = boxcox_normmax(series)
        transformed_series = apply_boxcox_with_lambda(series, optimal_lambda)
        return transformed_series, optimal_lambda
    else:
        logger.info(
            f"No significant heteroscedasticity detected via Breusch-Pagan test (p-value = {p_val:.4f})."
        )
        return series, None
