from typing import Tuple

import numpy as np
import pandas as pd
from mango.logging.logger import get_basic_logger
from statsmodels.tsa.seasonal import STL, MSTL

from mango_time_series.time_series.heteroscedasticity import (
    detect_and_transform_heteroscedasticity,
)

logger = get_basic_logger()


class SeasonalityDecompose:

    def __init__(self, fs_threshold: float = 0.64):
        """
        Initialize the SeasonalityDecompose with a threshold for seasonal strength.
        :param fs_threshold: The threshold to consider if a series has significant seasonality (default 0.64).
        """
        self.fs_threshold = fs_threshold

    @staticmethod
    def decompose_stl(series: pd.Series, period: int):
        """
        Decompose the time series using STL (Seasonal-Trend decomposition using LOESS).
        :param series: The time series data as a pandas Series.
        :param period: The seasonal period (e.g., 12 for monthly data with yearly seasonality).
        :return: A tuple of (trend, seasonal, residual) components.
        """

        series_array = series.values

        # Detect heteroscedasticity
        transformed_series, lambda_value = detect_and_transform_heteroscedasticity(
            series_array
        )

        transformed_series = pd.Series(transformed_series, index=series.index)
        if lambda_value is not None:
            # Multiplicative STL decomposition
            stl = STL(transformed_series, seasonal=period, robust=True)
            logger.info(
                "Applying multiplicative STL due to detected heteroscedasticity."
            )
        else:
            # Additive STL decomposition
            stl = STL(series, seasonal=period)
            logger.info("Applying additive STL (no heteroscedasticity detected).")

        result = stl.fit()
        return result.trend, result.seasonal, result.resid

    @staticmethod
    def decompose_mstl(
        series: pd.Series, periods: list
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Decompose the time series using MSTL (Multiple Seasonal-Trend decomposition using LOESS).
        :param series:
        :param periods: A tuple of seasonal periods to decompose the series.
        :return: Three Polars Series: trend, seasonal components, and residual.
        """
        if (series <= 0).any():
            mstl = MSTL(series, periods=periods)
            result = mstl.fit()
        else:
            mstl = MSTL(series, periods=periods, lmbda="auto")
            result = mstl.fit()

        return result.trend, result.seasonal, result.resid

    @staticmethod
    def calculate_seasonal_strength(seasonal: np.ndarray, resid: np.ndarray) -> float:
        """
        Calculate the seasonal strength (Fs) based on the decomposition components.
        Formula: Fs = max(0, 1 - Var(Rt) / Var(St + Rt))
        :param seasonal: The seasonal component of the time series.
        :param resid: The residual component of the time series.
        :return: The seasonal strength (Fs).
        """
        var_resid = np.var(resid)
        var_seasonal_resid = np.var(seasonal + resid)

        fs = max(0, 1 - (var_resid / var_seasonal_resid))
        return fs

    def detect_seasonality(self, series: pd.Series, period: int) -> bool:
        """
        Detect if the series has significant seasonality based on the seasonal strength.
        :param series: The time series data as a pandas Series.
        :param period: The seasonal period (e.g., 12 for monthly data with yearly seasonality).
        :return: True if the series has significant seasonality, False otherwise.
        """
        trend, seasonal, resid = self.decompose_stl(series=series, period=period)
        fs = self.calculate_seasonal_strength(seasonal=seasonal, resid=resid)
        return fs > self.fs_threshold
