from typing import Tuple

import numpy as np
import pandas as pd
from mango_time_series.exploratory_analysis.heteroscedasticity import (
    detect_and_transform_heteroscedasticity,
)
from mango_time_series.logging import get_configured_logger
from statsmodels.tsa.seasonal import STL, MSTL

logger = get_configured_logger()


class SeasonalityDecompose:
    """
    Class for time series decomposition and seasonality analysis.

    Provides methods for decomposing time series into trend, seasonal, and residual
    components using STL (Seasonal-Trend decomposition using LOESS) and MSTL
    (Multiple Seasonal-Trend decomposition using LOESS) methods. Also includes
    functionality for detecting heteroscedasticity and measuring seasonal strength.
    """

    def __init__(self, fs_threshold: float = 0.64):
        """
        Initialize the SeasonalityDecompose with a threshold for seasonal strength.

        :param fs_threshold: Threshold value to determine significant seasonality (default: 0.64)
        :type fs_threshold: float

        Note:
            The seasonal strength (Fs) is calculated as: Fs = max(0, 1 - Var(Rt) / Var(St + Rt))
            where Rt is the residual component and St is the seasonal component.
        """
        self.fs_threshold = fs_threshold

    @staticmethod
    def decompose_stl(series: pd.Series, period: int):
        """
        Decompose time series using STL (Seasonal-Trend decomposition using LOESS).

        Performs seasonal-trend decomposition using LOESS smoothing. Automatically
        detects heteroscedasticity and applies appropriate transformation (Box-Cox)
        if needed. Uses multiplicative decomposition for heteroscedastic series
        and additive decomposition otherwise.

        :param series: Time series data to decompose
        :type series: pandas.Series
        :param period: Seasonal period (e.g., 12 for monthly data with yearly seasonality)
        :type period: int
        :return: Tuple containing (trend, seasonal, residual) components
        :rtype: tuple[pandas.Series, pandas.Series, pandas.Series]
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
        Decompose time series using MSTL (Multiple Seasonal-Trend decomposition using LOESS).

        Performs decomposition with multiple seasonal components simultaneously.
        Automatically handles Box-Cox transformation for series with positive values
        and uses standard decomposition for series with non-positive values.

        :param series: Time series data to decompose
        :type series: pandas.Series
        :param periods: List of seasonal periods to decompose (e.g., [12, 24] for monthly and bi-monthly seasonality)
        :type periods: list[int]
        :return: Tuple containing (trend, seasonal, residual) components
        :rtype: tuple[pandas.Series, pandas.Series, pandas.Series]
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
        Calculate the seasonal strength (Fs) based on decomposition components.

        Measures the strength of seasonality in the time series using the formula:
        Fs = max(0, 1 - Var(Rt) / Var(St + Rt))

        where:
        - Rt is the residual component
        - St is the seasonal component
        - Var() represents variance

        Values closer to 1 indicate stronger seasonality, while values closer to 0
        indicate weaker or no seasonality.

        :param seasonal: Seasonal component from time series decomposition
        :type seasonal: numpy.ndarray
        :param resid: Residual component from time series decomposition
        :type resid: numpy.ndarray
        :return: Seasonal strength value between 0 and 1
        :rtype: float
        """
        var_resid = np.var(resid)
        var_seasonal_resid = np.var(seasonal + resid)

        fs = max(0, 1 - (var_resid / var_seasonal_resid))
        return fs

    def detect_seasonality(self, series: pd.Series, period: int) -> bool:
        """
        Detect if the time series has significant seasonality.

        Performs STL decomposition and calculates seasonal strength to determine
        if the series exhibits significant seasonality based on the configured
        threshold (fs_threshold).

        :param series: Time series data to analyze for seasonality
        :type series: pandas.Series
        :param period: Seasonal period to test (e.g., 12 for monthly data with yearly seasonality)
        :type period: int
        :return: True if seasonal strength exceeds the threshold, False otherwise
        :rtype: bool

        Example:
            >>> decomposer = SeasonalityDecompose(fs_threshold=0.5)
            >>> has_seasonality = decomposer.detect_seasonality(monthly_data, period=12)
        """
        trend, seasonal, resid = self.decompose_stl(series=series, period=period)
        fs = self.calculate_seasonal_strength(seasonal=seasonal, resid=resid)
        return fs > self.fs_threshold
