import numpy as np
import pandas as pd
from typing import Tuple
from statsmodels.tsa.seasonal import STL, MSTL


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
        stl = STL(series, seasonal=period)
        result = stl.fit()
        return result.trend, result.seasonal, result.resid

    @staticmethod
    def calculate_seasonal_strength(seasonal: np.ndarray, resid: np.ndarray) -> float:
        """
        Calculate the seasonal strength (Fs) based on the decomposition components.
        :param seasonal: The seasonal component of the time series.
        :param resid: The residual component of the time series.
        :return: The seasonal strength (Fs).
        """
        var_resid = np.var(resid)
        var_seasonal_resid = np.var(seasonal + resid)

        # Seasonal strength Fs = max(0, 1 - Var(Rt) / Var(St + Rt))
        fs = max(0, 1 - (var_resid / var_seasonal_resid))
        return fs

    def detect_seasonality(self, series: pd.Series, period: int) -> bool:
        """
        Detect if the series has significant seasonality based on the seasonal strength.
        :param series: The time series data as a pandas Series.
        :param period: The seasonal period (e.g., 12 for monthly data with yearly seasonality).
        :return: True if the series has significant seasonality, False otherwise.
        """
        trend, seasonal, resid = self.decompose_stl(series, period)
        fs = self.calculate_seasonal_strength(seasonal, resid)

        # If Fs > threshold, significant seasonality is present
        return fs > self.fs_threshold

    def decompose_mstl(self, series: pd.Series, periods: list) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Decompose the time series using MSTL (Multiple Seasonal-Trend decomposition using LOESS).
        :param series: The time series data as a Polars Series.
        :param periods: A tuple of seasonal periods to decompose the series.
        :return: Three Polars Series: trend, seasonal components, and residual.
        """
        # Apply MSTL decomposition
        mstl = MSTL(series, periods=periods)
        result = mstl.fit()


        return result.trend, result.seasonal, result.resid

