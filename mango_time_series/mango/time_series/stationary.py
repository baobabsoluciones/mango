import warnings
from typing import Tuple

import pandas as pd
import polars as pl
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from mango_base.mango.logging.logger import get_basic_logger
from mango_time_series.mango.time_series.decomposition import SeasonalityDecompose
from mango_time_series.mango.time_series.seasonal import SeasonalityDetector

logger = get_basic_logger()


class StationaryTester:

    def __init__(
        self,
        threshold: float = 0.05,
        fs_threshold: float = 0.64,
    ):
        """
        Initialize the StationaryTester with a significance level for the ADF test.

        :param threshold: Significance level for the ADF test (default 0.05).
        :param fs_threshold: Threshold for the seasonal strength to decide if seasonal differencing is needed (default 0.64).
        """
        self.threshold = threshold
        self.fs_threshold = fs_threshold
        self.seasonality_detector = SeasonalityDetector()
        self.stl_detector = SeasonalityDecompose(fs_threshold=self.fs_threshold)

    @staticmethod
    def test_adf(series: pd.Series) -> float:
        """
        Test the stationarity of the time series data using the Augmented Dickey-Fuller (ADF) test.

        :param series: The time series data as a pandas Series.
        :return: The p-value of the ADF test.
        """
        adf_result = adfuller(series)
        p_value = adf_result[1]
        return p_value

    @staticmethod
    def test_kpss(series: pd.Series):
        """
        Test the stationarity of the time series data using the KPSS test.

        :param series: The time series data as a pandas Series.
        :return: The p-value of the KPSS test.
        """
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                kpss_result = kpss(series, regression="c", nlags="auto")

                if len(w) > 0 and issubclass(w[-1].category, InterpolationWarning):
                    logger.warning(
                        "InterpolationWarning: KPSS p-value outside the expected range."
                    )

            p_value = kpss_result[1]
            return p_value

        except ValueError as e:
            logger.error(f"Error during KPSS test: {e}")
            return None

    def make_stationary(
        self, df: pl.DataFrame, target_column: str, date_column: str
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Iteratively transform the time series to make it stationary, both in regular and seasonal components.

        This function applies both the ADF (Augmented Dickey-Fuller) and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) tests to assess the stationarity of the time series.
        Based on the outcomes, it applies regular or seasonal differencing to make the series stationary.

        The approach works as follows:

        - ADF Test: If p-value < `threshold_adf`, the null hypothesis of non-stationarity is rejected, meaning the series is stationary.
        - KPSS Test: If p-value < `threshold_kpss`, the null hypothesis of stationarity is rejected, meaning the series is not stationary.

        The cases are handled as follows:
        1. Both tests indicate non-stationarity:
           Regular differencing is applied iteratively until the series becomes stationary (Case 1).
        2. ADF indicates non-stationarity but KPSS indicates stationarity:
           The series is trend-stationary, so detrending is applied (Case 2).
        3. ADF indicates stationarity but KPSS indicates non-stationarity:
           This indicates that the series is difference-stationary and requires differencing (Case 3).
        4. Seasonality Detection:
           If strong seasonality is detected using STL decomposition, seasonal differencing is applied iteratively until the seasonal strength drops below the threshold.

        After applying the transformations, the function returns the modified dataframe along with the number of regular and seasonal differencing steps (d, D).

        :param df : The input DataFrame containing the time series.
        :param target_column : The column name of the target time series to transform.
        :param date_column : The column name containing the dates to be used as the index.

        :return: Tuple[pd.DataFrame, int, int]. Transformed dataframe with the stationary series, number of regular differencing steps (d),
            and number of seasonal differencing steps (D).

        Notes
        -----
        - If both ADF and KPSS tests indicate stationarity after initial checks, no transformation is applied.
        - Seasonal periods detected by the `SeasonalityDetector` will be used for seasonal differencing if needed.
        - The function handles potential warnings from the KPSS test and logs them for user awareness.
        """
        df = df.to_pandas()
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")

        df = df.set_index(date_column)
        df.index = pd.to_datetime(df.index)

        series = df[target_column]

        p_value_adf = self.test_adf(series)
        p_value_kpss = self.test_kpss(series)

        # Initialize counters for regular differencing (d) and seasonal differencing (D)
        d_regular = 0
        d_seasonal = 0
        
        period_stl = 1

        # Detect seasonalities
        detected_seasonalities = self.seasonality_detector.detect_seasonality(
            ts=series.values
        )

        # Decompose the series using STL to calculate seasonal strength
        if detected_seasonalities:
            period = int(detected_seasonalities[0])
            period_stl = period if period % 2 != 0 else period + 1

            trend, seasonal, resid = self.stl_detector.decompose_stl(series, period_stl)
            seasonal_strength = self.stl_detector.calculate_seasonal_strength(
                seasonal, resid
            )
        else:
            period = 1
            seasonal_strength = 0

        # 1. Check if the series is already stationary in both regular and seasonal terms
        if (
            p_value_adf < self.threshold
            and p_value_kpss >= self.threshold
            and seasonal_strength < self.fs_threshold
        ):
            logger.info(
                f"The series is already stationary (ADF p-value = {p_value_adf:.4f}, KPSS p-value = {p_value_kpss:.4f}, Fs = {seasonal_strength:.4f}). Returning original series."
            )
            return df.reset_index(), d_regular, d_seasonal

        # 2. Apply regular differencing if needed
        differentiated_series = series.copy()

        while p_value_adf >= self.threshold or p_value_kpss < self.threshold:

            # Case 1: Both ADF and KPSS indicate non-stationarity
            if p_value_adf >= self.threshold and p_value_kpss < self.threshold:
                logger.info(
                    f"Both ADF and KPSS indicate non-stationarity. Applying regular differencing (d={d_regular + 1})."
                )
                differentiated_series = differentiated_series.diff().dropna()
                d_regular += 1
                p_value_adf = self.test_adf(differentiated_series)
                p_value_kpss = self.test_kpss(differentiated_series)

            # Case 2: KPSS indicates stationarity and ADF indicates non-stationarity (trend-stationary)
            elif (
                p_value_adf >= self.threshold
                and p_value_kpss >= self.threshold
            ):
                logger.info(
                    f"The series is trend stationary. Detrending required (d={d_regular + 1})."
                )
                differentiated_series = differentiated_series.diff().dropna()
                d_regular += 1
                p_value_adf = self.test_adf(differentiated_series)
                p_value_kpss = self.test_kpss(differentiated_series)

            # Case 3: KPSS indicates non-stationarity and ADF indicates stationarity (difference-stationary)
            elif (
                p_value_adf < self.threshold and p_value_kpss < self.threshold
            ):
                logger.info(f"The serie has to be detrended.")
                trend, seasonal, resid = self.stl_detector.decompose_stl(
                    differentiated_series, period_stl
                )

                differentiated_series = differentiated_series - trend
                differentiated_series = differentiated_series.dropna()

                # Recalculate the stationarity after detrending
                p_value_adf = self.test_adf(differentiated_series)
                p_value_kpss = self.test_kpss(differentiated_series)

        # 3. Apply seasonal differencing if Fs > fs_threshold
        while seasonal_strength > self.fs_threshold:
            logger.info(
                f"Applying seasonal differencing due to high seasonal strength (Fs={seasonal_strength:.4f})."
            )
            differentiated_series = differentiated_series.diff(periods=period).dropna()
            d_seasonal += 1

            trend, seasonal, resid = self.stl_detector.decompose_stl(
                differentiated_series, period_stl
            )
            seasonal_strength = self.stl_detector.calculate_seasonal_strength(
                seasonal, resid
            )

            logger.info(
                f"Seasonal strength after differencing: Fs={seasonal_strength:.4f}"
            )

        # 4. Adjust dates and return final dataframe without date as index
        original_dates = df.index.to_series()
        adjusted_dates = original_dates.iloc[-len(differentiated_series) :].reset_index(
            drop=True
        )

        df_transformed = pd.DataFrame(
            {date_column: adjusted_dates, target_column: differentiated_series.values}
        )

        logger.info(f"Final transformation: d={d_regular}, D={d_seasonal}")
        return df_transformed, d_regular, d_seasonal
