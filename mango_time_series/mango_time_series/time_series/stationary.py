import warnings
from typing import Tuple

import pandas as pd
import polars as pl
from mango.logging.logger import get_basic_logger
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from mango_time_series.time_series.decomposition import SeasonalityDecompose
from mango_time_series.time_series.seasonal import SeasonalityDetector

logger = get_basic_logger()


class StationaryTester:

    def __init__(
        self,
        threshold: float = 0.05,
        fs_threshold: float = 0.64,
    ):
        """
        Initialize the StationaryTester with a significance level for the ADF test and a threshold for seasonal strength.

        :param threshold: Significance level for the ADF test (default 0.05).
        :param fs_threshold: Threshold for the seasonal strength to decide if seasonal differencing is needed (default 0.64).
        """
        self.threshold = threshold
        self.fs_threshold = fs_threshold
        self.seasonality_detector = SeasonalityDetector()
        self.stl_detector = SeasonalityDecompose(fs_threshold=self.fs_threshold)

    @staticmethod
    def _convert_to_pandas(df: pl.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Convert a Polars DataFrame to a Pandas DataFrame and set the index.

        :param df: The input Polars DataFrame.
        :param date_column: The name of the date column to set as the index.
        :return: A Pandas DataFrame with the date column set as the index.
        :raises ValueError: If the date column is not found in the DataFrame.
        """
        df = df.to_pandas()
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
        df[date_column] = pd.to_datetime(df[date_column])
        return df.set_index(date_column)

    @staticmethod
    def test_adf(series: pd.Series) -> float:
        """
        Test the stationarity of the time series data using the Augmented Dickey-Fuller (ADF) test.

        :param series: The time series data as a pandas Series.
        :return: The p-value of the ADF test.
        """
        adf_result = adfuller(x=series)
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
                kpss_result = kpss(x=series, regression="c", nlags="auto")

                if len(w) > 0 and issubclass(w[-1].category, InterpolationWarning):
                    logger.warning(
                        "InterpolationWarning: KPSS p-value outside the expected range."
                    )

            p_value = kpss_result[1]
            return p_value

        except ValueError as e:
            logger.error(f"Error during KPSS test: {e}")
            return None

    def _run_stationarity_tests(self, series: pd.Series) -> Tuple[float, float]:
        """
        Run the ADF and KPSS tests to check for stationarity.

        :param series: The time series data as a pandas Series.
        :return: A tuple containing the p-values of the ADF and KPSS tests.
        """
        p_value_adf = self.test_adf(series=series)
        p_value_kpss = self.test_kpss(series=series)
        return p_value_adf, p_value_kpss

    def _is_already_stationary(
        self, p_value_adf: float, p_value_kpss: float, seasonal_strength: float
    ) -> bool:
        """
        Check if the series is already stationary in both regular and seasonal terms.

        :param p_value_adf: The p-value of the ADF test.
        :param p_value_kpss: The p-value of the KPSS test.
        :param seasonal_strength: The seasonal strength of the series.
        :return: True if the series is already stationary, False otherwise.
        """
        if (
            p_value_adf < self.threshold <= p_value_kpss
            and seasonal_strength < self.fs_threshold
        ):
            logger.info(
                f"The series is already stationary (ADF p-value = {p_value_adf:.4f}, KPSS p-value = {p_value_kpss:.4f}, Fs = {seasonal_strength:.4f}). Returning original series."
            )
            return True
        return False

    def _detect_seasonality(self, series: pd.Series) -> Tuple[int, int, float]:
        """
        Detect seasonality using STL decomposition and calculate seasonal strength.

        :param series: The time series data as a pandas Series.
        :return: A tuple containing the detected period, the period used for STL decomposition, and the seasonal strength.
        """
        detected_seasonalities = self.seasonality_detector.detect_seasonality(
            ts=series.values
        )
        period_stl = 1
        seasonal_strength = 0
        period = 1
        if detected_seasonalities:
            period = int(detected_seasonalities[0])
            period_stl = period if period % 2 != 0 else period + 1
            trend, seasonal, resid = self.stl_detector.decompose_stl(
                series=series, period=period_stl
            )
            seasonal_strength = self.stl_detector.calculate_seasonal_strength(
                seasonal=seasonal, resid=resid
            )
        return period, period_stl, seasonal_strength

    def _apply_regular_differencing(
        self,
        series: pd.Series,
        p_value_adf: float,
        p_value_kpss: float,
        period_stl: int,
    ) -> Tuple[pd.Series, int]:
        """
        Apply regular differencing iteratively until the series becomes stationary.

        This function applies both the ADF (Augmented Dickey-Fuller) and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) tests to assess the stationarity of the time series.
        Based on the outcomes, it applies regular or seasonal differencing to make the series stationary.

        The approach works as follows:

        - ADF Test: If p-value < `threshold_adf`, the null hypothesis of non-stationarity is rejected, meaning the series is stationary.
        - KPSS Test: If p-value < `threshold_kpss`, the null hypothesis of stationarity is rejected, meaning the series is not stationary.
        Possible outcomes of applying these stationary tests are as follows:

        Case 1: Both tests conclude that the series is not stationary - The series is not stationary.
        Case 2: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary.
                Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
        Case 3: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary.
                Differencing is to be used to make series stationary. The differenced series is checked for stationarity.

        :param series: The time series data as a pandas Series.
        :param p_value_adf: The p-value of the ADF test.
        :param p_value_kpss: The p-value of the KPSS test.
        :param period_stl: The period used for STL decomposition.
        :return: A tuple containing the differenced series and the number of regular differencing steps applied.
        """
        d_regular = 0
        while p_value_adf >= self.threshold or p_value_kpss < self.threshold:
            # Case 1: Both ADF and KPSS indicate non-stationarity
            if p_value_adf >= self.threshold > p_value_kpss:
                logger.info(
                    f"Both ADF and KPSS indicate non-stationarity. Applying regular differencing (d={d_regular + 1})."
                )
                series = series.diff().dropna()
                d_regular += 1
            # Case 2: KPSS indicates stationarity and ADF indicates non-stationarity (trend-stationary)
            elif p_value_adf >= self.threshold and p_value_kpss >= self.threshold:
                logger.info(
                    f"The series is trend stationary. Detrending required (d={d_regular + 1})."
                )
                series = series.diff().dropna()
                d_regular += 1
            # Case 3: KPSS indicates non-stationarity and ADF indicates stationarity (difference-stationary)
            elif p_value_adf < self.threshold and p_value_kpss < self.threshold:
                logger.info(f"The series needs detrending.")
                trend, _, _ = self.stl_detector.decompose_stl(
                    series=series, period=period_stl
                )
                series = (series - trend).dropna()

            # Recalculate the stationarity after transformations
            p_value_adf, p_value_kpss = self._run_stationarity_tests(series=series)

        return series, d_regular

    def _apply_seasonal_differencing(
        self, series: pd.Series, seasonal_strength: float, period: int, period_stl: int
    ) -> Tuple[pd.Series, int]:
        """
        Apply seasonal differencing iteratively until the seasonal strength drops below the threshold.

        :param series: The time series data as a pandas Series.
        :param seasonal_strength: The seasonal strength of the series.
        :param period: The detected seasonal period.
        :param period_stl: The period used for STL decomposition.
        :return: A tuple containing the differenced series and the number of seasonal differencing steps applied.
        """
        d_seasonal = 0
        while seasonal_strength > self.fs_threshold:
            logger.info(
                f"Applying seasonal differencing due to high seasonal strength (Fs={seasonal_strength:.4f})."
            )
            series = series.diff(periods=period).dropna()
            d_seasonal += 1
            _, seasonal, resid = self.stl_detector.decompose_stl(
                series=series, period=period_stl
            )
            seasonal_strength = self.stl_detector.calculate_seasonal_strength(
                seasonal=seasonal, resid=resid
            )
            logger.info(
                f"Seasonal strength after differencing: Fs={seasonal_strength:.4f}"
            )
        return series, d_seasonal

    @staticmethod
    def _prepare_final_dataframe(
        df: pd.DataFrame,
        differentiated_series: pd.Series,
        date_column: str,
        target_column: str,
    ) -> pd.DataFrame:
        """
        Prepare the final DataFrame with the transformed time series and adjusted date index.

        :param df: The original DataFrame.
        :param differentiated_series: The differenced time series data.
        :param date_column: The name of the date column.
        :param target_column: The name of the target column.
        :return: A DataFrame with the transformed time series and adjusted date index.
        """
        original_dates = df.index.to_series()
        adjusted_dates = original_dates.iloc[-len(differentiated_series) :].reset_index(
            drop=True
        )
        return pd.DataFrame(
            {date_column: adjusted_dates, target_column: differentiated_series.values}
        )

    def make_stationary(
        self, df: pl.DataFrame, target_column: str, date_column: str
    ) -> Tuple[pd.DataFrame, int, int]:
        """
        Main function to iteratively transform the time series to make it stationary in both regular and seasonal components.

        :param df: The input Polars DataFrame.
        :param target_column: The name of the target column containing the time series data.
        :param date_column: The name of the date column.
        :return: A tuple containing the transformed DataFrame, the number of regular differencing steps, and the number of seasonal differencing steps.
        """
        df = self._convert_to_pandas(df=df, date_column=date_column)
        series = df[target_column]

        # Run stationary tests and detect seasonality
        p_value_adf, p_value_kpss = self._run_stationarity_tests(series=series)
        period, period_stl, seasonal_strength = self._detect_seasonality(series=series)

        # Check if already stationary
        if self._is_already_stationary(
            p_value_adf=p_value_adf,
            p_value_kpss=p_value_kpss,
            seasonal_strength=seasonal_strength,
        ):
            return df.reset_index(), 0, 0

        # Apply differencing if necessary
        differentiated_series, d_regular = self._apply_regular_differencing(
            series=series,
            p_value_adf=p_value_adf,
            p_value_kpss=p_value_kpss,
            period_stl=period_stl,
        )
        differentiated_series, d_seasonal = self._apply_seasonal_differencing(
            series=differentiated_series,
            seasonal_strength=seasonal_strength,
            period_stl=period_stl,
            period=period,
        )

        # Prepare the final dataframe
        df_transformed = self._prepare_final_dataframe(
            df=df,
            differentiated_series=differentiated_series,
            date_column=date_column,
            target_column=target_column,
        )

        logger.info(f"Final regular differencing steps (d): {d_regular}")
        logger.info(f"Final seasonal differencing steps (D): {d_seasonal}")

        return df_transformed, d_regular, d_seasonal
