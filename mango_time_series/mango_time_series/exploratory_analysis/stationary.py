import warnings
from typing import Tuple

import pandas as pd
import polars as pl
from mango_time_series.exploratory_analysis.decomposition import SeasonalityDecompose
from mango_time_series.exploratory_analysis.seasonal import SeasonalityDetector
from mango_time_series.logging import get_configured_logger
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

logger = get_configured_logger()


class StationaryTester:
    """
    Tester for making time series stationary through differencing.

    Implements a comprehensive approach to stationarity testing and transformation
    using ADF and KPSS tests combined with seasonal strength analysis. Applies
    regular and seasonal differencing iteratively until the series becomes
    stationary in both trend and seasonal components.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        fs_threshold: float = 0.64,
    ):
        """
        Initialize the StationaryTester with analysis thresholds.

        Sets up the tester with configurable thresholds for stationarity testing
        and seasonal strength evaluation. Initializes seasonality detection
        and decomposition components.

        :param threshold: Significance level for ADF test (default: 0.05)
        :type threshold: float
        :param fs_threshold: Seasonal strength threshold for differencing (default: 0.64)
        :type fs_threshold: float
        """
        self.threshold = threshold
        self.fs_threshold = fs_threshold
        self.seasonality_detector = SeasonalityDetector()
        self.stl_detector = SeasonalityDecompose(fs_threshold=self.fs_threshold)

    @staticmethod
    def _convert_to_pandas(df: pl.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Convert Polars DataFrame to Pandas DataFrame with date index.

        Converts the input Polars DataFrame to a Pandas DataFrame and sets
        the specified date column as the index. Automatically converts the
        date column to datetime format.

        :param df: Input Polars DataFrame to convert
        :type df: polars.DataFrame
        :param date_column: Name of the date column to use as index
        :type date_column: str
        :return: Pandas DataFrame with date column as index
        :rtype: pandas.DataFrame
        :raises ValueError: If the specified date column is not found
        """
        df = df.to_pandas()
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in DataFrame.")
        df[date_column] = pd.to_datetime(df[date_column])
        return df.set_index(date_column)

    @staticmethod
    def test_adf(series: pd.Series) -> float:
        """
        Test stationarity using the Augmented Dickey-Fuller test.

        Performs the ADF test to determine if the time series is stationary.
        The null hypothesis is that the series has a unit root (non-stationary).

        :param series: Time series data to test
        :type series: pandas.Series
        :return: P-value of the ADF test
        :rtype: float

        Note:
            - p-value < 0.05 typically indicates stationarity
            - Lower p-values suggest stronger evidence against unit root
        """
        adf_result = adfuller(x=series)
        p_value = adf_result[1]
        return p_value

    @staticmethod
    def test_kpss(series: pd.Series) -> float | None:
        """
        Test stationarity using the KPSS test.

        Performs the KPSS test to determine if the time series is stationary.
        The null hypothesis is that the series is stationary around a constant.

        :param series: Time series data to test
        :type series: pandas.Series
        :return: P-value of the KPSS test, or None if test fails
        :rtype: float or None

        Note:
            - p-value < 0.05 typically indicates non-stationarity
            - Handles InterpolationWarning and returns None on errors
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
        Run both ADF and KPSS stationarity tests.

        Executes both the Augmented Dickey-Fuller and KPSS tests to comprehensively
        assess the stationarity of the time series. The combination of these tests
        provides a more robust assessment than either test alone.

        :param series: Time series data to test
        :type series: pandas.Series
        :return: Tuple containing (ADF_p_value, KPSS_p_value)
        :rtype: tuple[float, float]

        Note:
            - ADF test: null hypothesis is unit root (non-stationary)
            - KPSS test: null hypothesis is stationarity
            - Combined results help determine appropriate transformation
        """
        p_value_adf = self.test_adf(series=series)
        p_value_kpss = self.test_kpss(series=series)
        return p_value_adf, p_value_kpss

    def _is_already_stationary(
        self, p_value_adf: float, p_value_kpss: float, seasonal_strength: float
    ) -> bool:
        """
        Check if the series is already stationary.

        Evaluates whether the time series is already stationary based on
        ADF and KPSS test results combined with seasonal strength analysis.
        Considers both trend and seasonal stationarity.

        :param p_value_adf: P-value from ADF test
        :type p_value_adf: float
        :param p_value_kpss: P-value from KPSS test
        :type p_value_kpss: float
        :param seasonal_strength: Seasonal strength measure (Fs)
        :type seasonal_strength: float
        :return: True if series is stationary, False otherwise
        :rtype: bool

        Note:
            - Stationary if: ADF p-value < threshold AND KPSS p-value >= threshold AND seasonal_strength < fs_threshold
            - Logs the decision with test statistics
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
        Detect seasonality and calculate seasonal strength.

        Uses the seasonality detector to identify seasonal periods and then
        applies STL decomposition to calculate the seasonal strength measure.
        Adjusts the period for STL decomposition to ensure it's odd.

        :param series: Time series data to analyze
        :type series: pandas.Series
        :return: Tuple containing (detected_period, stl_period, seasonal_strength)
        :rtype: tuple[int, int, float]

        Note:
            - Returns period=1, stl_period=1, seasonal_strength=0 if no seasonality detected
            - STL period is adjusted to be odd (period+1 if even)
            - Seasonal strength calculated from STL decomposition components
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
        Apply regular differencing iteratively until series becomes stationary.

        Implements a comprehensive approach to regular differencing based on
        ADF and KPSS test results. Handles three main cases:

        1. Both tests indicate non-stationarity: Apply regular differencing
        2. KPSS indicates stationarity, ADF indicates non-stationarity:
           Series is trend-stationary, apply differencing
        3. KPSS indicates non-stationarity, ADF indicates stationarity:
           Series is difference-stationary, apply detrending

        :param series: Time series data to transform
        :type series: pandas.Series
        :param p_value_adf: P-value from ADF test
        :type p_value_adf: float
        :param p_value_kpss: P-value from KPSS test
        :type p_value_kpss: float
        :param period_stl: Period for STL decomposition
        :type period_stl: int
        :return: Tuple containing (differenced_series, number_of_differencing_steps)
        :rtype: tuple[pandas.Series, int]

        Note:
            - Iteratively applies transformations until stationarity is achieved
            - Recalculates test statistics after each transformation
            - Logs the transformation decisions and progress
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
        Apply seasonal differencing iteratively until seasonal strength is reduced.

        Applies seasonal differencing at the detected seasonal period until
        the seasonal strength measure falls below the configured threshold.
        Recalculates seasonal strength after each differencing step.

        :param series: Time series data to transform
        :type series: pandas.Series
        :param seasonal_strength: Current seasonal strength measure
        :type seasonal_strength: float
        :param period: Detected seasonal period for differencing
        :type period: int
        :param period_stl: Period for STL decomposition
        :type period_stl: int
        :return: Tuple containing (differenced_series, number_of_seasonal_differencing_steps)
        :rtype: tuple[pandas.Series, int]

        Note:
            - Continues until seasonal_strength < fs_threshold
            - Uses STL decomposition to recalculate seasonal strength
            - Logs seasonal strength values after each step
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
        Prepare final DataFrame with transformed series and adjusted dates.

        Creates a new DataFrame containing the transformed time series data
        with appropriately adjusted date indices to match the length of the
        differenced series.

        :param df: Original DataFrame with date index
        :type df: pandas.DataFrame
        :param differentiated_series: Transformed time series data
        :type differentiated_series: pandas.Series
        :param date_column: Name of the date column
        :type date_column: str
        :param target_column: Name of the target column
        :type target_column: str
        :return: DataFrame with transformed data and adjusted dates
        :rtype: pandas.DataFrame

        Note:
            - Adjusts date index to match the length of differenced series
            - Takes the last N dates where N is the length of differentiated series
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
        Transform time series to make it stationary in trend and seasonal components.

        Main function that implements a comprehensive approach to making time series
        stationary through iterative application of regular and seasonal differencing.
        Uses ADF and KPSS tests combined with seasonal strength analysis to determine
        the appropriate transformations.

        :param df: Input Polars DataFrame containing time series data
        :type df: polars.DataFrame
        :param target_column: Name of the column containing time series values
        :type target_column: str
        :param date_column: Name of the column containing dates
        :type date_column: str
        :return: Tuple containing (transformed_DataFrame, regular_differencing_steps, seasonal_differencing_steps)
        :rtype: tuple[pandas.DataFrame, int, int]

        Note:
            - Returns original data if already stationary
            - Applies regular differencing first, then seasonal differencing
            - Logs final differencing parameters for reference
            - Converts Polars DataFrame to Pandas for processing
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
