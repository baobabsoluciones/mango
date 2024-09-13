import pandas as pd
from statsmodels.tsa.stattools import adfuller
from mango_time_series.mango.time_series.seasonal import SeasonalityDetector
from mango_base.mango.logging.logger import get_basic_logger

logger = get_basic_logger()


class StationaryTester:

    def __init__(self, threshold: float = 0.05):
        """
        Initialize the StationaryTester with a significance level for the ADF test.

        :param threshold: Significance level for the ADF test (default 0.05).
        """
        self.threshold = threshold
        self.seasonality_detector = SeasonalityDetector()

    @staticmethod
    def test_stationarity(series: pd.Series) -> float:
        """
        Test the stationarity of the time series data using the Augmented Dickey-Fuller (ADF) test.

        :param series: The time series data as a pandas Series.
        :return: The p-value of the ADF test.
        """
        adf_result = adfuller(series)
        p_value = adf_result[1]
        return p_value

    def make_stationary(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Iteratively transform the time series to make it stationary, both in regular and seasonal components.
        Uses regular and seasonal differencing based on detected seasonalities.

        :param df: The input DataFrame containing the time series.
        :param target_column: The column name of the target time series to transform.
        :return: The transformed DataFrame with the series that is stationary.
        """
        series = df[target_column]
        p_value = self.test_stationarity(series)

        # Initialize counters for regular differencing (d) and seasonal differencing (D)
        d_regular = 0
        d_seasonal = {}

        # 1. Check if the serie isalready stationary (ADF test p-value)
        if p_value < self.threshold:
            logger.info(
                f"The series is already stationary (ADF p-value = {p_value:.4f}). Returning original series."
            )
            return df

        # 2. Detect seasonalities using detect_seasonalities from seasonal.py
        detected_seasonalities = self.seasonality_detector.detect_seasonality(series.values)

        # 3. Apply differencing until stationarity is achieved
        differenced_series = series.copy()
        while p_value >= self.threshold:
            # Regular differencing
            differenced_series = differenced_series.diff().dropna()
            d_regular += 1
            p_value = self.test_stationarity(differenced_series)

            if p_value < self.threshold:
                logger.info(
                    f"Series is stationary in regular terms after {d_regular} regular differencing(s) (ADF p-value = {p_value:.4f})."
                )
                break

        # 4. If seasonal periods are detected, apply seasonal differencing
        for seasonality in detected_seasonalities:
            d_seasonal[seasonality] = 0

            p_value = self.test_stationarity(differenced_series)
            while p_value >= self.threshold:
                logger.info(
                    f"Applying seasonal differencing with a period of {seasonality}..."
                )
                differenced_series = differenced_series.diff(
                    periods=seasonality
                ).dropna()
                d_seasonal[seasonality] += 1
                p_value = self.test_stationarity(differenced_series)

            logger.info(
                f"Series is stationary after {d_seasonal[seasonality]} seasonal differencing(s) with period = {seasonality}."
            )

        # 5. Return the transformed DataFrame with the stationary series
        df[f"{target_column}_stationary"] = differenced_series
        logger.info(f"Final transformation: d={d_regular}, D: {d_seasonal}")
        return df
