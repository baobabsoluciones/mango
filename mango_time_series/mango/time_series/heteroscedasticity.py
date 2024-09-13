import numpy as np
import statsmodels.api as sm
from scipy.stats import boxcox, boxcox_normmax
from sklearn.linear_model import LinearRegression
from statsmodels.stats.diagnostic import het_breuschpagan

from mango_base.mango.logging.logger import get_basic_logger

logger = get_basic_logger()

try:
    import pandas as pd
    import polars as pl
except ImportError:
    pd = None
    pl = None


class HeteroscedasticityTester:
    def __init__(self, method: str = "moving_window"):
        """
        Initialize the HeteroscedasticityTester class with the specified method for detecting heteroscedasticity.
        Available methods: 'breusch_pagan', 'moving_window'

        :param method: The method for heteroscedasticity detection (default: 'breusch_pagan').
        """
        self.method = method

    @staticmethod
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

    @staticmethod
    def apply_boxcox_with_lambda(series: np.ndarray, lambda_value: float) -> np.ndarray:
        """
        Apply the Box-Cox transformation using a specified lambda value.

        :param series: A numpy array representing the time series data to be transformed.
        :param lambda_value: The lambda value to use for the Box-Cox transformation.
        :return: A numpy array of the transformed time series.
        """
        min_value = min(series)
        if min_value < 0:
            series = series - min_value + 1

        transformed_series = boxcox(series, lmbda=lambda_value)
        return transformed_series

    @staticmethod
    def detect_heteroscedasticity_breusch_pagan(
        df: pl.DataFrame, target_column: str
    ) -> float:
        """
        Detect heteroscedasticity in a time series using the Breusch-Pagan test.

        :param df: A Polars dataframe containing the time series data.
        :param target_column: The name of the target column containing the time series values.
        :return: The p-value from the Breusch-Pagan test for heteroscedasticity.
        """
        df_pandas = df.to_pandas()

        if target_column not in df_pandas.columns:
            raise ValueError(f"Column '{target_column}' not found in DataFrame")

        series = df_pandas[target_column].values

        if len(series) <= 1:
            raise ValueError(
                "The time series must contain more than one data point for the test"
            )

        trend = np.arange(len(series))
        X = sm.add_constant(trend)
        model = sm.OLS(series, X).fit()

        residuals = model.resid
        bp_test = het_breuschpagan(residuals, X)
        pval = bp_test[1]

        if pval < 0.05:
            logger.info(
                f"Heteroscedasticity detected via Breusch-Pagan test (p-value = {pval:.4f})."
            )
        else:
            logger.info(
                f"No significant heteroscedasticity detected via Breusch-Pagan test (p-value = {pval:.4f})."
            )

        return pval

    @staticmethod
    def detect_heteroscedasticity_moving_window(
        df: pl.DataFrame, target_column: str, window_width: int
    ) -> tuple:
        """
        Detect heteroscedasticity using a moving window approach and compute Box-Cox lambda values.

        :param df: A Polars dataframe containing the time series data.
        :param target_column: The name of the target column containing the time series values.
        :param window_width: The size of the moving window used to compute the moving averages and standard deviations.
        :return: A tuple containing the R-squared value and lambda from the slope of the regression.
        """
        if window_width < 1:
            raise ValueError("window_width must be at least 1")
        if window_width > df.shape[0]:
            raise ValueError(
                f"window_width {window_width} is greater than the number of data samples."
            )

        target_values = df[target_column].to_numpy()

        min_value = min(target_values)
        if min_value < 0:
            target_values = target_values - min_value + 1

        log_moving_avg = np.zeros((len(target_values) - window_width + 1,))
        log_moving_std = np.zeros((len(target_values) - window_width + 1,))

        for i in range(log_moving_std.shape[0]):
            log_moving_avg[i] = np.log(np.mean(target_values[i : (i + window_width)]))
            log_moving_std[i] = np.log(np.std(target_values[i : (i + window_width)]))

        df_log = pl.DataFrame({"log_ma": log_moving_avg, "log_sd": log_moving_std})

        linear_model = LinearRegression().fit(
            df_log["log_ma"].to_numpy().reshape(-1, 1), df_log["log_sd"].to_numpy()
        )

        r_squared = linear_model.score(
            df_log["log_ma"].to_numpy().reshape(-1, 1), df_log["log_sd"].to_numpy()
        )
        slope = linear_model.coef_[0]
        lambda_from_slope = 1 - slope

        return r_squared, lambda_from_slope

    def apply_transformations(
        self, df: pl.DataFrame, target_column: str, window_width: int = 12
    ) -> tuple:
        """
        Apply transformations based on the selected heteroscedasticity detection method (Breusch-Pagan or moving window).
        If heteroscedasticity is detected, apply the Box-Cox transformation with the optimal lambda.

        :param df: The Polars DataFrame containing the time series data.
        :param target_column: The name of the target column that holds the series values.
        :param window_width: The size of the moving window (only used for the 'moving_window' method).
        :return: A tuple with the transformed series and the lambda value.
        """
        if self.method == "breusch_pagan":
            pval = self.detect_heteroscedasticity_breusch_pagan(df, target_column)
            if pval < 0.05:
                target_values = df[target_column].to_numpy()
                optimal_lambda = self.get_optimal_lambda(target_values)
                transformed_series = self.apply_boxcox_with_lambda(
                    target_values, optimal_lambda
                )
                return transformed_series, optimal_lambda
            else:
                logger.info(
                    "No heteroscedasticity detected, no transformation applied."
                )
                return df[target_column].to_numpy(), None

        elif self.method == "moving_window":
            r_squared, lambda_from_slope = self.detect_heteroscedasticity_moving_window(
                df, target_column, window_width
            )
            if r_squared > 0.4:
                logger.info("Detected heteroscedasticity via moving window method.")
                target_values = df[target_column].to_numpy()
                transformed_series = self.apply_boxcox_with_lambda(
                    target_values, lambda_from_slope
                )
                return transformed_series, lambda_from_slope
            else:
                logger.info(
                    "No significant heteroscedasticity detected via moving window method."
                )
                return df[target_column].to_numpy(), None

        else:
            raise ValueError(
                "Invalid method. Choose either 'breusch_pagan' or 'moving_window'."
            )
