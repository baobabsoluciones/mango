import numpy as np
from mango.logging.logger import get_basic_logger
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf

logger = get_basic_logger()


class SeasonalityDetector:
    def __init__(self, threshold_acf: float = 0.1, percentile_periodogram: float = 99):
        """
        Initialize the SeasonalityDetector with thresholds for ACF and periodogram.

        :param threshold_acf: The ACF value threshold to consider a peak significant (default 0.1).
        :param percentile_periodogram: The percentile to consider significant peaks in the periodogram (default 99).
        """
        self.threshold_acf = threshold_acf
        self.percentile_periodogram = percentile_periodogram

    @staticmethod
    def detect_significant_seasonality_acf(
        ts: np.ndarray,
        max_lag: int = 366,
        acf_threshold: float = 0.2,
        min_repetitions: int = 2,
    ) -> int:
        """
        Detect the most significant seasonality in time series data using ACF (Autocorrelation Function) analysis.

        This function calculates the ACF values for the time series up to the specified maximum lag. It identifies
        local maxima in the ACF values, which are indicative of potential seasonal periods. The local maxima are then
        filtered to retain only those that exceed the specified ACF threshold. The function determines the most common
        period among the significant local maxima and verifies that this period has a sufficient number of significant
        ACF values at its multiples. If the most common period meets these criteria, it is returned; otherwise, 0 is returned.

        :param ts: The time series data as a NumPy array.
        :param max_lag: The maximum lag to consider for ACF analysis. Default is set to 366 to cover yearly seasonality.
        :param acf_threshold: The ACF value threshold to consider peaks significant (default 0.2).
        :param min_repetitions: Minimum number of significant multiples to consider a period as a valid seasonality (default 2).
        :return: The most common detected seasonal period if significant; 0 otherwise.
        """
        acf_values, confint = acf(ts, nlags=max_lag, alpha=0.05)

        # Find local maxima in the ACF values (indicative of seasonality)
        local_maxima = (np.diff(np.sign(np.diff(acf_values))) < 0).nonzero()[0] + 1

        significant_maxima = [
            lag
            for lag in local_maxima
            if (acf_values[lag] > acf_threshold)
            and (acf_values[lag] > confint[lag, 1] or acf_values[lag] < confint[lag, 0])
        ]

        if len(significant_maxima) > 1:
            # Calculate differences between local maxima to find potential periods
            potential_periods = np.diff(significant_maxima)
            most_common_period = int(np.bincount(potential_periods).argmax())

            # Check if the most common period has a sufficient number of significant ACF values at its multiples
            period_lags = np.arange(most_common_period, max_lag, most_common_period)

            valid_period_lags = period_lags[period_lags < len(acf_values)]
            # Count how many multiples of the period are above the ACF threshold
            significant_multiples = np.sum(
                (acf_values[valid_period_lags] > acf_threshold)
                & (
                    (acf_values[valid_period_lags] > confint[valid_period_lags, 1])
                    | (acf_values[valid_period_lags] < confint[valid_period_lags, 0])
                )
            )

            if (
                significant_multiples >= min_repetitions
            ):  # Only consider the period if enough multiples are significant
                return most_common_period
            else:
                return 0
        else:
            return 0

    @staticmethod
    def detect_seasonality_periodogram(
        ts: np.ndarray, min_period: int = 2, max_period: int = 365
    ) -> [list, np.ndarray, np.ndarray]:
        """
        Detect seasonality in a time series using the periodogram.

        :param ts: The time series data as a numpy array.
        :param min_period: The minimum period to consider as a seasonality (default 2 days).
        :param max_period: The maximum period to consider as a seasonality (default 365 days).
        :return: A list of detected seasonal periods.
        """

        frequencies, power_spectrum = periodogram(x=ts)

        # Convert frequencies to periods (skip the zero frequency to avoid division by zero)
        periods = 1 / frequencies[1:]
        power_spectrum = power_spectrum[1:]

        # Filter periods within the desired range (min_period to max_period)
        valid_periods = (periods >= min_period) & (periods <= max_period)
        filtered_periods = periods[valid_periods]
        filtered_power_spectrum = power_spectrum[valid_periods]

        strict_percentile = 99
        threshold_value = np.percentile(filtered_power_spectrum, strict_percentile)

        # Detect peaks above the stricter threshold
        significant_periods = filtered_periods[
            filtered_power_spectrum > threshold_value
        ]

        # Refine peaks by ensuring sufficient difference in power
        refined_periods = []
        for i, period in enumerate(significant_periods):
            if (
                i == 0
                or filtered_power_spectrum[i] > 1.5 * filtered_power_spectrum[i - 1]
            ):
                refined_periods.append(period)

        # Remove redundant multiples of detected periods and keep only near-integer periods
        final_detected_periods = []
        for period in refined_periods:
            # Check if the period is close to an integer within a small tolerance (e.g., 0.05)
            if np.isclose(period, round(period), atol=0.05):
                rounded_period = round(period)
                # Ensure no redundant multiples of detected periods
                if not any(
                    np.isclose(rounded_period % other, 0, atol=0.1)
                    for other in final_detected_periods
                ):
                    final_detected_periods.append(rounded_period)

        return final_detected_periods, filtered_periods, filtered_power_spectrum

    def detect_seasonality(self, ts: np.ndarray, max_lag: int = 366) -> list:
        """
        Detect seasonality in a time series using ACF to check if there is seasonality,
        and then use the periodogram to identify the exact seasonal periods.

        ACF (Autocorrelation Function) is used to detect if there is seasonality in the time series.
        This is because the periodogram could find some periodic components even if there is no true seasonality.
        If ACF indicates the presence of seasonality, then the periodogram is used to accurately identify the specific seasonal periods.

        :param ts: The time series data as a numpy array.
        :param max_lag: The maximum lag to consider for ACF analysis. Default is set to 366 to cover yearly seasonality.
        :return: A list of detected seasonal periods. If no seasonality is detected, an empty list is returned.
        """
        # Adjust max_lag based on the length of the time series
        if len(ts) < max_lag:
            max_lag = len(ts) - 1

        # Step 1: Detect potential seasonality using ACF
        most_common_period = self.detect_significant_seasonality_acf(
            ts, max_lag=max_lag
        )

        # Initialize list of detected periods
        detected_periods = []

        # If ACF suggests a significant period, verify it with the periodogram
        if most_common_period:
            detected_periods.append(most_common_period)

        # Step 2: Use periodogram to validate the ACF period and find additional significant periods
        periodogram_periods, _, _ = self.detect_seasonality_periodogram(ts)
        for period in periodogram_periods:
            if period not in detected_periods:
                detected_periods.append(period)

        return sorted(detected_periods)
