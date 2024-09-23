import numpy as np
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf
from mango_base.mango.logging.logger import get_basic_logger

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
        The function ensures that the most common period found through ACF has significant ACF values at multiples,
        and ignores isolated lags that do not show a repetitive pattern.

        :param ts: The time series data as a NumPy array.
        :param max_lag: The maximum lag to consider for ACF analysis. Default is set to 366 to cover yearly seasonality.
        :param acf_threshold: The ACF value threshold to consider peaks significant (default 0.2).
        :param min_repetitions: Minimum number of significant multiples to consider a period as a valid seasonality (default 3).
        :return: The most common detected seasonal period if significant; 0 otherwise.
        """
        acf_values = acf(ts, nlags=max_lag)

        # Find local maxima in the ACF values (indicative of seasonality)
        local_maxima = (np.diff(np.sign(np.diff(acf_values))) < 0).nonzero()[0] + 1

        significant_maxima = [lag for lag in local_maxima if acf_values[lag] > acf_threshold]

        if len(significant_maxima) > 1:
            # Calculate differences between local maxima to find potential periods
            potential_periods = np.diff(significant_maxima)
            most_common_period = int(np.bincount(potential_periods).argmax())

            # Check if the most common period has a sufficient number of significant ACF values at its multiples
            period_lags = np.arange(most_common_period, max_lag, most_common_period)

            valid_period_lags = period_lags[period_lags < len(acf_values)]
            # Count how many multiples of the period are above the ACF threshold
            significant_multiples = np.sum(acf_values[valid_period_lags] > acf_threshold)

            if (
                significant_multiples >= min_repetitions
            ):  # Only consider the period if enough multiples are significant
                return most_common_period
            else:
                return 0
        else:
            return 0

    def detect_seasonality_periodogram(
        self, ts: np.ndarray, min_period: int = 2, max_period: int = 365
    ) -> list:
        """
        Detect seasonality in a time series using the periodogram.

        :param ts: The time series data as a numpy array.
        :param min_period: The minimum period to consider as a seasonality (default 2 days).
        :param max_period: The maximum period to consider as a seasonality (default 365 days).
        :return: A list of detected seasonal periods.
        """
        # Calculate the periodogram (frequencies and power spectrum)
        frequencies, power_spectrum = periodogram(ts)

        # Convert frequencies to periods (skip the zero frequency to avoid division by zero)
        periods = 1 / frequencies[1:]
        power_spectrum = power_spectrum[1:]

        # Filter periods within the desired range (min_period to max_period)
        valid_periods = (periods >= min_period) & (periods <= max_period)
        filtered_periods = periods[valid_periods]
        filtered_power_spectrum = power_spectrum[valid_periods]

        # Detect peaks in the power spectrum that are above the threshold
        significant_periods = filtered_periods[
            filtered_power_spectrum
            > np.percentile(filtered_power_spectrum, self.percentile_periodogram)
        ]

        final_detected_periods = sorted(np.unique(np.round(significant_periods)))

        return final_detected_periods

    def detect_seasonality(self, ts: np.ndarray, max_lag: int = 366) -> list:
        """
        Detect seasonality in a time series using ACF to check if there is seasonality,
        and then use the periodogram to identify the exact seasonal periods.

        :param ts: The time series data as a numpy array.
        :param max_lag: The maximum lag to consider for ACF analysis. Default is set to 366 to cover yearly seasonality.
        :return: A list of detected seasonal periods. If no seasonality is detected, an empty list is returned.
        """
        # Adjust max_lag based on the length of the time series
        if max_lag is None:
            max_lag = min(366, len(ts))

        # Step 1: Check for the presence of seasonality using ACF
        if self.detect_significant_seasonality_acf(ts, max_lag):
            # Step 2: If seasonality is detected by ACF, use the periodogram to find the specific seasonal periods
            detected_seasonalities = self.detect_seasonality_periodogram(ts)
            if len(detected_seasonalities) > 0:
                logger.info(f"Seasonalities detected: {detected_seasonalities}")
            else:
                logger.info("No significant seasonal periods detected.")
            return detected_seasonalities
        else:
            logger.info("No seasonality detected by ACF.")
            return []
