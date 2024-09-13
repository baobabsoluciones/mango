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

    def detect_seasonality_acf(self, ts: np.ndarray, max_lag: int = 366) -> list:
        """
        Detect seasonality or multiple seasonalities in time series data using ACF analysis.
        The function ensures that only the smallest significant period is detected and removes any multiples
        (e.g., if 7 days is detected, it will remove 14, 21, etc.). Other independent seasonalities are also detected.

        :param ts: The time series data as a NumPy array.
        :param max_lag: The maximum lag to consider for ACF analysis. Default is set to 366 to cover yearly seasonality.
        :return: A list of detected seasonal periods. If no seasonality is detected, an empty list is returned.
        """

        # Ensure max_lag does not exceed the length of the time series
        max_lag = min(max_lag, len(ts) - 1)

        # Calculate ACF values up to max_lag
        acf_values = acf(ts, nlags=max_lag)

        # Find local maxima in the ACF values (indicative of seasonality)
        local_maxima = (np.diff(np.sign(np.diff(acf_values))) < 0).nonzero()[0] + 1

        # Filter local maxima that have ACF values greater than the threshold
        significant_maxima = [lag for lag in local_maxima if acf_values[lag] > self.threshold_acf]

        # List to hold detected seasonal periods
        detected_periods = []

        if len(significant_maxima) > 0:
            # Start by detecting the smallest period (like 7 for weekly seasonality)
            smallest_period = significant_maxima[0]
            detected_periods.append(smallest_period)

            # Separate the remaining lags that are not multiples of the smallest period
            non_multiples = [lag for lag in significant_maxima if lag % smallest_period != 0]

            # Add non-multiple periods to detected periods
            detected_periods.extend(non_multiples)

        final_detected_periods = sorted(list(set(detected_periods)))

        return final_detected_periods

    def detect_seasonality_periodogram(self, ts: np.ndarray, min_period: int = 2, max_period: int = 365) -> list:
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
            filtered_power_spectrum > np.percentile(filtered_power_spectrum, self.percentile_periodogram)
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
        # Step 1: Check for the presence of seasonality using ACF
        if self.detect_seasonality_acf(ts, max_lag):
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
