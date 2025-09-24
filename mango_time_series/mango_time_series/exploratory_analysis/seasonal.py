import numpy as np
from mango_time_series.logging import get_configured_logger
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf

logger = get_configured_logger()


class SeasonalityDetector:
    """
    Detector for identifying seasonal patterns in time series data.

    Combines autocorrelation function (ACF) analysis and periodogram analysis
    to detect and validate seasonal patterns in time series. Uses configurable
    thresholds to determine significance of detected patterns.
    """

    def __init__(self, threshold_acf: float = 0.1, percentile_periodogram: float = 99):
        """
        Initialize the SeasonalityDetector with analysis thresholds.

        Sets up the detector with configurable thresholds for determining
        the significance of seasonal patterns detected through different methods.

        :param threshold_acf: ACF value threshold for significant peaks
        :type threshold_acf: float
        :param percentile_periodogram: Percentile threshold for periodogram peaks
        :type percentile_periodogram: float
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
        Detect significant seasonality using autocorrelation function analysis.

        Analyzes the time series using ACF to identify seasonal patterns by finding
        local maxima in autocorrelation values. Validates detected periods by ensuring
        sufficient repetitions at period multiples, indicating true seasonality.

        :param ts: Time series data to analyze
        :type ts: numpy.ndarray
        :param max_lag: Maximum lag for ACF analysis (default: 366 for yearly seasonality)
        :type max_lag: int
        :param acf_threshold: ACF threshold for significant peaks (default: 0.2)
        :type acf_threshold: float
        :param min_repetitions: Minimum significant multiples for valid seasonality (default: 2)
        :type min_repetitions: int
        :return: Most significant seasonal period, or 0 if none detected
        :rtype: int

        Note:
            - Identifies local maxima in ACF values as potential seasonal periods
            - Filters peaks above the ACF threshold and confidence intervals
            - Validates periods by checking for significant ACF values at multiples
            - Returns 0 if no valid seasonality pattern is found
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
    ) -> tuple[list, np.ndarray, np.ndarray]:
        """
        Detect seasonality using periodogram analysis.

        Analyzes the power spectral density of the time series to identify
        significant periodic components. Filters periods within specified range,
        applies strict percentile thresholds, and refines peaks to avoid
        redundant multiples while keeping only near-integer periods.

        :param ts: Time series data to analyze
        :type ts: numpy.ndarray
        :param min_period: Minimum period to consider (default: 2)
        :type min_period: int
        :param max_period: Maximum period to consider (default: 365)
        :type max_period: int
        :return: Tuple containing:
            - List of detected seasonal periods
            - Array of filtered periods
            - Array of filtered power spectrum values
        :rtype: tuple[list, numpy.ndarray, numpy.ndarray]

        Note:
            - Uses 99th percentile threshold for peak detection
            - Refines peaks by ensuring sufficient power difference (1.5x)
            - Removes redundant multiples of detected periods
            - Keeps only periods close to integers (tolerance: 0.05)
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
        Detect seasonality using combined ACF and periodogram analysis.

        Implements a two-step approach for robust seasonality detection:
        1. Uses ACF analysis to confirm the presence of seasonality
        2. Uses periodogram analysis to identify specific seasonal periods

        This combination prevents false positives from periodogram analysis
        while ensuring accurate identification of true seasonal patterns.

        :param ts: Time series data to analyze
        :type ts: numpy.ndarray
        :param max_lag: Maximum lag for ACF analysis (default: 366 for yearly seasonality)
        :type max_lag: int
        :return: Sorted list of detected seasonal periods, empty if none found
        :rtype: list

        Note:
            - ACF analysis validates the presence of seasonality
            - Periodogram analysis identifies specific periods
            - Automatically adjusts max_lag based on time series length
            - Combines results from both methods, removing duplicates
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
