import logging
import time


class Chrono:
    """
    High-precision timing utility for measuring operation durations.

    This class provides functionality to measure and track execution times
    for multiple named operations. It supports individual timing, batch
    operations, and can be used as a decorator for automatic function
    timing. All timing information is logged using the configured logger.

    :param name: Name of the initial chronometer to create
    :type name: str
    :param silent: Whether to suppress automatic logging of timing events
    :type silent: bool
    :param precision: Number of decimal places for time reporting
    :type precision: int
    :param logger: Name of the logger to use for output
    :type logger: str

    Example:
        >>> chrono = Chrono("operation1")
        >>> chrono.start("data_processing")
        >>> # ... perform operation ...
        >>> duration = chrono.stop("data_processing")
        >>> print(f"Operation took {duration:.2f} seconds")
    """

    def __init__(
        self, name: str, silent: bool = False, precision: int = 2, logger: str = "root"
    ):
        """
        Initialize a new Chrono instance with timing capabilities.

        Creates a new chronometer instance with the specified configuration.
        The initial chronometer is automatically started upon creation.
        """
        self.silent = silent
        self.precision = precision
        self.start_time = {name: time.time()}
        self.end = {name: None}
        self.logger = logging.getLogger(logger)

    def new(self, name: str):
        """
        Create a new chronometer entry without starting it.

        Initializes a new chronometer with the specified name but does not
        start timing. Use start() to begin timing this chronometer.

        :param name: Name of the new chronometer to create
        :type name: str
        :return: None

        Example:
            >>> chrono = Chrono("main")
            >>> chrono.new("sub_operation")
            >>> chrono.start("sub_operation")
        """
        self.start_time[name] = None
        self.end[name] = None

    def start(self, name, silent=True):
        """
        Start timing a chronometer.

        Begins timing the specified chronometer. If the chronometer doesn't
        exist, it will be created first. Optionally logs the start event.

        :param name: Name of the chronometer to start
        :type name: str
        :param silent: Whether to suppress the start log message
        :type silent: bool
        :return: None

        Example:
            >>> chrono = Chrono("main")
            >>> chrono.start("data_processing", silent=False)
        """
        self.new(name)
        self.start_time[name] = time.time()
        if not silent:
            self.logger.info(f"Operation {name} starts")

    def stop(self, name: str, report: bool = True):
        """
        Stop timing a chronometer and return the elapsed duration.

        Stops the specified chronometer and calculates the elapsed time.
        Optionally logs the completion with the duration.

        :param name: Name of the chronometer to stop
        :type name: str
        :param report: Whether to log the completion message
        :type report: bool
        :return: Elapsed time in seconds
        :rtype: float
        :raises KeyError: If the chronometer name doesn't exist

        Example:
            >>> chrono = Chrono("main")
            >>> chrono.start("operation")
            >>> # ... perform operation ...
            >>> duration = chrono.stop("operation")
            >>> print(f"Completed in {duration:.3f} seconds")
        """
        self.end[name] = time.time()
        duration = self.end[name] - self.start_time[name]
        if not self.silent or report:
            self.logger.info(
                f"Operation {name} took: {round(duration, self.precision)} seconds"
            )
        return duration

    def stop_all(self, report: bool = True):
        """
        Stop all running chronometers and return their durations.

        Stops all chronometers that are currently running and returns a
        dictionary mapping chronometer names to their elapsed durations.
        Only chronometers that haven't been stopped yet are affected.

        :param report: Whether to log completion messages for each chronometer
        :type report: bool
        :return: Dictionary mapping chronometer names to durations in seconds
        :rtype: dict

        Example:
            >>> chrono = Chrono("main")
            >>> chrono.start("op1")
            >>> chrono.start("op2")
            >>> durations = chrono.stop_all()
            >>> print(f"Total operations: {len(durations)}")
        """
        durations = dict()
        for name in self.start_time.keys():
            if self.end[name] is None:
                durations[name] = self.stop(name, report)

        return durations

    def report(self, name: str, message: str = None):
        """
        Report the current status and duration of a chronometer.

        Logs the current timing information for the specified chronometer.
        If the chronometer has been stopped, reports the final duration.
        If still running, reports the current elapsed time.

        :param name: Name of the chronometer to report
        :type name: str
        :param message: Optional additional message to include in the log
        :type message: str, optional
        :return: Elapsed time in seconds (final duration if stopped, current if running)
        :rtype: float
        :raises KeyError: If the chronometer name doesn't exist

        Example:
            >>> chrono = Chrono("main")
            >>> chrono.start("long_operation")
            >>> # ... some time later ...
            >>> elapsed = chrono.report("long_operation", "Still processing...")
        """

        if self.end[name] is not None:
            msg = f"Operation {name} took: {round(self.end[name] - self.start_time[name], self.precision)} seconds"
            if message is not None:
                msg = f"{msg}. {message}"

            self.logger.info(msg)

            return self.end[name] - self.start_time[name]
        else:
            duration = time.time() - self.start_time[name]
            msg = (
                f"Operation {name} is still running. "
                f"Time elapsed: {round(duration, self.precision)} seconds"
            )
            if message is not None:
                msg = f"{msg}. {message}"

            self.logger.info(msg)

            return duration

    def report_all(self):
        """
        Report the status and durations of all chronometers.

        Logs timing information for all chronometers and returns a dictionary
        with their current durations. For stopped chronometers, reports final
        duration; for running ones, reports current elapsed time.

        :return: Dictionary mapping chronometer names to durations in seconds
        :rtype: dict

        Example:
            >>> chrono = Chrono("main")
            >>> chrono.start("op1")
            >>> chrono.start("op2")
            >>> chrono.stop("op1")
            >>> durations = chrono.report_all()
            >>> # Reports final duration for op1, current elapsed for op2
        """
        durations = dict()
        for name in self.start_time.keys():
            durations[name] = self.report(name)
        return durations

    def __call__(self, func: callable, *args, **kwargs):
        """
        Use the chronometer as a decorator to automatically time function execution.

        Allows the chronometer to be used as a decorator or called directly
        with a function to automatically measure its execution time. The
        function name is used as the chronometer name.

        :param func: Function to time and execute
        :type func: callable
        :param args: Positional arguments to pass to the function
        :param kwargs: Keyword arguments to pass to the function
        :return: Result returned by the function
        :rtype: Any

        Example:
            >>> chrono = Chrono("main")
            >>> @chrono
            >>> def my_function(x, y):
            ...     return x + y
            >>> result = my_function(1, 2)  # Automatically timed
            >>> # Or call directly:
            >>> result = chrono(my_function, 1, 2)
        """
        self.start(func.__name__)
        result = func(*args, **kwargs)
        self.stop(func.__name__, False)
        self.report(func.__name__)
        return result
