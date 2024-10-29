import logging
import time


class Chrono:
    """
    Class to measure time
    """

    def __init__(
        self, name: str, silent: bool = False, precision: int = 2, logger: str = "root"
    ):
        """
        Constructor of the class

        :param name: name of the chrono
        :param silent: if the chrono should be silent
        :param precision: number of decimals to round the time to
        :param str logger: the name of the logger to use. It defaults to root
        """
        self.silent = silent
        self.precision = precision
        self.start_time = {name: time.time()}
        self.end = {name: None}
        self.logger = logging.getLogger(logger)

    def new(self, name: str):
        """
        Method to create a new chrono

        :param name: name of the chrono
        """
        self.start_time[name] = None
        self.end[name] = None

    def start(self, name, silent=True):
        """
        Method to start a chrono

        :param str name: name of the chrono
        :param bool silent: if the chrono should be silent. True by default
        """
        self.new(name)
        self.start_time[name] = time.time()
        if not silent:
            self.logger.info(f"Operation {name} starts")

    def stop(self, name: str, report: bool = True):
        """
        Method to stop a chrono and get back its duration

        :param str name: name of the chrono
        :param bool report: if the chrono should be reported. True by default
        :return: the time elapsed for the specific chrono
        :rtype: float
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
        Method to stop all chronos and get back a dict with their durations

        :param bool report: if the chronos should be reported. True by default
        :return: a dict with the name of the chronos as key and the durations as value
        :rtype: dict
        """
        durations = dict()
        for name in self.start_time.keys():
            if self.end[name] is None:
                durations[name] = self.stop(name, report)

        return durations

    def report(self, name: str, message: str = None):
        """
        Method to report the time of a chrono and get back its duration

        :param str name: name of the chrono
        :param str message: additional message to display in the log
        :return: the time elapsed for the specific chrono
        :rtype: float
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
        Method to report the time of all chronos and get back a dict with all the durations

        :return: a dict with the name of the chronos as key and the durations as value
        :rtype: dict
        """
        durations = dict()
        for name in self.start_time.keys():
            durations[name] = self.report(name)
        return durations

    def __call__(self, func: callable, *args, **kwargs):
        """
        Method to use the chrono as a callable with a function inside

        :param func: function to decorate
        :param args: arguments to pass to the function
        :param kwargs: keyword arguments to pass to the function
        :return: the result of the function
        """
        self.start(func.__name__)
        result = func(*args, **kwargs)
        self.stop(func.__name__, False)
        self.report(func.__name__)
        return result
