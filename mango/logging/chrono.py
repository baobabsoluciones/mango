import time
import logging

logger = logging.getLogger("root")


class Chrono:
    """
    Class to measure time
    """

    def __init__(self, name: str, silent: bool = False, precision: int = 2):
        """
        Constructor of the class

        :param name: name of the chrono
        :param silent: if the chrono should be silent
        :param precision: number of decimals to round the time to
        """
        self.silent = silent
        self.precision = precision
        self.start_time = {name: time.time()}
        self.end = {name: None}

    def new(self, name: str):
        """
        Method to create a new chrono

        :param name: name of the chrono
        """
        self.start_time[name] = None
        self.end[name] = None

    def start(self, name):
        """
        Method to start a chrono

        :param name: name of the chrono
        """
        self.new(name)
        self.start_time[name] = time.time()

    def stop(self, name: str):
        """
        Method to stop a chrono and get back its duration

        :param name: name of the chrono
        :return: the time elapsed for the specific chrono
        :rtype: float
        """
        self.end[name] = time.time()
        duration = self.end[name] - self.start_time[name]
        if not self.silent:
            logger.info(
                f"Operation {name} took: {round(duration, self.precision)} seconds"
            )
        return duration

    def stop_all(self):
        """
        Method to stop all chronos and get back a dict with their durations
        """
        durations = dict()
        for name in self.start_time.keys():
            if self.end[name] is None:
                durations[name] = self.stop(name)

        return durations

    def report(self, name: str, message: str = None):
        """
        Method to report the time of a chrono and get back its duration

        :param name: name of the chrono
        """

        if self.end[name] is not None:
            msg = f"Operation {name} took: {round(self.end[name] - self.start_time[name], self.precision)} seconds"
            if message is not None:
                msg = f"{msg}. {message}"

            logger.info(msg)

            return self.end[name] - self.start_time[name]
        else:
            duration = time.time() - self.start_time[name]
            msg = (
                f"Operation {name} is still running. "
                f"Time elapsed: {round(duration, self.precision)} seconds"
            )
            if message is not None:
                msg = f"{msg}. {message}"

            logger.info(msg)

            return duration

    def report_all(self):
        """
        Method to report the time of all chronos and get back a dict with all the durations
        """
        durations = dict()
        for name in self.start_time.keys():
            durations[name] = self.report(name)
        return durations

    def __call__(self, func: callable, *args, **kwargs):
        """
        Method to decorate a function

        :param func: function to decorate
        :param args: arguments to pass to the function
        :param kwargs: keyword arguments to pass to the function
        :return: the result of the function
        """
        self.start(func.__name__)
        result = func(*args, **kwargs)
        self.stop(func.__name__)
        self.report(func.__name__)
        return result
