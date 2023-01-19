import time
import logging as log


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
        Method to stop a chrono

        :param name: name of the chrono
        """
        self.end[name] = time.time()
        if not self.silent:
            log.info(
                f"Operation {name} took: {round(self.end[name] - self.start_time[name], self.precision)} seconds"
            )

    def stop_all(self):
        """
        Method to stop all chronos
        """
        for name in self.start_time.keys():
            if self.end[name] is None:
                self.stop(name)

    def report(self, name):
        """
        Method to report the time of a chrono

        :param name: name of the chrono
        """
        if self.end[name] is not None:
            log.info(
                f"Operation {name} took: {round(self.end[name] - self.start_time[name], self.precision)} seconds"
            )
        else:
            log.info(
                f"Operation {name} is still running. "
                f"Time elapsed: {round(time.time() - self.start_time[name], self.precision)}"
            )

    def report_all(self):
        """
        Method to report the time of all chronos
        """
        for name in self.start_time.keys():
            self.report(name)

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
