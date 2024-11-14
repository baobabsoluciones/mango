from mango_time_series.utils.processing import process_time_series


class TimeSeriesProcessor:
    """
    Base class for processing time series
    :param config: configuration object
    """

    def __init__(self, config):
        """
        Constructor
        :param config: configuration object
        """
        self.config = config
        self.data = None

    def load_data(self, data):
        """
        Load data
        :param data: data to load
        """
        self.data = self.preprocess_raw_data(data)

    def preprocess_raw_data(self, data):
        """
        Preprocess raw data
        :param data: data to preprocess
        :return: preprocessed data
        """
        return process_time_series(data, self.config.__dict__)

    def process_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")
