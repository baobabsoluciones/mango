from mango_time_series.mango.utils.processing import process_time_series


class TimeSeriesProcessor:
    def __init__(self, config):
        self.config = config
        self.data = None

    def load_data(self, data):
        self.data = self.preprocess_raw_data(data)

    def preprocess_raw_data(self, data):
        return process_time_series(data, self.config.__dict__)

    def process_data(self):
        raise NotImplementedError("This method should be overridden by subclasses")