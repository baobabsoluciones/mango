import pandas
from mango.processing.file_functions import normalize_path


def get_ts_dataset():
    """
    Function to get the time series dataset used in the tests
    :return: the time series dataset
    :rtype: pandas.DataFrame
    """
    df = pandas.read_pickle(normalize_path("./ts_dataset.pkl"))
    return df
