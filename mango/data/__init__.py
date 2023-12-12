import os

import pandas

from .calendar_data import get_calendar


def get_ts_dataset():
    """
    Function to get the time series dataset used in the tests
    :return: the time series dataset
    :rtype: pandas.DataFrame
    """
    this_dir, file = os.path.split(__file__)
    df = pandas.read_pickle(f"{this_dir}/ts_dataset.pkl")
    return df
