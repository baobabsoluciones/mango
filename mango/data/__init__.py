import os

import pandas


def get_ts_dataset():
    """
    Function to get the time series dataset used in the tests
    :return: the time series dataset
    :rtype: pandas.DataFrame
    """
    dir, file = os.path.split(__file__)
    df = pandas.read_pickle(f"{dir}/ts_dataset.pkl")
    return df
