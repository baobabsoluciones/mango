import pandas


def get_ts_dataset():
    """
    Function to get the time series dataset used in the tests
    :return: the time series dataset
    :rtype: pandas.DataFrame
    """
    df = pandas.read_pickle("./ts_dataset.pkl")
    return df