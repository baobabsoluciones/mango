import pickle
from collections.abc import Iterable

try:
    import pandas as pd
except ImportError:
    pd = None


def pickle_copy(instance):
    """
    The pickle_copy function accepts an instance of a class and returns a copy of that
    instance. The pickle module is used to create the copy, so it can be unpickled again.

    :param instance: specify the object to be copied
    :return: a copy of the instance
    :doc-author: baobab soluciones
    """
    return pickle.loads(pickle.dumps(instance, -1))


def unique(lst: list):
    """
    The unique function takes a list and returns the unique elements of that list.
    For example, if given [2, 2, 3], it will return [2, 3].

    :param lst: the list from where to extract the unique values
    :return: a list of unique values in the inputted list
    :doc-author: baobab soluciones
    """
    return list(set(lst))


def reverse_dict(data):
    """
    The reverse_dict function takes a dictionary and returns a dictionary with the keys and values
    reversed.

    :param data: the dictionary to be reversed
    :return: a dictionary with the keys and values reversed
    :doc-author: baobab soluciones
    """
    return {v: k for k, v in data.items()}


def cumsum(lst: list) -> list:
    """
    The cumsum function takes a list of numbers and returns a list of the cumulative sum of those numbers.

    :param lst: the list of numbers to be summed
    :return: a list of the cumulative sum of the inputted list
    :doc-author: baobab soluciones
    """
    return [sum(lst[: i + 1]) for i in range(len(lst))]


def lag_list(lst: list, lag: int = 1) -> list:
    """
    The lag_list function takes a list and returns a list with the values lagged by the specified amount.

    :param lst: the list to be lagged
    :param lag: the amount by which to lag the list
    :return: a list with the values lagged by the specified amount
    :doc-author: baobab soluciones
    """
    return [None] * lag + lst[:-lag]


def lead_list(lst: list, lead: int = 1) -> list:
    """
    The lead_list function takes a list and returns a list with the values led by the specified amount.

    :param lst: the list to be led
    :param lead: the amount by which to lead the list
    :return: a list with the values led by the specified amount
    :doc-author: baobab soluciones
    """
    return lst[lead:] + [None] * lead


def row_number(lst: list, start: int = 0) -> list:
    """
    The row_number function takes a list and returns a list with the row number of each element.

    :param lst: the list to be numbered
    :param start: the number to start the row numbering at
    :return: a list with the row number of each element
    :doc-author: baobab soluciones
    """
    return [i + start for i, _ in enumerate(lst)]


def flatten(lst: Iterable) -> list:
    """
    The flatten function takes a list of lists and returns a flattened list.

    :param lst: the list of lists to be flattened
    :return: a flattened list
    :doc-author: baobab soluciones
    """
    return [item for sublist in lst for item in as_list(sublist)]


def df_to_list(df: pd.DataFrame) -> list:
    """
    The data_frame_to_list function takes a DataFrame and returns a list of dictionaries with the
    column names as keys and the values as values.

    :param df: the DataFrame to be converted to a list of dictionaries
    :return: a list of dictionaries
    :doc-author: baobab soluciones
    """
    return df.to_dict(orient="records")


def df_to_dict(df: pd.DataFrame) -> dict:
    """
    The data_frame_to_dict function takes a dict of DataFrames and returns a dictionary with the
    sheet names as keys and the DataFrames in records as values.

    :param dict df: the dict of DataFrames to be converted to a dictionary
    :return: a dictionary
    :doc-author: baobab soluciones
    """
    return {name: content.to_dict(orient="records") for name, content in df.items()}


def as_list(x):
    """
    Transform an object into a list without nesting lists or iterating over strings.
    Behave like [x] if x is a scalar or a string and list(x) if x is another iterable.

    as_list(1) -> [1]
    as_list("one") -> ["one"]
    as_list([1,2]) -> [1,2]
    as_list({1,2}) -> [1,2]
    as_list((1,2)) -> [1,2]

    :param x: an object
    :return: a list
    """
    if isinstance(x, Iterable) and not isinstance(x, str) and not isinstance(x, dict):
        return list(x)
    else:
        return [x]


def first(lst):
    """
    Return the first value of a list.
    Return None if the list is empty instead of getting an error.

    :param lst: a list
    :return: a value
    """
    if not len(lst):
        return None
    return lst[0]


def last(lst):
    """
    Return the last value of a list.
    Return None if the list is empty instead of getting an error.

    :param lst: a list
    :return: a value
    """
    if not len(lst):
        return None
    return lst[-1]
