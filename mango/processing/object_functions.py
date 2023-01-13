import pickle


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
