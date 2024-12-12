import warnings

from mango.processing import as_list, flatten
from pytups import TupList


def is_subset(set_a, set_b):
    """
    Tell if set_a is a subset of set_b without iterating over strings.

    :param set_a: A group of element (can be a string)
    :param set_b: Another group of element (can be a string)
    :return: True if A is a subset of set_b
    """
    set_a = {set_a} if isinstance(set_a, str) else set(set_a)
    set_b = {set_b} if isinstance(set_b, str) else set(set_b)
    return set_a.issubset(set_b)


def str_key(dic):
    """
    Transform the keys of a dict into string.
    This is useful when converting a dict to json format.

    :param dic: a dictionary
    :return: the same dictionary with keys as strings.
    """
    result = {str(k): v for k, v in dic.items()}
    if len(result) != len(dic):
        n = len(dic) - len(result)
        warnings.warn(
            f"str_key has eliminated {n} keys of the dict,"
            f" this is due to some keys existing as int and str at the same time ('1' and 1)",
            category=SyntaxWarning,
        )
    return result


def to_len(lst, n):
    """
    Extend a single value to a list of length n.
    if lst is not a single value, return it unchanged.

    :param lst: list, number or string.
    :param n: integer.
    :return: a list of length n or lst.
    """
    if len(as_list(lst)) == 1:
        return as_list(lst) * n
    return lst


def join_lists(*lists):
    """
    Join lists together

    :param lists: any number of lists
    :return: a list
    example:
    a = join_lists([1,2], [3,4], [5,6])
    a: [1,2,3,4,5,6]
    """
    return [i for lst in lists for i in as_list(lst)]


def mean(*args):
    """
    Calculate the mean of a series.

    :param args: scalars or list of scalars
    :return: the mean
    """
    args = flatten(as_list(args))
    if len(args) > 0:
        return sum(i for i in args) / len(args)
    else:
        return None


def cumsum(x):
    """
    Return the cumulative sum of  a list.

    :param x: a list
    :return: a list of the same size as x
    """
    return [sum(x[: (i + 1)]) for i in range(len(x))]


def invert_dict_list(dictlist, unique=True):
    """
    Transform a list of dict into a dict of lists.
    [{a:1, b:2}, {a=2, b=3}] -> {a:[1,2], b:[2,3]}

    :param dictlist: a TupList (with dict and not tuples)
    :param unique: remove duplicates in the lists.
    :return: a dict
    """
    assert isinstance(dictlist, TupList)

    inverted = {k: [d.get(k, None) for d in dictlist] for k in dictlist[0].keys()}
    if unique:
        return {k: simplify(v) for k, v in inverted.items()}
    else:
        return inverted


def simplify(x):
    """
    Remove duplicates and replace lists of size 1 by x[0]

    :param x: a list
    :return: a simplified list.
    """
    if isinstance(x, list):
        x1 = list(set(x))
        if len(x1) == 1:
            return x1[0]
        else:
            return x1
    return x
