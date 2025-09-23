import warnings

from mango.processing import as_list, flatten
from pytups import TupList


def is_subset(set_a, set_b):
    """
    Check if set_a is a subset of set_b without iterating over strings.

    Determines whether all elements in set_a are also present in set_b.
    Handles both individual strings and collections of elements.

    :param set_a: A group of elements (can be a string or collection)
    :type set_a: str or list or set
    :param set_b: Another group of elements (can be a string or collection)
    :type set_b: str or list or set
    :return: True if set_a is a subset of set_b
    :rtype: bool

    Example:
        >>> print(is_subset("a", "abc"))
        True
        >>> print(is_subset(["a", "b"], ["a", "b", "c"]))
        True
        >>> print(is_subset(["a", "d"], ["a", "b", "c"]))
        False
    """
    set_a = {set_a} if isinstance(set_a, str) else set(set_a)
    set_b = {set_b} if isinstance(set_b, str) else set(set_b)
    return set_a.issubset(set_b)


def str_key(dic):
    """
    Transform dictionary keys to strings.

    Converts all keys in a dictionary to string format.
    This is useful when converting a dictionary to JSON format
    or when working with mixed key types.

    :param dic: Dictionary to transform
    :type dic: dict
    :return: Same dictionary with keys converted to strings
    :rtype: dict
    :warns SyntaxWarning: If some keys are lost due to int/str conflicts

    Example:
        >>> result = str_key({1: "one", 2: "two", "3": "three"})
        >>> print(result)
        {'1': 'one', '2': 'two', '3': 'three'}
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
    Extend a single value to a list of specified length.

    If the input is a single value, repeats it to create a list
    of the specified length. If the input is already a list,
    returns it unchanged.

    :param lst: List, number, or string to extend
    :type lst: list or any
    :param n: Target length for the resulting list
    :type n: int
    :return: List of length n or the original list
    :rtype: list

    Example:
        >>> result = to_len(5, 3)
        >>> print(result)
        [5, 5, 5]
        >>> result = to_len([1, 2, 3], 5)
        >>> print(result)
        [1, 2, 3]
    """
    if len(as_list(lst)) == 1:
        return as_list(lst) * n
    return lst


def join_lists(*lists):
    """
    Concatenate multiple lists into a single list.

    Combines all provided lists into one flat list.
    Handles any number of input lists.

    :param lists: Any number of lists to join
    :type lists: list
    :return: Single list containing all elements
    :rtype: list

    Example:
        >>> result = join_lists([1, 2], [3, 4], [5, 6])
        >>> print(result)
        [1, 2, 3, 4, 5, 6]
        >>> result = join_lists(["a", "b"], ["c"])
        >>> print(result)
        ['a', 'b', 'c']
    """
    return [i for lst in lists for i in as_list(lst)]


def mean(*args):
    """
    Calculate the arithmetic mean of a series of values.

    Computes the average of all provided values, handling both
    individual scalars and lists of values.

    :param args: Scalars or lists of scalars to average
    :type args: number or list[number]
    :return: The arithmetic mean, or None if no values provided
    :rtype: float or None

    Example:
        >>> result = mean(1, 2, 3, 4, 5)
        >>> print(result)
        3.0
        >>> result = mean([10, 20], [30, 40])
        >>> print(result)
        25.0
        >>> result = mean()
        >>> print(result)
        None
    """
    args = flatten(as_list(args))
    if len(args) > 0:
        return sum(i for i in args) / len(args)
    else:
        return None


def cumsum(x):
    """
    Calculate the cumulative sum of a list.

    Returns a new list where each element is the sum of all
    previous elements plus the current element.

    :param x: List of numbers to compute cumulative sum for
    :type x: list[number]
    :return: List of cumulative sums with same length as input
    :rtype: list[number]

    Example:
        >>> result = cumsum([1, 2, 3, 4, 5])
        >>> print(result)
        [1, 3, 6, 10, 15]
        >>> result = cumsum([10, -5, 3])
        >>> print(result)
        [10, 5, 8]
    """
    return [sum(x[: (i + 1)]) for i in range(len(x))]


def invert_dict_list(dictlist, unique=True):
    """
    Transform a list of dictionaries into a dictionary of lists.

    Converts from row-oriented format (list of dicts) to column-oriented
    format (dict of lists). Each key becomes a column with all its values.

    :param dictlist: TupList containing dictionaries (not tuples)
    :type dictlist: TupList
    :param unique: Whether to remove duplicates in the resulting lists
    :type unique: bool
    :return: Dictionary with keys as column names and lists as values
    :rtype: dict
    :raises AssertionError: If input is not a TupList

    Example:
        >>> dictlist = TupList([
        ...     {"name": "Alice", "age": 30},
        ...     {"name": "Bob", "age": 25},
        ...     {"name": "Alice", "age": 30}  # Duplicate
        ... ])
        >>> result = invert_dict_list(dictlist, unique=True)
        >>> print(result)
        {'name': 'Alice', 'age': 30}  # Simplified due to unique=True
        >>> result = invert_dict_list(dictlist, unique=False)
        >>> print(result)
        {'name': ['Alice', 'Bob', 'Alice'], 'age': [30, 25, 30]}
    """
    assert isinstance(dictlist, TupList)

    inverted = {k: [d.get(k, None) for d in dictlist] for k in dictlist[0].keys()}
    if unique:
        return {k: simplify(v) for k, v in inverted.items()}
    else:
        return inverted


def simplify(x):
    """
    Simplify a list by removing duplicates and reducing single-element lists.

    Removes duplicate values from a list and converts single-element
    lists to their scalar value. Non-list inputs are returned unchanged.

    :param x: List to simplify
    :type x: list or any
    :return: Simplified list or scalar value
    :rtype: list or any

    Example:
        >>> result = simplify([1, 2, 2, 3, 1])
        >>> print(result)
        [1, 2, 3]
        >>> result = simplify([42])
        >>> print(result)
        42
        >>> result = simplify([1, 2, 3])
        >>> print(result)
        [1, 2, 3]
        >>> result = simplify("not a list")
        >>> print(result)
        not a list
    """
    if isinstance(x, list):
        x1 = list(set(x))
        if len(x1) == 1:
            return x1[0]
        else:
            return x1
    return x
