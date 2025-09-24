import pickle
from collections.abc import Iterable

try:
    import pandas as pd
except ImportError:
    pd = None

from mango.logging import get_configured_logger

log = get_configured_logger(__name__)


def pickle_copy(instance):
    """
    Create a deep copy of an object using pickle serialization.

    Uses Python's pickle module to serialize and deserialize the object,
    creating a complete deep copy. This method works with any pickleable
    object and preserves the exact state of the original.

    :param instance: Object to be copied
    :type instance: Any
    :return: Deep copy of the original object
    :rtype: Any
    :raises pickle.PicklingError: If the object cannot be pickled
    :raises pickle.UnpicklingError: If the object cannot be unpickled

    Example:
        >>> original = {"a": [1, 2, 3], "b": {"nested": True}}
        >>> copy = pickle_copy(original)
        >>> copy["a"].append(4)
        >>> print(original["a"])  # [1, 2, 3] - original unchanged
        >>> print(copy["a"])      # [1, 2, 3, 4] - copy modified
    """
    return pickle.loads(pickle.dumps(instance, -1))


def unique(lst: list):
    """
    Extract unique elements from a list.

    Returns a list containing only the unique elements from the input list.
    The order of elements in the result is not guaranteed as it uses set
    operations internally.

    :param lst: List from which to extract unique values
    :type lst: list
    :return: List of unique values from the input list
    :rtype: list

    Example:
        >>> unique([2, 2, 3, 1, 3, 1])
        [1, 2, 3]
        >>> unique(['a', 'b', 'a', 'c'])
        ['a', 'b', 'c']
    """
    return list(set(lst))


def reverse_dict(data):
    """
    Reverse the key-value pairs in a dictionary.

    Creates a new dictionary where the original values become keys and
    the original keys become values. Note that if the original dictionary
    has duplicate values, only the last key for each value will be preserved.

    :param data: Dictionary to be reversed
    :type data: dict
    :return: Dictionary with keys and values swapped
    :rtype: dict
    :raises ValueError: If the dictionary has duplicate values (which would cause key conflicts)

    Example:
        >>> reverse_dict({'a': 1, 'b': 2, 'c': 3})
        {1: 'a', 2: 'b', 3: 'c'}
        >>> reverse_dict({'name': 'John', 'age': 30})
        {'John': 'name', 30: 'age'}
    """
    return {v: k for k, v in data.items()}


def cumsum(lst: list) -> list:
    """
    Calculate the cumulative sum of a list of numbers.

    Returns a list where each element is the sum of all elements up to
    and including that position in the original list.

    :param lst: List of numbers to calculate cumulative sum for
    :type lst: list[Union[int, float]]
    :return: List of cumulative sums
    :rtype: list[Union[int, float]]
    :raises TypeError: If the list contains non-numeric values

    Example:
        >>> cumsum([1, 2, 3, 4])
        [1, 3, 6, 10]
        >>> cumsum([2, 4, 6])
        [2, 6, 12]
    """
    return [sum(lst[: i + 1]) for i in range(len(lst))]


def lag_list(lst: list, lag: int = 1) -> list:
    """
    Create a lagged version of a list.

    Shifts the list values backward by the specified lag amount, filling
    the beginning with None values. This is useful for time series analysis
    where you need to compare current values with previous values.

    :param lst: List to be lagged
    :type lst: list
    :param lag: Number of positions to lag (default: 1)
    :type lag: int
    :return: List with values shifted backward by lag positions
    :rtype: list
    :raises ValueError: If lag is negative or greater than list length

    Example:
        >>> lag_list([1, 2, 3, 4], lag=1)
        [None, 1, 2, 3]
        >>> lag_list(['a', 'b', 'c'], lag=2)
        [None, None, 'a']
    """
    return [None] * lag + lst[:-lag]


def lead_list(lst: list, lead: int = 1) -> list:
    """
    Create a lead version of a list.

    Shifts the list values forward by the specified lead amount, filling
    the end with None values. This is useful for time series analysis
    where you need to compare current values with future values.

    :param lst: List to be led
    :type lst: list
    :param lead: Number of positions to lead (default: 1)
    :type lead: int
    :return: List with values shifted forward by lead positions
    :rtype: list
    :raises ValueError: If lead is negative or greater than list length

    Example:
        >>> lead_list([1, 2, 3, 4], lead=1)
        [2, 3, 4, None]
        >>> lead_list(['a', 'b', 'c'], lead=2)
        ['c', None, None]
    """
    return lst[lead:] + [None] * lead


def row_number(lst: list, start: int = 0) -> list:
    """
    Generate row numbers for list elements.

    Returns a list of sequential numbers corresponding to the position
    of each element in the input list, starting from the specified value.

    :param lst: List to generate row numbers for
    :type lst: list
    :param start: Starting number for row numbering (default: 0)
    :type start: int
    :return: List of row numbers
    :rtype: list[int]

    Example:
        >>> row_number(['a', 'b', 'c'])
        [0, 1, 2]
        >>> row_number(['x', 'y'], start=1)
        [1, 2]
    """
    return [i + start for i, _ in enumerate(lst)]


def flatten(lst: Iterable) -> list:
    """
    Flatten a nested iterable structure into a single list.

    Recursively flattens nested lists, tuples, and other iterables into
    a single flat list. Uses the as_list function to handle different
    iterable types consistently.

    :param lst: Nested iterable structure to flatten
    :type lst: Iterable
    :return: Flattened list containing all elements
    :rtype: list

    Example:
        >>> flatten([[1, 2], [3, [4, 5]]])
        [1, 2, 3, 4, 5]
        >>> flatten([(1, 2), [3, 4]])
        [1, 2, 3, 4]
    """
    return [item for sublist in lst for item in as_list(sublist)]


def df_to_list(df: pd.DataFrame) -> list:
    """
    Convert a pandas DataFrame to a list of dictionaries.

    Transforms each row of the DataFrame into a dictionary where column
    names are keys and row values are values. This is useful for JSON
    serialization or when working with list-based data structures.

    :param df: DataFrame to convert
    :type df: pandas.DataFrame
    :return: List of dictionaries, one per row
    :rtype: list[dict]
    :raises ImportError: If pandas is not installed

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df_to_list(df)
        [{'A': 1, 'B': 3}, {'A': 2, 'B': 4}]
    """
    return df.to_dict(orient="records")


def df_to_dict(df: pd.DataFrame) -> dict:
    """
    Convert a dictionary of DataFrames to a dictionary of record lists.

    Transforms each DataFrame in the input dictionary into a list of
    dictionaries (records format). This is useful for JSON serialization
    of multiple DataFrames or when working with nested data structures.

    :param df: Dictionary of DataFrames to convert
    :type df: dict[str, pandas.DataFrame]
    :return: Dictionary with sheet names as keys and record lists as values
    :rtype: dict[str, list[dict]]
    :raises ImportError: If pandas is not installed

    Example:
        >>> import pandas as pd
        >>> dfs = {
        ...     'sheet1': pd.DataFrame({'A': [1, 2]}),
        ...     'sheet2': pd.DataFrame({'B': [3, 4]})
        ... }
        >>> df_to_dict(dfs)
        {
            'sheet1': [{'A': 1}, {'A': 2}],
            'sheet2': [{'B': 3}, {'B': 4}]
        }
    """
    return {name: content.to_dict(orient="records") for name, content in df.items()}


def as_list(x):
    """
    Convert an object to a list without nesting or string iteration.

    Intelligently converts various object types to lists:
    - Scalars and strings become single-element lists
    - Iterables (except strings and dicts) become lists
    - Prevents unwanted string character iteration

    :param x: Object to convert to list
    :type x: Any
    :return: List representation of the input object
    :rtype: list

    Example:
        >>> as_list(1)
        [1]
        >>> as_list("hello")
        ["hello"]
        >>> as_list([1, 2, 3])
        [1, 2, 3]
        >>> as_list((1, 2, 3))
        [1, 2, 3]
        >>> as_list({1, 2, 3})
        [1, 2, 3]
    """
    if isinstance(x, Iterable) and not isinstance(x, str) and not isinstance(x, dict):
        return list(x)
    else:
        return [x]


def first(lst):
    """
    Get the first element of a list safely.

    Returns the first element of the list, or None if the list is empty.
    This prevents IndexError exceptions when working with potentially
    empty lists.

    :param lst: List to get the first element from
    :type lst: list
    :return: First element of the list, or None if empty
    :rtype: Any

    Example:
        >>> first([1, 2, 3])
        1
        >>> first(['a', 'b', 'c'])
        'a'
        >>> first([])
        None
    """
    if not len(lst):
        return None
    return lst[0]


def last(lst):
    """
    Get the last element of a list safely.

    Returns the last element of the list, or None if the list is empty.
    This prevents IndexError exceptions when working with potentially
    empty lists.

    :param lst: List to get the last element from
    :type lst: list
    :return: Last element of the list, or None if empty
    :rtype: Any

    Example:
        >>> last([1, 2, 3])
        3
        >>> last(['a', 'b', 'c'])
        'c'
        >>> last([])
        None
    """
    if not len(lst):
        return None
    return lst[-1]
