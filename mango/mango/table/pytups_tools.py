from typing import Callable

import numpy as np
from mango.processing import as_list, flatten, reverse_dict
from mango.table.table_tools import str_key, to_len, invert_dict_list
from pytups import TupList


def mutate(table, **kwargs):
    """
    Add or modify columns in a table.

    Creates new columns or modifies existing ones using various methods:
    - Single values applied to all rows
    - Lists of values for each row
    - Functions that operate on row data

    Note: All changes are applied in the order of the arguments.

    :param table: Table to modify
    :type table: TupList
    :param kwargs: Named arguments with column names and their values
    :type kwargs: dict
    :return: New table with modified columns
    :rtype: TupList
    :raises TypeError: If argument format is unexpected

    Example:
        >>> table = TupList([{'a':2, 'b':3}, {'a':2, 'b':6}, {'a':2, 'b':8}])
        >>> result = mutate(table, a=3, b=[4,5,6], c=lambda v: v["a"]+v["b"])
        >>> print(result)
        [{'a': 3, 'b': 4, 'c': 7}, {'a': 3, 'b': 5, 'c': 8}, {'a': 3, 'b': 6, 'c': 9}]
    """
    assert isinstance(table, TupList)

    if len(table) == 0:
        return table

    # Copy deep of table has been removed.
    table2 = table

    # Transform TupList in list of dict
    if isinstance(table2[0], tuple):
        table2 = table2.to_dictlist([i for i in range(len(table2[0]))])

    # Update table
    nrow = table2.len()
    for k, v in kwargs.items():
        if isinstance(v, Callable):
            table2 = [{**row, **{k: v(row)}} for row in table2]
        elif v is None or len(as_list(v)) == 1:
            table2 = [{**row, **{k: v}} for row in table2]
        elif len(as_list(v)) == nrow:
            table2 = [{**row, **{k: v[i]}} for i, row in enumerate(table2)]
        else:
            raise TypeError(f"Unexpected argument to mutate {v}")

    return TupList(table2)


def sum_all(table, group_by=None):
    """
    Group by specified columns and sum all numeric columns.

    Groups the table by the specified columns and sums all numeric
    columns in each group. Non-numeric columns are ignored.

    :param table: Table to process
    :type table: TupList
    :param group_by: Column name(s) to group by
    :type group_by: str or list[str], optional
    :return: New table with grouped and summed data
    :rtype: TupList

    Example:
        >>> table = TupList([
        ...     {'a':2, 'b':3, "val":3},
        ...     {'a':3, 'b':6, "val":6},
        ...     {'a':3, 'b':6, "val":5}
        ... ])
        >>> result = sum_all(table, group_by=["a", "b"])
        >>> print(result)
        [{'a': 2, 'b': 3, 'val': 3}, {'a': 3, 'b': 6, 'val': 11}]
    """
    assert isinstance(table, TupList)
    if len(table) == 0:
        return table
    if group_by is None:
        group_by = []

    return (
        table.to_dict(indices=group_by, result_col=None, is_list=True)
        .vapply(lambda v: invert_dict_list(v, unique=False))
        .vapply(lambda v: {k: sum(v[k]) if k not in group_by else v[k][0] for k in v})
        .values_tl()
    )


def group_by(table, col):
    """
    Group rows of the table by specified column values.

    Groups the table rows based on the values in the specified column(s).
    Returns a SuperDict where keys are the unique values and values are
    lists of rows that have that value.

    :param table: Table to group
    :type table: TupList
    :param col: Column name or list of column names to group by
    :type col: str or list[str]
    :return: SuperDict with grouped data
    :rtype: SuperDict

    Example:
        >>> table = TupList([
        ...     {"name": "Alice", "city": "Madrid"},
        ...     {"name": "Bob", "city": "Barcelona"},
        ...     {"name": "Charlie", "city": "Madrid"}
        ... ])
        >>> result = group_by(table, "city")
        >>> print(result)
        {'Madrid': [{'name': 'Alice', 'city': 'Madrid'}, {'name': 'Charlie', 'city': 'Madrid'}],
         'Barcelona': [{'name': 'Bob', 'city': 'Barcelona'}]}
    """
    assert isinstance(table, TupList)
    return table.to_dict(indices=col, result_col=None, is_list=True)


def summarise(table, group_by, default: [None, Callable] = None, **func):
    """
    Group by specified columns and apply aggregation functions.

    Groups the table by specified columns and applies custom aggregation
    functions to other columns. More flexible than group_mutate as it
    allows specifying a default function for non-explicitly handled columns.

    :param table: Table to process
    :type table: TupList
    :param group_by: Column name(s) to group by
    :type group_by: str or list[str]
    :param default: Default function to apply to columns not explicitly specified
    :type default: Callable, optional
    :param func: Functions to apply to specific columns
    :type func: dict[str, Callable]
    :return: New table with grouped and summarized data
    :rtype: TupList

    Example:
        >>> table = TupList([
        ...     {'a':2, 'b':3, "c":3},
        ...     {'a':3, 'b':6, "c":6},
        ...     {'a':3, 'b':6, "c":5}
        ... ])
        >>> result = summarise(table, "a", b=sum, c=len)
        >>> print(result)
        [{'a': 2, 'b': 3, 'c': 1}, {'a': 3, 'b': 12, 'c': 2}]
    """
    assert isinstance(table, TupList)

    if len(table) == 0:
        return table
    if group_by is None:
        group_by = []
    if default is not None:
        apply_func = {k: default for k in table[0] if k not in group_by}
    else:
        apply_func = {}
    apply_func.update(dict(**func))
    return (
        table.to_dict(indices=group_by, result_col=None, is_list=True)
        .vapply(lambda v: invert_dict_list(v, unique=False))
        .vapply(lambda v: {k: v[k] for k in v if k in group_by or k in apply_func})
        .vapply(
            lambda v: {
                k: apply_func[k](v[k]) if k not in group_by else v[k][0] for k in v
            }
        )
        .values_tl()
    )


def group_mutate(table, group_by, **func):
    """
    Group by specified columns and apply functions to other columns.

    Groups the table by the specified columns and applies aggregation
    functions to the remaining columns. Similar to SQL GROUP BY with
    aggregate functions. Equivalent to group_by %>% mutate %>% ungroup in R dplyr.

    :param table: Table to process
    :type table: TupList
    :param group_by: Column name(s) to group by
    :type group_by: str or list[str]
    :param func: Functions to apply to columns (e.g., a=sum, b=mean)
    :type func: dict[str, Callable]
    :return: New table with grouped and aggregated data
    :rtype: TupList

    Example:
        >>> table = TupList([
        ...     {'a': 2, 'b': 3, "c": 3},
        ...     {'a': 3, 'b': 6, "c": 6},
        ...     {'a': 3, 'b': 6, "c": 5}
        ... ])
        >>> result = group_mutate(table, "a", sum_b=lambda v: sum(v["b"]))
        >>> print(result)
        [{'a': 2, 'b': 3, 'c': 3, 'sum_b': 3},
         {'a': 3, 'b': 6, 'c': 6, 'sum_b': 12},
         {'a': 3, 'b': 6, 'c': 5, 'sum_b': 12}]
    """
    assert isinstance(table, TupList)
    grouped = (
        table.to_dict(indices=group_by, result_col=None, is_list=True)
        .vapply(lambda v: invert_dict_list(v, unique=False))
        .values_tl()
    )
    mutated = mutate(grouped, **func).vapply(lambda v: to_dictlist(v))
    return TupList(flatten(mutated))


def select(table, *args):
    """
    Select specific columns from a table.

    Creates a new table containing only the specified columns.
    Maintains the original row order.

    :param table: Table to select columns from
    :type table: TupList
    :param args: Names of columns to select
    :type args: str
    :return: New table with only the selected columns
    :rtype: TupList
    :raises ValueError: If any specified column doesn't exist

    Example:
        >>> table = TupList([
        ...     {"id": 1, "name": "Alice", "age": 30, "city": "Madrid"},
        ...     {"id": 2, "name": "Bob", "age": 25, "city": "Barcelona"}
        ... ])
        >>> result = select(table, "name", "age")
        >>> print(result)
        [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """
    assert isinstance(table, TupList)

    if not len(table):
        return TupList()

    keep = as_list(args)
    missing = [k for k in keep if k not in get_col_names(table)]
    if len(missing):
        raise ValueError("Column %s not found" % missing)
    return table.vapply(lambda v: {k: v[k] for k in keep})


def drop(table, *args):
    """
    Remove specific columns from a table.

    Creates a new table with the specified columns removed.
    Maintains the original row order.

    :param table: Table to remove columns from
    :type table: TupList
    :param args: Names of columns to remove
    :type args: str
    :return: New table without the specified columns
    :rtype: TupList

    Example:
        >>> table = TupList([
        ...     {"id": 1, "name": "Alice", "age": 30, "city": "Madrid"},
        ...     {"id": 2, "name": "Bob", "age": 25, "city": "Barcelona"}
        ... ])
        >>> result = drop(table, "id", "city")
        >>> print(result)
        [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """
    assert isinstance(table, TupList)

    remove = as_list(args)
    keep = [k for k in get_col_names(table) if k not in remove]
    return table.vapply(lambda v: {k: v[k] for k in keep if k in v})


def rename(table, **kwargs):
    """
    Rename columns in a table.

    Changes column names using a mapping of old names to new names.
    Maintains the original row order and data.

    :param table: Table to rename columns in
    :type table: TupList
    :param kwargs: Mapping of old column names to new names
    :type kwargs: dict[str, str]
    :return: New table with renamed columns
    :rtype: TupList

    Example:
        >>> table = TupList([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        >>> result = rename(table, id="user_id", name="full_name")
        >>> print(result)
        [{'user_id': 1, 'full_name': 'Alice'}, {'user_id': 2, 'full_name': 'Bob'}]
    """
    assert isinstance(table, TupList)

    new_names = dict(**kwargs)
    return table.vapply(
        lambda v: {new_names[k] if k in new_names else k: v[k] for k in v}
    )


def get_col_names(table, fast=False):
    """
    Get the names of all columns in a table.

    Returns a list of column names. By default, scans all rows to ensure
    all possible columns are included. Use fast=True for better performance
    if you're certain the first row contains all columns.

    :param table: Table to get column names from
    :type table: TupList
    :param fast: If True, only check the first row for column names
    :type fast: bool
    :return: List of column names
    :rtype: list[str]
    :raises IndexError: If table is empty

    Example:
        >>> table = TupList([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
        >>> columns = get_col_names(table)
        >>> print(columns)
        ['name', 'age']
    """
    assert isinstance(table, TupList)

    if len(table) < 1:
        return []
    if fast:
        return [k for k in table[0].keys()]
    else:
        columns = []
        for row in table:
            columns += [k for k in row.keys() if k not in columns]
        return columns


def left_join(
    table1,
    table2,
    by=None,
    suffix=None,
    empty=None,
    if_empty_table_1=None,
    if_empty_table_2=None,
):
    """
    Perform a left join with another table.

    Returns all rows from the left table (table1) and matching rows from
    the right table (table2). Rows from the left table without matches
    will have None values for columns from the right table.
    Shortcut to join(type="left"). Inspired by R dplyr join functions.

    :param table1: First table to join
    :type table1: TupList
    :param table2: Second table to join with
    :type table2: TupList or list[dict]
    :param by: Column specification for joining
    :type by: list, dict, or None
    :param suffix: Suffixes for disambiguating column names
    :type suffix: list[str], optional
    :param empty: Value to use for empty cells created by the join
    :param if_empty_table_1: Replacement if table 1 is empty
    :param if_empty_table_2: Replacement if table 2 is empty
    :return: New table containing the left join result
    :rtype: TupList

    Example:
        >>> table1 = TupList([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        >>> table2 = TupList([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
        >>> result = left_join(table1, table2, by="id")
        >>> print(result)
        [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': None}]
    """
    return join(
        table1,
        table2,
        by=by,
        suffix=suffix,
        jtype="left",
        empty=empty,
        if_empty_table_1=if_empty_table_1,
        if_empty_table_2=if_empty_table_2,
    )


def right_join(
    table1,
    table2,
    by=None,
    suffix=None,
    empty=None,
    if_empty_table_1=None,
    if_empty_table_2=None,
):
    """
    Perform a right join with another table.

    Returns all rows from the right table (table2) and matching rows from
    the left table (table1). Rows from the right table without matches
    will have None values for columns from the left table.
    Shortcut to join(type="right"). Inspired by R dplyr join functions.

    :param table1: First table to join
    :type table1: TupList
    :param table2: Second table to join with
    :type table2: TupList or list[dict]
    :param by: Column specification for joining
    :type by: list, dict, or None
    :param suffix: Suffixes for disambiguating column names
    :type suffix: list[str], optional
    :param empty: Value to use for empty cells created by the join
    :param if_empty_table_1: Replacement if table 1 is empty
    :param if_empty_table_2: Replacement if table 2 is empty
    :return: New table containing the right join result
    :rtype: TupList

    Example:
        >>> table1 = TupList([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        >>> table2 = TupList([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
        >>> result = right_join(table1, table2, by="id")
        >>> print(result)
        [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 3, 'name': None, 'age': 25}]
    """
    return join(
        table1,
        table2,
        by=by,
        suffix=suffix,
        jtype="right",
        empty=empty,
        if_empty_table_1=if_empty_table_1,
        if_empty_table_2=if_empty_table_2,
    )


def full_join(
    table1,
    table2,
    by=None,
    suffix=None,
    empty=None,
    if_empty_table_1=None,
    if_empty_table_2=None,
):
    """
    Perform a full outer join with another table.

    Returns all rows from both tables, with None values where there are
    no matches. This is the default join type and combines left and right joins.
    Shortcut to join(type="full"). Inspired by R dplyr join functions.

    :param table1: First table to join
    :type table1: TupList
    :param table2: Second table to join with
    :type table2: TupList or list[dict]
    :param by: Column specification for joining
    :type by: list, dict, or None
    :param suffix: Suffixes for disambiguating column names
    :type suffix: list[str], optional
    :param empty: Value to use for empty cells created by the join
    :param if_empty_table_1: Replacement if table 1 is empty
    :param if_empty_table_2: Replacement if table 2 is empty
    :return: New table containing the full join result
    :rtype: TupList

    Example:
        >>> table1 = TupList([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        >>> table2 = TupList([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
        >>> result = full_join(table1, table2, by="id")
        >>> print(result)
        [{'id': 1, 'name': 'Alice', 'age': 30},
         {'id': 2, 'name': 'Bob', 'age': None},
         {'id': 3, 'name': None, 'age': 25}]
    """
    return join(
        table1,
        table2,
        by=by,
        suffix=suffix,
        jtype="full",
        empty=empty,
        if_empty_table_1=if_empty_table_1,
        if_empty_table_2=if_empty_table_2,
    )


def inner_join(
    table1,
    table2,
    by=None,
    suffix=None,
    empty=None,
    if_empty_table_1=None,
    if_empty_table_2=None,
):
    """
    Perform an inner join with another table.

    Returns only rows that have matching values in both tables.
    Rows without matches in either table are excluded from the result.
    Shortcut to join(type="inner"). Inspired by R dplyr join functions.

    :param table1: First table to join
    :type table1: TupList
    :param table2: Second table to join with
    :type table2: TupList or list[dict]
    :param by: Column specification for joining
    :type by: list, dict, or None
    :param suffix: Suffixes for disambiguating column names
    :type suffix: list[str], optional
    :param empty: Value to use for empty cells created by the join
    :param if_empty_table_1: Replacement if table 1 is empty
    :param if_empty_table_2: Replacement if table 2 is empty
    :return: New table containing only matching rows
    :rtype: TupList

    Example:
        >>> table1 = TupList([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        >>> table2 = TupList([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
        >>> result = inner_join(table1, table2, by="id")
        >>> print(result)
        [{'id': 1, 'name': 'Alice', 'age': 30}]
    """
    return join(
        table1,
        table2,
        by=by,
        suffix=suffix,
        jtype="inner",
        empty=empty,
        if_empty_table_1=if_empty_table_1,
        if_empty_table_2=if_empty_table_2,
    )


def get_join_keys(tab1, tab2, jtype):
    """
    Get the keys to use for the join depending on the join type.

    Determines which join keys to use based on the join type and the
    available keys in both tables.

    :param tab1: Table 1 grouped by join keys
    :type tab1: dict
    :param tab2: Table 2 grouped by join keys
    :type tab2: dict
    :param jtype: Join type
    :type jtype: str
    :return: TupList of unique join key combinations
    :rtype: TupList
    :raises ValueError: If jtype is not one of the supported types

    Example:
        >>> tab1 = {(1, 2): [{"a": 1, "b": 2}], (3, 4): [{"a": 3, "b": 4}]}
        >>> tab2 = {(1, 2): [{"c": 1}], (5, 6): [{"c": 2}]}
        >>> keys = get_join_keys(tab1, tab2, "inner")
        >>> print(keys)
        [(1, 2)]
    """
    if jtype == "full":
        join_keys = [k for k in tab1.keys()] + [k for k in tab2.keys()]
    elif jtype == "left":
        join_keys = [k for k in tab1.keys()]
    elif jtype == "right":
        join_keys = [k for k in tab2.keys()]
    elif jtype == "inner":
        join_keys = [k for k in tab1.keys() if k in tab2.keys()]
    else:
        raise ValueError("jtype must be full, inner, right or left")

    result = (
        TupList(join_keys)
        .vfilter(lambda v: all(i is not None for i in as_list(v)))
        .unique2()
        .sorted()
    )
    return result


def manage_join_none(tab1, tab2, empty, t1_keys, t2_keys, by, jtype):
    """
    Handle None values in join operations.

    None values should never join with other None values.
    Depending on the join type, returns the relevant rows with None values in keys.

    :param tab1: Table 1 grouped by join keys
    :type tab1: SuperDict
    :param tab2: Table 2 grouped by join keys
    :type tab2: SuperDict
    :param empty: Value to use for missing values
    :param t1_keys: Column names of table 1
    :type t1_keys: list[str]
    :param t2_keys: Column names of table 2
    :type t2_keys: list[str]
    :param by: Keys to join by
    :type by: list[str]
    :param jtype: Join type (left, right, full, inner)
    :type jtype: str
    :return: TupList of rows joined on None values
    :rtype: TupList
    :raises ValueError: If jtype is not supported

    Example:
        >>> tab1 = {None: [{"a": None, "b": 1}]}
        >>> tab2 = {None: [{"a": None, "c": 1}]}
        >>> result = manage_join_none(tab1, tab2, None, ["a", "b"], ["a", "c"], ["a"], "left")
        >>> print(result)
        [{'a': None, 'b': 1, 'c': None}]
    """
    result = []
    if jtype == "left":
        # if left join, any None value in join keys result in empty values in the second table columns.
        for i in tab1.keys():
            if any(v is None for v in as_list(i)):
                tab2[i] = [{k: empty for k in t2_keys if k not in by}]
                result += [{**d1, **d2} for d1 in tab1[i] for d2 in tab2[i]]
    elif jtype == "right":
        # a right join is a left join with table in other order.
        result = manage_join_none(tab2, tab1, empty, t2_keys, t1_keys, by, jtype="left")
    elif jtype == "full":
        # a full join is the combination of left and right join.
        result = manage_join_none(
            tab1, tab2, empty, t1_keys, t2_keys, by, jtype="left"
        ) + manage_join_none(tab1, tab2, empty, t1_keys, t2_keys, by, jtype="right")
    elif jtype == "inner":
        #  if inner join, only existing values from both tables are kept.
        return TupList()
    else:
        raise ValueError("jtype must be full, inner, right or left")

    return result


def join(
    table1,
    table2,
    by=None,
    suffix=None,
    jtype="full",
    empty=None,
    drop_if_nested=False,
    if_empty_table_1=None,
    if_empty_table_2=None,
) -> TupList:
    """
    Join two tables using various join types.

    Performs table joins inspired by R's dplyr join functions. Supports
    different join types and flexible column matching strategies.

    :param table1: First table to join
    :type table1: TupList
    :param table2: Second table to join with
    :type table2: TupList or list[dict]
    :param by: Column specification for joining
    :type by: list, dict, or None
    :param suffix: Suffixes for disambiguating column names
    :type suffix: list[str], optional
    :param jtype: Type of join ("full", "left", "right", "inner")
    :type jtype: str
    :param empty: Value to use for empty cells created by the join
    :param drop_if_nested: Drop nested columns (may cause errors if False)
    :type drop_if_nested: bool
    :param if_empty_table_1: Replacement if table 1 is empty
    :param if_empty_table_2: Replacement if table 2 is empty
    :return: New table containing the joined data
    :rtype: TupList
    :raises ValueError: If jtype is not supported or keys don't exist

    Example:
        >>> table1 = TupList([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        >>> table2 = TupList([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
        >>> result = join(table1, table2, by="id", jtype="left")
        >>> print(result)
        [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': None}]
    """
    assert isinstance(table1, TupList)
    assert isinstance(table2, list)
    table2 = TupList(table2)

    join_types = ["full", "left", "right", "inner"]
    if jtype not in join_types:
        raise ValueError("jtype must be one of those: %s" % join_types)

    # Default suffix
    if suffix is None:
        suffix = ["", "_2"]

    # If a table is empty return the other one
    if len(table2) == 0:
        if jtype in ["inner", "right"]:
            return TupList()
        elif if_empty_table_2 is None:
            return table1
        else:
            table2 = TupList(as_list(if_empty_table_2))
    if len(table1) == 0:
        if jtype in ["inner", "left"]:
            return TupList()
        elif if_empty_table_1 is None:
            return table2
        else:
            table1 = TupList(as_list(if_empty_table_1))

    if drop_if_nested:
        table1 = drop_nested(table1)
        table2 = drop_nested(table2)

    t1_keys = get_col_names(table1)
    t2_keys = get_col_names(table2)
    shared_keys = set(t1_keys).intersection(set(t2_keys))
    names_table2 = {}
    if by is None:
        by = shared_keys
    elif isinstance(by, dict):
        names_table2 = reverse_dict(by)
        by = [k for k in by.keys()]
        shared_keys = set(list(shared_keys) + by)
    else:
        by = as_list(by)

    if set(by) - shared_keys:
        raise ValueError(
            f"Some keys in {by} do not exist in the tables. Shared keys are {shared_keys}"
        )

    # Add suffixes if some shared keys are not in by
    shared_but_not_by = set(by) ^ shared_keys
    table1 = rename(
        table1, **{k: str(k) + suffix[0] for k in t1_keys if k in shared_but_not_by}
    )
    table2 = rename(
        table2,
        **{
            k: str(k) + suffix[1]
            for k in t2_keys
            if (k in shared_but_not_by or k in names_table2.values())
            and k not in names_table2.keys()
        },
    )
    # If by was a dict, rename table2 columns to coincides with table1
    table2 = rename(table2, **{k: v for k, v in names_table2.items()})
    t1_keys = get_col_names(table1)
    t2_keys = get_col_names(table2)

    # Do the join
    tab1 = group_by(table1, by)
    tab2 = group_by(table2, by)

    # select the keys of the join
    join_keys = get_join_keys(tab1, tab2, jtype)
    result = []

    for i in join_keys:
        if i not in tab1:
            tab1[i] = [{k: empty for k in t1_keys if k not in by}]
        if i not in tab2:
            tab2[i] = [{k: empty for k in t2_keys if k not in by}]
        result += [{**d1, **d2} for d1 in tab1[i] for d2 in tab2[i]]

    return TupList(result) + manage_join_none(
        tab1, tab2, empty, t1_keys, t2_keys, by, jtype
    )


def auto_join(table, by=None, suffix=None, empty=None):
    """
    Join a table with itself to create combinations.

    Performs a self-join to create all possible combinations of rows.
    Useful for creating Cartesian products or finding relationships
    within the same table.

    :param table: Table to perform self-join on
    :type table: TupList
    :param by: Column specification for the self-join
    :type by: list, dict, or None
    :param suffix: Suffixes to add to column names to distinguish them
    :type suffix: list[str], optional
    :param empty: Value to use for empty cells created by the join
    :return: New table with all combinations
    :rtype: TupList

    Example:
        >>> table = TupList([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        >>> result = auto_join(table)
        >>> print(result)
        [{'id': 1, 'name': 'Alice', 'id_2': 1, 'name_2': 'Alice'},
         {'id': 1, 'name': 'Alice', 'id_2': 2, 'name_2': 'Bob'},
         {'id': 2, 'name': 'Bob', 'id_2': 1, 'name_2': 'Alice'},
         {'id': 2, 'name': 'Bob', 'id_2': 2, 'name_2': 'Bob'}]
    """
    drop_d = False
    if by is None:
        table = mutate(table, _=1)
        by = "_"
        drop_d = True
    table2 = table
    result = full_join(table, table2, by=by, suffix=suffix, empty=empty)
    if drop_d:
        result = drop(result, by)
    return result


def str_key_tl(tl):
    """
    Transform all dictionary keys in a TupList to strings.

    Converts all dictionary keys in the TupList to string format.
    Useful for JSON serialization or when working with mixed key types.

    :param tl: TupList containing dictionaries
    :type tl: TupList
    :return: Same TupList with all keys converted to strings
    :rtype: TupList

    Example:
        >>> tl = TupList([{1: "one", 2: "two"}, {3: "three"}])
        >>> result = str_key_tl(tl)
        >>> print(result)
        [{'1': 'one', '2': 'two'}, {'3': 'three'}]
    """
    return TupList([str_key(dic) for dic in tl])


def replace(tl, replacement=None, to_replace=None, fast=False):
    """
    Replace specific values in a TupList.

    Replaces specified values with new values in the TupList. Can replace
    values across all columns or target specific columns.

    :param tl: TupList to process
    :type tl: TupList
    :param replacement: New values to use as replacements
    :type replacement: any or dict[str, any]
    :param to_replace: Values to be replaced
    :type to_replace: any or dict[str, any]
    :param fast: If True, assume first row contains all columns
    :type fast: bool
    :return: New TupList with replaced values
    :rtype: TupList

    Example:
        >>> tl = TupList([
        ...     {"age": 25, "city": "Madrid"},
        ...     {"age": 30, "city": "Barcelona"}
        ... ])
        >>> result = replace(
        ...     tl,
        ...     replacement={"age": 35, "city": "Paris"},
        ...     to_replace={"age": 25, "city": "Madrid"}
        ... )
        >>> print(result)
        [{'age': 35, 'city': 'Paris'}, {'age': 30, 'city': 'Barcelona'}]
    """
    apply_to_col = []
    if isinstance(replacement, dict):
        apply_to_col += [i for i in replacement.keys()]
    else:
        replacement = {k: replacement for k in get_col_names(tl, fast)}
    if isinstance(to_replace, dict):
        apply_to_col += [i for i in to_replace.keys()]
        to_replace_dict = to_replace
    else:
        to_replace_dict = {k: to_replace for k in get_col_names(tl, fast)}
    if not len(apply_to_col):
        apply_to_col = get_col_names(tl, fast)

    return TupList(
        [
            {
                **replacement,
                **{
                    k: v
                    for k, v in dic.items()
                    if k not in apply_to_col or v != to_replace_dict[k]
                },
            }
            for dic in tl
        ]
    )


def replace_empty(tl, replacement=0, fast=False):
    """
    Replace empty values in a TupList.

    Replaces empty values (None, empty strings, etc.) with specified
    replacement values. Can use different replacements for different columns.

    :param tl: TupList to process
    :type tl: TupList
    :param replacement: Values to use for replacing empty values
    :type replacement: any or dict[str, any]
    :param fast: If True, assume first row contains all columns
    :type fast: bool
    :return: New TupList with empty values replaced
    :rtype: TupList

    Example:
        >>> tl = TupList([
        ...     {"name": "Alice", "age": None},
        ...     {"name": "", "age": 25}
        ... ])
        >>> result = replace_empty(tl, replacement={"name": "Unknown", "age": 0})
        >>> print(result)
        [{'name': 'Alice', 'age': 0}, {'name': 'Unknown', 'age': 25}]
    """
    return replace(tl, replacement=replacement, to_replace=None, fast=fast)


def replace_nan(tl, replacement=None):
    """
    Replace NaN values in a TupList.

    Replaces NaN (Not a Number) values with specified replacement values.
    Useful for cleaning numeric data with missing values.

    :param tl: TupList to process
    :type tl: TupList
    :param replacement: Value to use for replacing NaN values
    :type replacement: any
    :return: New TupList with NaN values replaced
    :rtype: TupList

    Example:
        >>> import math
        >>> tl = TupList([
        ...     {"value": 10.5, "score": math.nan},
        ...     {"value": math.nan, "score": 85.0}
        ... ])
        >>> result = replace_nan(tl, replacement=0)
        >>> print(result)
        [{'value': 10.5, 'score': 0}, {'value': 0, 'score': 85.0}]
    """
    return TupList(
        [{k: replacement if is_null(v) else v for k, v in dic.items()} for dic in tl]
    )


def drop_empty(tl, cols=None, fast=False):
    """
    Remove rows with empty values in specified columns.

    Drops rows where the specified columns contain empty values
    (None, empty strings, etc.).

    :param tl: TupList to process
    :type tl: TupList
    :param cols: Column name(s) to check for empty values
    :type cols: str or list[str], optional
    :param fast: If True, assume first row contains all columns
    :type fast: bool
    :return: New TupList with empty rows removed
    :rtype: TupList

    Example:
        >>> tl = TupList([
        ...     {"name": "Alice", "age": 30},
        ...     {"name": "", "age": 25},
        ...     {"name": "Bob", "age": None}
        ... ])
        >>> result = drop_empty(tl, "name")
        >>> print(result)
        [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': None}]
    """
    if cols is None:
        cols = get_col_names(tl, fast)
    else:
        cols = as_list(cols)
    tl2 = replace(tl, replacement=None, to_replace=None)

    return tl2.vfilter(lambda v: not any(k not in v or is_null(v[k]) for k in cols))


def is_null(v):
    """
    Check if a value is null (None, NaN, or NaT).

    Determines whether a value represents a null/missing value.
    Handles both scalar values and lists recursively.

    :param v: Value to check
    :type v: any
    :return: True if the value is null
    :rtype: bool or list[bool]

    Example:
        >>> import math
        >>> print(is_null(None))
        True
        >>> print(is_null(math.nan))
        True
        >>> print(is_null(42))
        False
        >>> print(is_null([None, 1, math.nan]))
        [True, False, True]
    """
    if isinstance(v, list):
        return [is_null(i) for i in v]
    return v is None or is_nan(v) or is_nat(v)


def is_nan(v):
    """
    Check if a value is NaN (Not a Number).

    Similar to np.isnan but returns False instead of raising an error
    if the value is not a number. Handles both scalar values and lists recursively.

    :param v: Value to check
    :type v: any
    :return: True if the value is NaN
    :rtype: bool or list[bool]

    Example:
        >>> import math
        >>> print(is_nan(math.nan))
        True
        >>> print(is_nan(42))
        False
        >>> print(is_nan("text"))
        False
        >>> print(is_nan([1, math.nan, "text"]))
        [False, True, False]
    """
    from numbers import Number

    if isinstance(v, list):
        return [is_nan(i) for i in v]
    elif isinstance(v, Number):
        return np.isnan(v)
    else:
        return False


def is_nat(v):
    """
    Check if a value is NaT (Not a Time).

    Similar to np.isnat but returns False instead of raising an error
    if the value is not a datetime. Handles both scalar values and lists recursively.

    :param v: Value to check
    :type v: any
    :return: True if the value is NaT
    :rtype: bool or list[bool]

    Example:
        >>> import numpy as np
        >>> nat_value = np.datetime64('NaT')
        >>> print(is_nat(nat_value))
        True
        >>> print(is_nat(np.datetime64('2023-01-01')))
        False
        >>> print(is_nat("2023-01-01"))
        False
    """
    if isinstance(v, list):
        return [is_nat(i) for i in v]
    elif isinstance(v, np.datetime64):
        return np.isnat(v)
    else:
        return False


def pivot_longer(tl, cols, names_to="variable", value_to="value"):
    """
    Transform table from wide to long format.

    "Lengthens" data by increasing the number of rows and decreasing
    the number of columns. The inverse transformation of pivot_wider().

    :param tl: TupList to transform
    :type tl: TupList
    :param cols: List of column names to pivot
    :type cols: list[str]
    :param names_to: Name for the new column containing variable names
    :type names_to: str
    :param value_to: Name for the new column containing values
    :type value_to: str
    :return: New TupList in long format
    :rtype: TupList
    :raises AssertionError: If inputs are not of expected types

    Example:
        >>> table = TupList([
        ...     {"id": 1, "Q1": 100, "Q2": 150, "Q3": 200},
        ...     {"id": 2, "Q1": 120, "Q2": 180, "Q3": 220}
        ... ])
        >>> result = pivot_longer(table, ["Q1", "Q2", "Q3"], "quarter", "sales")
        >>> print(result)
        [{'id': 1, 'quarter': 'Q1', 'sales': 100},
         {'id': 1, 'quarter': 'Q2', 'sales': 150},
         {'id': 1, 'quarter': 'Q3', 'sales': 200},
         {'id': 2, 'quarter': 'Q1', 'sales': 120},
         {'id': 2, 'quarter': 'Q2', 'sales': 180},
         {'id': 2, 'quarter': 'Q3', 'sales': 220}]
    """
    assert isinstance(tl, TupList)
    assert isinstance(cols, list)
    assert set(cols).issubset(set(get_col_names(tl)))
    assert isinstance(names_to, str)
    assert isinstance(value_to, str)

    return TupList(
        [
            {
                **{k: d[k] for k in d if k not in cols},
                **{names_to: col, value_to: d[col]},
            }
            for col in cols
            for d in tl
        ]
    )


def pivot_wider(
    tl, names_from="variable", value_from="value", id_cols=None, values_fill=None
):
    """
    Transform table from long to wide format.

    "Widens" data by increasing the number of columns and decreasing
    the number of rows. The inverse transformation of pivot_longer().

    :param tl: TupList to transform
    :type tl: TupList
    :param names_from: Name of the column containing variable names
    :type names_from: str
    :param value_from: Name of the column containing values
    :type value_from: str
    :param id_cols: Columns that uniquely identify each observation
    :type id_cols: list[str], optional
    :param values_fill: Value or dict to fill missing values
    :type values_fill: any or dict, optional
    :return: New TupList in wide format
    :rtype: TupList
    :raises AssertionError: If inputs are not of expected types

    Example:
        >>> tl = TupList([
        ...     {"id": 1, "quarter": "Q1", "sales": 100},
        ...     {"id": 1, "quarter": "Q2", "sales": 150},
        ...     {"id": 2, "quarter": "Q1", "sales": 120}
        ... ])
        >>> result = pivot_wider(tl, "quarter", "sales", "id")
        >>> print(result)
        [{'id': 1, 'Q1': 100, 'Q2': 150}, {'id': 2, 'Q1': 120, 'Q2': None}]
    """
    assert isinstance(tl, TupList)
    assert isinstance(names_from, str)
    assert isinstance(value_from, str)

    if id_cols is None:
        id_cols = [c for c in get_col_names(tl) if c not in [names_from, value_from]]
    tl_sum = summarise(tl, id_cols, default=list)
    tl_result = TupList(
        [
            {
                **{k: d[k] for k in id_cols},
                **{k: v for k, v in zip(d[names_from], d[value_from])},
            }
            for d in tl_sum
        ]
    )
    return replace_empty(tl_result, values_fill)


def to_dictlist(dic):
    """
    Transform a dictionary of lists into a list of dictionaries.

    Converts a column-oriented dictionary (dict of lists) into a
    row-oriented list of dictionaries.

    :param dic: Dictionary with column names as keys and lists of values
    :type dic: dict[str, list]
    :return: TupList with row-oriented data
    :rtype: TupList

    Example:
        >>> data = {"name": ["Alice", "Bob"], "age": [30, 25]}
        >>> result = to_dictlist(data)
        >>> print(result)
        [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """
    n_rows = max(len(as_list(dic[k])) for k in dic)
    dic2 = {k: to_len(dic[k], n_rows) for k in dic}
    return TupList([{k: dic2[k][i] for k in dic2} for i in range(n_rows)])


def lag(table, col, i=1):
    """
    Create a lagged version of a column.

    Returns a list with values from the specified column shifted by
    the specified number of steps. Earlier values are filled with None.

    :param table: TupList to process
    :type table: TupList
    :param col: Name of the column to lag
    :type col: str
    :param i: Number of steps to lag (default: 1)
    :type i: int
    :return: List with lagged values
    :rtype: list

    Example:
        >>> table = TupList([
        ...     {"date": "2023-01", "sales": 100},
        ...     {"date": "2023-02", "sales": 150},
        ...     {"date": "2023-03", "sales": 200}
        ... ])
        >>> result = lag(table, "sales", 1)
        >>> print(result)
        [None, 100, 150]
    """
    return table.kvapply(
        lambda k, v: table[k - i][col] if i <= k < len(table) + i else None
    )


def lag_col(table, col, i=1, replace=False):
    """
    Add a lagged column to a TupList.

    Creates a new column with values from the specified column shifted by
    the specified number of steps. Useful for time series analysis.

    :param table: TupList to process
    :type table: TupList
    :param col: Name of the column to lag
    :type col: str
    :param i: Number of steps to lag (default: 1)
    :type i: int
    :param replace: If True, replace original column; if False, create new column
    :type replace: bool
    :return: New TupList with lagged column
    :rtype: TupList

    Example:
        >>> table = TupList([
        ...     {"date": "2023-01", "sales": 100},
        ...     {"date": "2023-02", "sales": 150},
        ...     {"date": "2023-03", "sales": 200}
        ... ])
        >>> result = lag_col(table, "sales", 1)
        >>> print(result)
        [{'date': '2023-01', 'sales': 100, 'lag_sales_1': None},
         {'date': '2023-02', 'sales': 150, 'lag_sales_1': 100},
         {'date': '2023-03', 'sales': 200, 'lag_sales_1': 150}]
    """
    pre = "lag" if i > 0 else "lead"
    index = str(i) if i > 0 else str(-i)
    new_col = f"{pre}_{str(col)}_{index}" if not replace else col
    return mutate(table, **{new_col: lag(table, col, i)})


def drop_nested(df):
    """
    Remove columns containing nested data structures.

    Drops columns that contain nested values (dictionaries, lists, or tuples).
    Assumes homogeneous table structure and checks only the first row.

    :param df: TupList to process
    :type df: TupList
    :return: New TupList with nested columns removed
    :rtype: TupList

    Example:
        >>> df = TupList([
        ...     {"name": "Alice", "age": 30, "hobbies": ["reading", "swimming"]},
        ...     {"name": "Bob", "age": 25, "hobbies": ["gaming"]}
        ... ])
        >>> result = drop_nested(df)
        >>> print(result)
        [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """
    for col in df[0]:
        if isinstance(df[0][col], list) or isinstance(df[0][col], dict):
            df = drop(df, col)
    return df


def distinct(table, columns):
    """
    Keep only unique combinations of values in specified columns.

    Removes duplicate rows based on the values in the specified columns.
    When duplicates are found, the first occurrence is kept.

    :param table: TupList to process
    :type table: TupList
    :param columns: Column name(s) to check for uniqueness
    :type columns: str or list[str]
    :return: New TupList with duplicate rows removed
    :rtype: TupList

    Example:
        >>> table = TupList([
        ...     {"name": "Alice", "city": "Madrid"},
        ...     {"name": "Bob", "city": "Barcelona"},
        ...     {"name": "Alice", "city": "Madrid"}  # Duplicate
        ... ])
        >>> result = distinct(table, "name")
        >>> print(result)
        [{'name': 'Alice', 'city': 'Madrid'}, {'name': 'Bob', 'city': 'Barcelona'}]
    """
    return (
        TupList(table)
        .to_dict(indices=columns, result_col=None, is_list=True)
        .vapply(lambda v: v[0])
        .values_tl()
    )


def order_by(table, columns, reverse=False):
    """
    Sort the table by specified columns.

    Reorders the table rows based on the values in the specified columns.
    Supports both ascending and descending order.

    :param table: TupList to sort
    :type table: TupList
    :param columns: Column name(s) to sort by
    :type columns: str or list[str]
    :param reverse: If True, sort in descending order
    :type reverse: bool
    :return: New TupList with sorted rows
    :rtype: TupList

    Example:
        >>> table = TupList([
        ...     {"name": "Charlie", "age": 35},
        ...     {"name": "Alice", "age": 30},
        ...     {"name": "Bob", "age": 25}
        ... ])
        >>> result = order_by(table, "age")
        >>> print(result)
        [{'name': 'Bob', 'age': 25},
         {'name': 'Alice', 'age': 30},
         {'name': 'Charlie', 'age': 35}]
    """
    return TupList(table).sorted(
        key=lambda v: [v[c] for c in as_list(columns)], reverse=reverse
    )
