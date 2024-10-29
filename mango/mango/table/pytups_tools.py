from typing import Callable

import numpy as np
from mango.processing import as_list, flatten, reverse_dict
from mango.table.table_tools import str_key, to_len, invert_dict_list
from pytups import TupList


def mutate(table, **kwargs):
    """
    Add or modify a column in a table.

    Example:
    table = TupList([{'a':2, 'b':3}, {'a':2, 'b':6}, {'a':2, 'b':8}])
    result = mutate(table, a=3, b=[4,5,6], c=lambda v: v["a"]+v["b"], d = lambda v: sum(v.values()))
    result: [{'a': 3, 'b': 4, 'c': 7, 'd': 14},
            {'a': 3, 'b': 5, 'c': 8, 'd': 16},
            {'a': 3, 'b': 6, 'c': 9, 'd': 18}]

    Note: all changes are applied in the order of the arguments.

    :param table: TupList
    :param kwargs: named arguments with the changes to apply.
    The values can be:
     - a single value which will be applied to each row. ex: a=3
     - a list with all the values of the column. ex: b=[4,5,6]
     - a function to apply to the row. ex: c=lambda v: v["a"]+v["b"]
    :return: a TupList
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
    Group by the given columns and sum the others.

    Example:
    table = TupList([{'a':2, 'b':3, "val":3}, {'a':3, 'b':6, "val":6}, {'a':3, 'b':6, "val":5}])
    result=sum_all(table, group_by=["a", "b"])
    result: [{'a': 2, 'b': 3, 'val': 3}, {'a': 3, 'b': 6, 'val': 11}]

    :param table: a table (TupList of dict).
    :param group_by: name of the columns to group.
    :return: a table (TupList of dict)
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
    Group the rows of a table by the value of some columns

    :param table: a table (TupList of dict).
    :param col: single name of list of columns to use to group the rows
    :return a SuperDict
    """
    assert isinstance(table, TupList)
    return table.to_dict(indices=col, result_col=None, is_list=True)


def summarise(table, group_by, default: [None, Callable] = None, **func):
    """
    Group by the given columns and apply the given functions to the others.

    Example:
    table = TupList([{'a':2, 'b':3, "c":3}, {'a':3, 'b':6, "c":6}, {'a':3, 'b':6, "c":5}])
    result = summarise(table, "a", b=sum, c=len)
    result: [{'a': 2, 'b': 3, 'c': 1}, {'a': 3, 'b': 12, 'c': 2}]

    :param table: a table (TupList of dict).
    :param group_by: name of the columns to group.
    :param default: default function to apply to non-grouped columns.
    :param func: function to apply to the named column. ex: a = first, b = mean
    :return: a table (TupList of dict).
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
    Group by the given columns and apply the given functions to the others.
    Equivalent to group_by %>% mutate %>% ungroup in R dplyr.
    Can be useful to get a total or a count in a column while keeping all the rows.

    Example:
    table = TupList([{'a': 2, 'b': 3, "c": 3}, {'a': 3, 'b': 6, "c": 6}, {'a': 3, 'b': 6, "c": 5}])
    # For every value of a get the sum of b and count the number of rows.
    result = group_mutate(table, "a", sum_b=lambda v: sum(v["b"]),
     count=lambda v: [1+i for i in range(len(v["a"]))])
    result:
    [{'a': 2, 'b': 3, 'c': 3, 'sum_b': 3, 'count': 1},
     {'a': 3, 'b': 6, 'c': 6, 'sum_b': 12, 'count': 1},
     {'a': 3, 'b': 6, 'c': 5, 'sum_b': 12, 'count': 2}]

    In this example the function are applied to this grouped object:
    grouped: [{'a': [2], 'b': [3], 'c': [3]}, {'a': [3, 3], 'b': [6, 6], 'c': [6, 5]}]

    :param table: a table (TupList of dict).
    :param group_by: name of the columns to group.
    :param func: named arguments with the changes to apply.
    The values can be:
     - a single value which will be applied to each row
     - a list with all the values of the column
     - a function to apply to the row.
    :return: a TupList
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
    Select columns from a table

    :param table: a table
    :param args: names of the columns to select
    :return: a table (TupList) with the selected columns.
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
    Drop columns from a table

    :param table: a table
    :param args: names of the columns to drop
    :return: a table (TupList) without the selected columns.
    """
    assert isinstance(table, TupList)

    remove = as_list(args)
    keep = [k for k in get_col_names(table) if k not in remove]
    return table.vapply(lambda v: {k: v[k] for k in keep if k in v})


def rename(table, **kwargs):
    """
    Rename columns from a table

    :param table: a table
    :param kwargs: names of the columns to rename and new names old_name=new_name
    :return: a table (TupList) without the selected columns.
    """
    assert isinstance(table, TupList)

    new_names = dict(**kwargs)
    return table.vapply(
        lambda v: {new_names[k] if k in new_names else k: v[k] for k in v}
    )


def get_col_names(table, fast=False):
    """
    Get the names of the column of a tuplist

    :param table: a table (TupList of dict)
    :param fast: assume that the first row has all the columns.
    :return: a list of keys
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
    Join two tables with a left join.
    Shortcut to join(type="left")
    Inspired by R dplyr join functions.

    :param table1: 1st table (TupList with dict)
    :param table2: 2nd table (TupList with dict)
    :param by: list, dict or None.
        If the columns have the same name in both tables, a list of keys/column to use for the join.
        If the columns have the different names in both tables, a dict in the format {name_table1: name_table2}
        If by is None, use all the shared keys.
    :param suffix: if some columns have the same name in both tables but are not
     in "by", a suffix will be added to their names.
     With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
    :param empty: values to give to empty cells created by the join.
    :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
    :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
    :return: a TupList
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
    Join two tables with a right join.
    Shortcut to join(type="right")
    Inspired by R dplyr join functions.

    :param table1: 1st table (TupList with dict)
    :param table2: 2nd table (TupList with dict)
    :param by: list, dict or None.
        If the columns have the same name in both tables, a list of keys/column to use for the join.
        If the columns have the different names in both tables, a dict in the format {name_table1: name_table2}
        If by is None, use all the shared keys.
    :param suffix: if some columns have the same name in both tables but are not
     in "by", a suffix will be added to their names.
     With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
    :param empty: values to give to empty cells created by the join.
    :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
    :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
    :return: a TupList
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
    Join two tables with a full join.
    Shortcut to join(type="full")
    Inspired by R dplyr join functions.

    :param table1: 1st table (TupList with dict)
    :param table2: 2nd table (TupList with dict)
    :param by: list, dict or None.
        If the columns have the same name in both tables, a list of keys/column to use for the join.
        If the columns have the different names in both tables, a dict in the format {name_table1: name_table2}
        If by is None, use all the shared keys.
    :param suffix: if some columns have the same name in both tables but are not
     in "by", a suffix will be added to their names.
     With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
    :param empty: values to give to empty cells created by the join.
    :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
    :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
    :return: a TupList
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
    Join two tables with a left join.
    Shortcut to join(type="inner")
    Inspired by R dplyr join functions.

    :param table1: 1st table (TupList with dict)
    :param table2: 2nd table (TupList with dict)
    :param by: list, dict or None.
        If the columns have the same name in both tables, a list of keys/column to use for the join.
        If the columns have the different names in both tables, a dict in the format {name_table1: name_table2}
        If by is None, use all the shared keys.
    :param suffix: if some columns have the same name in both tables but are not
     in "by", a suffix will be added to their names.
     With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
    :param empty: values to give to empty cells created by the join.
    :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
    :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
    :return: a TupList
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

    :param tab1: (dict) table 1 grouped by join keys.
    :param tab2: (dict) table 2 grouped by join keys.
    :param jtype: (str) join type. Must be full, inner, right or left

    :return: Tuplist of unique join keys combinations.
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
    None values should never join with other None.
    Depending on the type of join, return the relevant rows with None values in keys.
    Example:
    result = left_join([{"a":1, "b":2}, {"a":None, "b":1}], [{"a":1, "c":1}, {"a":None, "c":1}])
    result should be [{"a":1, "b":2, "c":1}, {"a":None, "b":1, "c":None}]

    manage_join_none returns the part of the table where the join key is None: {"a":None, "b":1, "c":None}

    :param tab1: SuperDict table 1 grouped by join keys {(1,2): [{...}, {...}], (1,3):[{...}, {...}]}
    :param tab2: SuperDict table 2 grouped by join keys {(1,2): [{...}, {...}], (1,3):[{...}, {...}]}
    :param empty: value to use for missing values.
    :param t1_keys: columns of table 1
    :param t2_keys: columns of table 2
    :param by: keys to join by
    :param jtype: join type (left, right, full, inner)
    :return: Tuplist of rows joined on None values
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
    Join to tables.
    Inspired by R dplyr join functions.

    :param table1: 1st table (TupList with dict)
    :param table2: 2nd table (TupList with dict)
    :param by: list, dict or None.
        If the columns have the same name in both tables, a list of keys/column to use for the join.
        If the columns have the different names in both tables, a dict in the format {name_table1: name_table2}
        If by is None, use all the shared keys.
    :param suffix: if some columns have the same name in both tables but are not
     in "by", a suffix will be added to their names.
     With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
    :param jtype: type of join: "full"
    :param empty: values to give to empty cells created by the join.
    :param drop_if_nested: drop any nested dict from the table (columns containing list or dicts instead of scalars)
    drop_if_nested=False may generate and error o unexpected result.
    :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
    :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
    :return: a TupList
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

    # TODO: check if that always work
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
    Join a table with itself.
    Useful to create combinations of values from columns of a table.

    :param table: the table
    :param by: list, dict or None.
        If by is a list of keys/column, those columns will be used for the join.
        If by is a dict in the format {name_table1: name_table2}, those columns will be used for the join.
        If by is None, all combinations of rows will be created (join by dummy).
    :param suffix: suffix to add to column to create different names.
        Default is ("", "_2"). With default suffix, column id will appear as id and id_2.
    :param empty: values to give to empty cells created by the join.
    :return: a tuplist
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
    Transform all the keys of the dict of a tuplist into strings.

    :param tl: a tuplist of dict
    :return: The same tuplist with all keys as strings
    """
    return TupList([str_key(dic) for dic in tl])


def replace(tl, replacement=None, to_replace=None, fast=False):
    """
    Fill missing values of a TupList.

    :param tl: a TupList
    :param replacement: a single value or a dict of columns and values to use as replacement.
    :param to_replace: a single value or a dict of columns and values to replace.
    :param fast: assume that the first row has all the columns.

    :return: a TupList with missing values filled.
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
    Fill empty values of a tuplist.

    :param tl: a tuplist
    :param replacement: a single value or a dict of columns and values to use as replacement.
    :param fast: assume that the first row has all the columns.
    :return: a tuplist with empty values filled.
    """
    return replace(tl, replacement=replacement, to_replace=None, fast=fast)


def replace_nan(tl, replacement=None):
    """
    Fill nan values of a tuplist.

    :param tl: a tuplist
    :param replacement: the value to replace nan.
    :return: a tuplist with nan values filled.
    """
    return TupList(
        [{k: replacement if is_null(v) else v for k, v in dic.items()} for dic in tl]
    )


def drop_empty(tl, cols=None, fast=False):
    """
    Drop rows of a tuplist with empty values.

    :param tl: a tuplist
    :param cols: list of column names or single name.
    :param fast: assume that the first row has all the columns.
    :return: a tuplist with empty values dropped.
    """
    if cols is None:
        cols = get_col_names(tl, fast)
    else:
        cols = as_list(cols)
    tl2 = replace(tl, replacement=None, to_replace=None)

    return tl2.vfilter(lambda v: not any(k not in v or is_null(v[k]) for k in cols))


def is_null(v):
    """
    Return True if the value is None, NaN or NaT

    :param v: a scalar value
    :return: boolean
    """
    if isinstance(v, list):
        return [is_null(i) for i in v]
    return v is None or is_nan(v) or is_nat(v)


def is_nan(v):
    """
    Return True if the value is nan.
    Similar to np.isnan but return False instead of error if value is not a number.

    :param v: a value
    :return: bool
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
    Return True if the value is nat.
    Similar to np.isnat but return False instead of error if value is not a date.

    :param v: a value
    :return: bool
    """
    if isinstance(v, list):
        return [is_nat(i) for i in v]
    elif isinstance(v, np.datetime64):
        return np.isnat(v)
    else:
        return False


def pivot_longer(tl, cols, names_to="variable", value_to="value"):
    """
    pivot_longer() "lengthens" data, increasing the number of rows and decreasing the number of columns.
    The inverse transformation is pivot_wider()

    :param tl: a tuplist
    :param cols a list of columns to pivot
    :param names_to: the name of the new names column
    :param value_to: the name of the new value column

    :return: a tuplist
    example:
    table = TupList([{"a":1, "b":2, "c":3}, {"a":2, "b":3, "c":3}, {"a":5, "b":4, "c":3}])
    result = pivot_longer(table, ["b", "c"])
    result:
    [{'a': 1, 'variable': 'b', 'value': 2},
     {'a': 2, 'variable': 'b', 'value': 3},
     {'a': 5, 'variable': 'b', 'value': 4},
     {'a': 1, 'variable': 'c', 'value': 3},
     {'a': 2, 'variable': 'c', 'value': 3},
     {'a': 5, 'variable': 'c', 'value': 3}]
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
    pivot_wider() "widens" data, increasing the number of columns and decreasing the number of rows.
    The inverse transformation is pivot_longer()

    :param tl: a tuplist
    :param names_from: the name of the new names column
    :param value_from: the name of the new value column
    :param id_cols: set_a set of columns that uniquely identifies each observation.
     If None, use all columns except names_from and value_from.
    :param values_fill: set_a value or dict of values to fill in the missing values.
    :return: a tuplist
    example:
    tl = TupList([{'a': 1, 'variable': 'b', 'value': 2},
     {'a': 2, 'variable': 'b', 'value': 3},
     {'a': 5, 'variable': 'b', 'value': 4},
     {'a': 1, 'variable': 'c', 'value': 3},
     {'a': 2, 'variable': 'c', 'value': 3},
     {'a': 5, 'variable': 'c', 'value': 3}])
    result = pivot_wider(tl)
    result: [{'a': 1, 'b': 2, 'c': 3}, {'a': 2, 'b': 3, 'c': 3}, {'a': 5, 'b': 4, 'c': 3}]
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
    Transform a dict of lists into a list of dict. (see example)

    :param dic: a dict
    :return: a tuplist
    example:
    dic = dict(a=[1,2,3], b=[4,5,6])
    result= to_dictlist(dic)
    result: [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]
    """
    n_rows = max(len(as_list(dic[k])) for k in dic)
    dic2 = {k: to_len(dic[k], n_rows) for k in dic}
    return TupList([{k: dic2[k][i] for k in dic2} for i in range(n_rows)])


def lag(table, col, i=1):
    """
    Return column col from the table with lag i.

    :param table: a tuplist
    :param col: a key of the tuplist
    :param i: number of lags
    :return: a list
    example:
    table = TupList([{"a":1, "b":2}, {"a":2, "b":3}, {"a":5, "b":4}])
    result = lag(table, "a", 1)
    result: [None, 1, 2]
    """
    return table.kvapply(
        lambda k, v: table[k - i][col] if i <= k < len(table) + i else None
    )


def lag_col(table, col, i=1, replace=False):
    """
    Add a column to a tuplist which column col with lag i.

    :param table: a tuplist
    :param col: column name
    :param i: number of lags
    :param replace: replace the former value of the column. If not, create a new column lag_{col}_i
    :return: a tuplist
    example:
    table = TupList([{"a":1, "b":2}, {"a":2, "b":3}, {"a":5, "b":4}])
    result = lag_col(table, "a", 1)
    [{'a': 1, 'b': 2, 'lag_a_1': None}, {'a': 2, 'b': 3, 'lag_a_1': 1}, {'a': 5, 'b': 4, 'lag_a_1': 2}]
    """
    pre = "lag" if i > 0 else "lead"
    index = str(i) if i > 0 else str(-i)
    new_col = f"{pre}_{str(col)}_{index}" if not replace else col
    return mutate(table, **{new_col: lag(table, col, i)})


def drop_nested(df):
    """
    Drop any nested value from a tuplist.
    Nested value are dict or lists nested as dict values in the tuplist.
    This function assume df structure is homogenous and only look at the first row to find nested values.

    :param df: a tuplist
    :return: the tuplist without nested values.
    example:
    df = TupList([{"a":1, "b":2, "c": {"d":3}}, {"a":2, "b":2, "c":{"d":4}}])
    result = drop_nested(df)
    """
    for col in df[0]:
        if isinstance(df[0][col], list) or isinstance(df[0][col], dict):
            df = drop(df, col)
    return df


def distinct(table, columns):
    """
    Only keeps unique combinations values of the selected columns.
    When there are rows with duplicated values, the first one is kept.

    :param table: a tuplist (list of dict)
    :param columns: names of the columns
    :return: a tuplist (list of dict) with unique data.
    """
    return (
        TupList(table)
        .to_dict(indices=columns, result_col=None, is_list=True)
        .vapply(lambda v: v[0])
        .values_tl()
    )


def order_by(table, columns, reverse=False):
    """
    Reorder the table according to the given columns.

    :param table: a tuplist (list of dict).
    :param columns: names of the columns to use to sort the table.
    :param reverse:  if True, the sorted list is sorted in descending order.
    :return: the sorted tuplist
    """
    return TupList(table).sorted(
        key=lambda v: [v[c] for c in as_list(columns)], reverse=reverse
    )
