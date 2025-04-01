from typing import Callable

from mango.processing import (
    load_json,
    write_json,
    as_list,
    flatten,
    load_excel_light,
    write_excel_light,
    load_csv_light,
    write_csv_light,
)
from pytups import TupList, SuperDict

from .pytups_tools import (
    mutate,
    sum_all,
    join,
    left_join,
    right_join,
    full_join,
    inner_join,
    select,
    rename,
    get_col_names,
    drop,
    summarise,
    str_key_tl,
    replace,
    replace_empty,
    pivot_longer,
    pivot_wider,
    drop_empty,
    lag_col,
    replace_nan,
    group_mutate,
    to_dictlist,
    distinct,
    order_by,
    drop_nested,
    group_by,
    auto_join,
)
from .table_tools import is_subset


class Table(TupList):
    def __init__(self, data=None, check=False):
        if data is None:
            data = []
        super().__init__(data)
        if check:
            self.check_type()

    # Adapt TupList methods to return table
    def __getitem__(self, key: int):
        if not isinstance(key, slice):
            return list.__getitem__(self, key)
        else:
            return Table(list.__getitem__(self, key))

    def __add__(self, *args, **kwargs) -> "Table":
        return Table(super().__add__(*args, **kwargs))

    def vapply(self, func: Callable, *args, **kwargs) -> "Table":
        """
        maps function into each element of TupList

        :param callable func: function to apply
        :return: new :py:class:`Table`
        """
        return Table(super().vapply(func, *args, **kwargs))

    def kapply(self, func, *args, **kwargs) -> "Table":
        """
        maps function into each key of TupList

        :param callable func: function to apply
        :return: new :py:class:`Table`
        """
        return Table(super().kapply(func, *args, **kwargs))

    def kvapply(self, func, *args, **kwargs) -> "Table":
        """
        maps function into each element of TupList with indexes

        :param callable func: function to apply
        :return: new :py:class:`Table`
        """
        return Table(super().kvapply(func, *args, **kwargs))

    def copy_shallow(self) -> "Table":
        """
        Copies the list only. Not it's contents

        :return: new :py:class:`Table`
        """
        return Table(super().copy_shallow())

    def copy_deep(self) -> "Table":
        return Table(super().copy_deep())

    def vfilter(self, function: Callable) -> "Table":
        """
        Filter a table. Keeps the rows for which the function returns True.

        :param func: function to use to filter
        :return: new :py:class:`Table`
        """
        return Table(super().vfilter(function))

    def unique(self) -> "Table":
        """
        Eliminates duplicated rows.
        When there are rows with duplicated values, the first one is kept.

        :return: new :py:class:`Table`
        """
        try:
            columns = self.get_col_names()
            return self.distinct(columns)
        except:
            raise NotImplementedError(
                "Cannot apply unique to a list of dict. Use distinct instead"
            )

    def unique2(self) -> "Table":
        """
        Eliminates duplicated rows.
        When there are rows with duplicated values, the first one is kept.

        :return: new :py:class:`Table`
        """
        try:
            columns = self.get_col_names()
            return self.distinct(columns)
        except:
            raise NotImplementedError(
                "Cannot apply unique2 to a list of dict. Use distinct instead"
            )

    def sorted(self, **kwargs) -> "Table":
        """
        Sorts the table according to the given function (key argument)

        :param kwargs: arguments for sorted function
            main arguments for sorted are:
            - key
            - reverse

        :example: my_table.sorted(key= lambda x : (x['Distance']-30)**2, reverse=True)
        :return: new :py:class:`Table`
        """
        try:
            return Table(super().sorted(**kwargs))
        except:
            raise NotImplementedError(
                "A list of dict cannot be sorted. Use order_by instead"
            )

    # New or modified methods
    def take(self, *args, use_numpy=False) -> TupList:
        """
        Extract values from a columns of a table.

        Example:
        result=Table([{"a":1, "b":2}, {"a":3, "b":4}]).take("a", "b")
        result: [(1,2), (3,4)]

        :param args: name of the columns to extract
        :param use_numpy: use numpy methods in take
        :return: a list of tuples.
        """
        indices = flatten(args)
        if len(indices) == 1:
            indices = indices[0]
        return TupList(self).take(indices, use_numpy)

    def __str__(self):
        if self.len():
            if not isinstance(self[0], dict):
                return super.__str__(self)
            columns = len(self[0]) if self[0] is not None else 0
            return f"Table ({self.len()} rows, {columns} columns):\n" + self.show_rows(
                0, self.len()
            )
        else:
            return "Empty table"

    def show_rows(self, n1, n2=None) -> str:
        """
        Show the n1 th row of the table or rows between n1 and n2.

        :param n1: row number to start
        :param n2: row number to end (if None, only show row n1)
        :return: `string`
        """
        if n2 is None:
            n2 = n1 + 1
        return "".join(f"{i} {self[i]}\n" for i in range(n1, n2))

    def head(self, n=10) -> "Table":
        """
        Return the first n rows of a table.

        :param n: Number of rows to show (default:10).
        :return: new :py:class:`Table`
        """
        if self.len() < n:
            return self
        else:
            return self[:n]

    def peek(self, n=3, name=None) -> "Table":
        """
        Show the first, middle and last n rows of the table.

        :param n: number of rows to show in each part.
        :param name: name or message to print with the table
        :return: new :py:class:`Table`
        """
        if name is None:
            name = ""
        else:
            name = name + ": "
        if self.len() < 3 * n:
            print(f"{name}{self}")
            return self
        else:
            middle = (self.len() - n) // 2
            message = (
                f"{name}Table ({self.len()} rows, , {len(self[0])} columns):\n"
                + self.show_rows(0, n)
                + "...\n"
                + self.show_rows(middle, middle + n)
                + "...\n"
                + self.show_rows(self.len() - n, self.len())
            )
            print(message)
            return self

    def mutate(self, **kwargs) -> "Table":
        """
        Add or modify a column in a table.

        Example: creates or overwrites the columns a,b and c
            mutate(table, a=3, b=[4,5,6], c=lambda v: v["a"]+v["b"])

        Note: All changes are applied over the input table and do not take into account the other changes.

        :param kwargs: Named arguments with the changes to apply.
            The values can be:
            - A single value, which will be applied to each row.
            - A list with all the values of the column.
            - A function to apply to the row.
        :return: new :py:class:`Table`
        """
        return Table(mutate(self, **kwargs))

    def group_by(self, col) -> SuperDict:
        """
        Group the rows of a table by the value fo some columns

        :param col: single name of list of columns to use to group the rows

        :return: `SuperDict`
        """
        return group_by(self, col)

    def group_mutate(self, group_by, **func) -> "Table":
        """
        Group by the given columns and apply the given functions to the others.

        :param group_by: name of the columns to group.
        :param func: function to apply to the named column. ex: a = first, b = mean
        :return: new :py:class:`Table`
        """
        return Table(group_mutate(self, group_by, **func))

    def sum_all(self, group_by=None) -> "Table":
        """
        Group by the given columns and sum the others.

        :param group_by: name of the columns to group.
        :return: new :py:class:`Table`
        """
        return Table(sum_all(self, group_by))

    def summarise(self, group_by=None, default: Callable = None, **func) -> "Table":
        """
        Group by the given columns and apply the given functions to the others.

        :param group_by: name of the columns to group.
        :param default: default function to apply to non-grouped columns.
        :param func: function to apply to the named column. ex: a = first, b = mean
        :return: new :py:class:`Table`
        """
        return Table(summarise(self, group_by, default=default, **func))

    def join(
        self,
        table2,
        by=None,
        suffix=None,
        jtype="full",
        empty=None,
        if_empty_table_1=None,
        if_empty_table_2=None,
    ) -> "Table":
        """
        Join to tables.
        Inspired by R dplyr join functions.

        :param table2: 2nd table (Tuplist with dict)
        :param by: list, dict or None.
            If the columns have the same name in both tables, a list of keys/column to use for the join.
            If the columns have different names in both tables, a dict in the format {name_table1: name_table2}
            If by is None, use all the shared keys.
        :param suffix: if some columns have the same name in both tables but are not
         in "by", a suffix will be added to their names.
         With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
        :param jtype: type of join: "full"
        :param empty: values to give to empty cells created by the join.
        :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
        :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
        :return: new :py:class:`Table`
        """
        return Table(
            join(
                self,
                table2,
                by=by,
                suffix=suffix,
                jtype=jtype,
                empty=empty,
                if_empty_table_1=if_empty_table_1,
                if_empty_table_2=if_empty_table_2,
            )
        )

    def left_join(
        self,
        table2,
        by=None,
        suffix=None,
        empty=None,
        if_empty_table_1=None,
        if_empty_table_2=None,
    ) -> "Table":
        """
        Shortcut to join(type="left")

        :param table2: 2nd table (Tuplist with dict)
        :param by: list, dict or None.
            If the columns have the same name in both tables, a list of keys/column to use for the join.
            If the columns have different names in both tables, a dict in the format {name_table1: name_table2}
            If by is None, use all the shared keys.
        :param suffix: if some columns have the same name in both tables but are not
         in "by", a suffix will be added to their names.
         With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
        :param empty: values to give to empty cells created by the join.
        :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
        :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
        :return: new :py:class:`Table`
        """
        return Table(
            left_join(
                self,
                table2,
                by=by,
                suffix=suffix,
                empty=empty,
                if_empty_table_1=if_empty_table_1,
                if_empty_table_2=if_empty_table_2,
            )
        )

    def right_join(
        self,
        table2,
        by=None,
        suffix=None,
        empty=None,
        if_empty_table_1=None,
        if_empty_table_2=None,
    ) -> "Table":
        """
        Shortcut to join(type="right")

        :param table2: 2nd table (Tuplist with dict)
        :param by: list, dict or None.
            If the columns have the same name in both tables, a list of keys/column to use for the join.
            If the columns have different names in both tables, a dict in the format {name_table1: name_table2}
            If by is None, use all the shared keys.
        :param suffix: if some columns have the same name in both tables but are not
         in "by", a suffix will be added to their names.
         With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
        :param empty: values to give to empty cells created by the join.
        :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
        :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
        :return: new :py:class:`Table`
        """
        return Table(
            right_join(
                self,
                table2,
                by=by,
                suffix=suffix,
                empty=empty,
                if_empty_table_1=if_empty_table_1,
                if_empty_table_2=if_empty_table_2,
            )
        )

    def full_join(
        self,
        table2,
        by=None,
        suffix=None,
        empty=None,
        if_empty_table_1=None,
        if_empty_table_2=None,
    ) -> "Table":
        """
        Shortcut to join(type="full")

        :param table2: 2nd table (Tuplist with dict)
        :param by: list, dict or None.
            If the columns have the same name in both tables, a list of keys/column to use for the join.
            If the columns have different names in both tables, a dict in the format {name_table1: name_table2}
            If by is None, use all the shared keys.
        :param suffix: if some columns have the same name in both tables but are not
         in "by", a suffix will be added to their names.
         With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
        :param empty: values to give to empty cells created by the join.
        :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
        :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
        :return: new :py:class:`Table`
        """
        return Table(
            full_join(
                self,
                table2,
                by=by,
                suffix=suffix,
                empty=empty,
                if_empty_table_1=if_empty_table_1,
                if_empty_table_2=if_empty_table_2,
            )
        )

    def inner_join(
        self,
        table2,
        by=None,
        suffix=None,
        empty=None,
        if_empty_table_1=None,
        if_empty_table_2=None,
    ) -> "Table":
        """
        Shortcut to join(type="inner")

        :param table2: 2nd table (Tuplist with dict)
        :param by: list, dict or None.
            If the columns have the same name in both tables, a list of keys/column to use for the join.
            If the columns have different names in both tables, a dict in the format {name_table1: name_table2}
            If by is None, use all the shared keys.
        :param suffix: if some columns have the same name in both tables but are not
         in "by", a suffix will be added to their names.
         With suffix=["_1","_2"], shared column "x" will become "x_1", "x_2"
        :param empty: values to give to empty cells created by the join.
        :param if_empty_table_1: (dict or list) if table 1 is empty, it will be replaced by this dict in the join.
        :param if_empty_table_2: (dict or list) if table 2 is empty, it will be replaced by this dict in the join.
        :return: new :py:class:`Table`
        """
        return Table(
            inner_join(
                self,
                table2,
                by=by,
                suffix=suffix,
                empty=empty,
                if_empty_table_1=if_empty_table_1,
                if_empty_table_2=if_empty_table_2,
            )
        )

    def auto_join(self, by=None, suffix=None, empty=None) -> "Table":
        """
        Join a table with itself.
        Useful to create combinations of values from columns of a table.

        :param by: list, dict or None.
            If by is a list of keys/column, those columns will be used for the join.
            If by is a dict in the format {name_table1: name_table2}, those columns will be used for the join.
            If by is None, all combinations of rows will be created (join by dummy).
        :param suffix: suffix to add to column to create different names.
            Default is ("", "_2"). With default suffix, column id will appear as id and id_2.
        :param empty: values to give to empty cells created by the join.
        :return: new :py:class:`Table`
        """
        return Table(auto_join(self, by=by, suffix=suffix, empty=empty))

    def select(self, *args) -> "Table":
        """
        Select columns from a table

        :param args: names of the columns to select
        :return: new :py:class:`Table`
        """
        return Table(select(self, *args))

    def drop(self, *args) -> "Table":
        """
        Drop columns from a table

        :param args: names of the columns to drop
        :return: new :py:class:`Table`
        """
        return Table(drop(self, *args))

    def rename(self, **kwargs) -> "Table":
        """
        Rename columns from a table

        :param kwargs: names of the columns to rename and new names old_name="new_name"
        :return: new :py:class:`Table`
        """
        return Table(rename(self, **kwargs))

    def filter(self, func) -> "Table":
        """
        Filter a table. Keeps the rows for which the function returns True.

        :param func: function to use to filter
        :return: new :py:class:`Table`
        """
        if not self.len():
            return self

        return Table(self.vfilter(func))

    def get_col_names(self, fast=False) -> "list":
        """
        Get the names of the column of the table.

        :param fast: assume that the first row has all the columns.
        :return: :py:class:`list`
        """
        return get_col_names(self, fast)

    def to_columns(self) -> "SuperDict":
        """
        Create a dict with a list of values for each column of the table.

        :return: a SuperDict
        """
        if self.len() == 0:
            return SuperDict()
        table = self.replace_empty(None)
        return SuperDict({col: table.take(col) for col in self.get_col_names()})

    @classmethod
    def from_columns(cls, dct) -> "Table":
        """
        Create a table from a dict of list (columns)

        Example:
        dic = dict(a=[1,2,3], b=[4,5,6])
        result= Table.from_column(dic)
        result: [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]

        :param dct: a dict of list
        :return: new :py:class:`Table`
        """
        return cls(to_dictlist(dct))

    def get_index(self, cond) -> "list":
        """
        Get row number for rows which respect a condition.

        :param cond: condition/filter (function) to apply to the rows
        :return: :py:class:`list`
        """
        return [i for i, v in enumerate(self) if cond(v)]

    def replace(self, replacement=None, to_replace=None, fast=False) -> "Table":
        """
        Fill missing values of a tuplist.

        :param replacement: a single value or a dict of columns and values to use as replacement.
        :param to_replace: a single value or a dict of columns and values to replace.
        :param fast: assume that the first row has all the columns.

        :Example: replacing the value 25 in the column Edad by 35 and "Madrid" in the column Ciudad by "Paris"
        my_table.replace({"Edad":35,"Ciudad":"Paris"},{"Edad":25,"Ciudad":"Madrid"})

        :return: new :py:class:`Table`
        """
        return Table(replace(self, replacement, to_replace, fast))

    def replace_empty(self, replacement=None, fast=False) -> "Table":
        """
        Fill empty values.

        :param replacement: a single value or a dict of columns and values to use as replacement.
        :param fast: assume that the first row has all the columns.
        :return: new :py:class:`Table`
        """
        return Table(replace_empty(self, replacement, fast))

    def replace_nan(self, replacement=None) -> "Table":
        """
        Fill nan values.

        :param replacement: a single value or a dict of columns and values to use as replacement.
        :return: new :py:class:`Table`
        """
        return Table(replace_nan(self, replacement))

    def pivot_longer(self, cols, names_to="variable", value_to="value") -> "Table":
        """
        pivot_longer() "lengthens" data, increasing the number of rows and decreasing the number of columns.
        The inverse transformation is pivot_wider()

        :param cols: a list of columns to pivot
        :param names_to: the name of the new names column
        :param value_to: the name of the new value column
        :return: new :py:class:`Table`
        """
        return Table(pivot_longer(self, cols, names_to, value_to))

    def pivot_wider(
        self, names_from="variable", value_from="value", id_cols=None, values_fill=None
    ) -> "Table":
        """
        pivot_wider() "widens" data, increasing the number of columns and decreasing the number of rows.
        The inverse transformation is pivot_longer()

        :param names_from: the name of the new name column
        :param value_from: the name of the new value column
        :param id_cols: set_a set of columns that uniquely identifies each observation.
         If None, use all columns except names_from and value_from.
        :param values_fill: set_a value or dict of values to fill in the missing values.

        :return: new :py:class:`Table`
        """
        return Table(pivot_wider(self, names_from, value_from, id_cols, values_fill))

    def drop_empty(self, cols=None, fast=False) -> "Table":
        """
        Drop rows whose value in the specified columns is empty.

        :param cols: list of column names or single name.
        :param fast: assume that the first row has all the columns.
        :return: new :py:class:`Table`
        """
        return Table(drop_empty(self, cols, fast))

    def lag_col(self, col, i=1, replace=False) -> "Table":
        """
        Lag a column by i steps.

        :param col: the name of the column to lag
        :param i: the number of steps to lag
        :param replace: replace the former value of the column. If not, create a new column lag_{col}_i
        :return: new :py:class:`Table`
        """
        return Table(lag_col(self, col, i, replace))

    def distinct(self, columns) -> "Table":
        """
        Only keeps unique combinations values of the selected columns.
        When there are rows with duplicated values, the first one is kept.

        :param columns: names of the columns.
        :return: new :py:class:`Table`
        """
        return Table(distinct(self, columns))

    def order_by(self, columns, reverse=False) -> "Table":
        """
        Reorder the table according to the given columns.

        :param columns: names of the columns to use to sort the table.
        :param reverse:  if True, the sorted list is sorted in descending order.
        :return: new :py:class:`Table`
        """
        return Table(order_by(self, columns=columns, reverse=reverse))

    def drop_nested(self) -> "Table":
        """
        Drop any nested column from a table.
        Nested value are dict, lists or tuples.
        This function assume the table structure is homogenous and only look at the first row to find nested values.

        :return: new :py:class:`Table`
        """

        def drop_nested_temp(df):
            for col in df[0]:
                print(type(df[0][col]))
                if (
                    isinstance(df[0][col], list)
                    or isinstance(df[0][col], dict)
                    or isinstance(df[0][col], tuple)
                ):
                    df = drop(df, col)
            return df

        return Table(drop_nested_temp(self))

    def check_type(self):
        """
        Check that the table is a list of dict. Raises an error if not.
        :return: `None`
        """
        if not isinstance(self, list):
            raise TypeError("set_a Table must be a list of dict, not %s" % type(self))
        if not self._all_rows_are_dict():
            if self[0]:
                raise TypeError(
                    "set_a Table must be a list of dict, not list of %s" % type(self[0])
                )

    def _row_is_dict(self, i=0) -> bool:
        """
        Checks if the row i is a dict

        :return: :py:class:`bool`
        """
        return isinstance(self[i], dict)

    def _all_rows_are_dict(self):
        """
        Checks if all the rows are dicts

        :return: :py:class:`bool`
        """
        return all(self._row_is_dict(i) for i in range(len(self)))

    def to_set2(self, columns) -> "TupList":
        """
        Create a list of unique conmbinations of the specified columns

        :param columns: Columns to select to create the set. (string or list of strings)
        :return: a tuplist with unique values
        """
        if len(self) == 0:
            return TupList()
        table_col = set(self[0].keys())
        if not is_subset(columns, table_col):
            raise KeyError(f"key(s) {columns} are not in table.")
        return self.take(columns).unique2()

    def to_param(self, keys, value, is_list=False) -> "SuperDict":
        """
        Create a dict with the given columns as keys and values.

        :param keys: columns to use as keys.
        :param value: column to use as values.
        :param is_list: True if the values are a list instead of a single value.
        :return: a superdict indexed by the given keys.
        """
        if not self.len():
            return SuperDict()

        table_col = set(self[0].keys())
        if not is_subset(keys, table_col):
            raise KeyError(f"key(s) {keys} are not in table.")
        if not is_list and not self.is_unique(keys):
            raise ValueError(
                "There are duplicate values for keys {keys}."
                + " Method to_param with is_list=False expect a single value for each keys."
            )
        return self.to_dict(indices=keys, result_col=value, is_list=is_list)

    def is_unique(self, columns) -> "bool":
        """
        Check if the combination of values of given columns is unique.

        :param columns: combination of columns to check.
        :return: `bool`
        """
        len_unique = self.distinct(columns).len()
        return len_unique == self.len()

    def add_row(self, **kwargs) -> "Table":
        """
        Add a row to the table.
        Missing columns are filled with value None.

        :param kwargs: values of the column in the format column_name=value
        :return: new :py:class:`Table`
        """
        result = self + [{**kwargs}]
        return result.replace(replacement=None, to_replace=None)

    def rbind(self, table: list) -> "Table":
        """
        Bind two tables by rows.

        :param table: another table
        :return: new :py:class:`Table`
        """
        return (self + Table(table)).replace(replacement=None, to_replace=None)

    def col_apply(self, columns, func: Callable, **kwargs) -> "Table":
        """
        Apply a function to the specfied columns

        :param columns: column or list of columns.
        :param func: function to apply.
        :return: new :py:class:`Table`
        """
        result = self
        for col in as_list(columns):
            result = result.mutate(**{col: lambda v: func(v[col], **kwargs)})
        return result

    @classmethod
    def format_dataset(cls, dataset) -> dict:
        """
        For each key/value pair in the dataset, cast the value into Table if it is a list.

        :param dataset: a data instance in dict/json format.
        :return: a dict
        """
        return {
            k: cls(str_key_tl(v)) if isinstance(v, list) else v
            for k, v in dataset.items()
        }

    @classmethod
    def dataset_from_json(cls, path, **kwargs) -> dict:
        """
        Load a json file and For each key/value pair in the dataset, cast the value into Table if it is a list.

        :param path: path of the json file
        :return: new :py:class:`dict`

        """
        data = load_json(path, **kwargs)
        return cls.format_dataset(data)

    # Save and load functions
    @classmethod
    def from_pandas(cls, df) -> "Table":
        """
        Create a table from a pandas dataframe.

        :param df: a pandas dataframe
        :return: :py:class:`Table`
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is not present in your system. Try: pip install pandas"
            )
        return cls(pd.DataFrame(df).to_dict(orient="records"))

    def to_pandas(self) -> "pandas.DataFrame":
        """
        Create a pandas dataframe from a table.

        :return: new :py:class:`pandas.DataFrame`
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is not present in your system. Try: pip install pandas"
            )
        return pd.DataFrame.from_records(self)

    def pk_save(self, file):
        """
        Save the table in a pickle file.

        :param file: path of the file
        :return: nothing
        """
        import pickle as pk

        if ".pickle" not in file:
            file = file + ".pickle"

        with open(file, "wb") as handle:
            pk.dump(self, handle)

    @classmethod
    def pk_load(cls, file) -> "Table":
        """
        Load a Table from a pickle file.

        :param file: path of the file
        :return: :py:class:`Table`
        """
        import pickle as pk

        if ".pickle" not in file:
            file = file + ".pickle"
        with open(file, "rb") as f:
            data = pk.load(f)
        return cls(data)

    def to_json(self, path):
        """
        Export the table to a json file.
        If any column name is a number, transform it into string.

        :param path: path of the json file
        :return: nothing
        """
        write_json(str_key_tl(self), path)

    @classmethod
    def from_json(cls, path, **kwargs) -> "Table":
        """
        Create a table from a json file.

        :param path: path to json file
        :return: :py:class:`Table`
        """
        return cls(load_json(path, **kwargs))

    def apply(self, func: Callable, *args, **kwargs):
        """
        Apply a function to the entire table.
        Useful to chain varius functions applied to the entire table.

        :param func: a function which take the table as a first argument.
        :param args: args of the function
        :param kwargs: kwargs of the function
        :return: what the function returns.
        """
        return func(self, *args, **kwargs)

    @classmethod
    def dataset_from_excel(cls, path, sheets=None) -> dict:
        """
        Read an Excel file and return a dict of Table()

        :param path: path fo the Excel file.
        :param sheets: list of sheets to read (all the sheets are read if None)
        :return: :py:class:`dict`
        """
        data = load_excel_light(path, sheets)
        return cls.format_dataset(data)

    def to_excel(self, path, sheet_name=None):
        """
        Write the table to an Excel file

        :param path: path fo the Excel file.
        :param sheet_name: Name of the Excel sheet.
        :return: None
        """
        if sheet_name is None:
            sheet_name = "Sheet1"
        return write_excel_light(path, {sheet_name: self})

    @classmethod
    def from_csv(cls, path, sep=",", encoding=None) -> "Table":
        """
        Load the table from a csv file.

        :param path: path fo the csv file.
        :param sep: column separator in the csv file. (detected automatically if None).
        :param encoding: encoding.
        :return: :py:class:`Table`
        """
        data = load_csv_light(path, sep, encoding)
        return Table(data)

    def to_csv(self, path, sep=",", encoding=None):
        """
        Write the table to a csv file.

        :param path: path fo the Excel file.
        :param sep: column separator in the csv file.
        :param encoding: encoding.
        :return: nothing.
        """
        write_csv_light(path, self, sep, encoding)
