from pytups import TupList, SuperDict
from typing import Callable
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
from mango.processing import load_json, write_json, as_list


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
        return Table(super().vapply(func, *args, **kwargs))

    def kapply(self, func, *args, **kwargs) -> "Table":
        return Table(super().kapply(func, *args, **kwargs))

    def kvapply(self, func, *args, **kwargs) -> "Table":
        return Table(super().kvapply(func, *args, **kwargs))

    def copy_shallow(self) -> "Table":
        return Table(super().copy_shallow())

    def copy_deep(self) -> "Table":
        return Table(super().copy_deep())

    def vfilter(self, function: Callable) -> "Table":
        return Table(super().vfilter(function))

    def unique(self) -> "Table":
        try:
            return Table(super().unique())
        except:
            raise NotImplementedError(
                "Cannot apply unique to a list of dict. Use distinct instead"
            )

    def unique2(self) -> "Table":
        try:
            return Table(super().unique2())
        except:
            raise NotImplementedError(
                "Cannot apply unique2 to a list of dict. Use distinct instead"
            )

    def sorted(self, **kwargs) -> "Table":
        raise NotImplementedError(
            "A list of dict cannot be sorted. Use order_by instead"
        )

    def take(self, indices, use_numpy=False) -> TupList:
        indices = as_list(indices)
        if len(indices) == 1:
            indices = indices[0]

        return TupList(self).take(indices, use_numpy)

    # New or modified methods
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

    def show_rows(self, n1, n2=None):
        """
        Show the n1 th row of the table or rows between n1 and n2.

        :param n1: row number to start
        :param n2: row number to end (if None, only show row n1)
        :return: a string
        """
        if n2 is None:
            n2 = n1 + 1
        return "".join(f"{i} {self[i]}\n" for i in range(n1, n2))

    def head(self, n=10):
        """
        Return the first n rows of a table.

        :param n: Number of rows to show (default:10).
        :return: set_a table with n rows or less.
        """
        if self.len() < n:
            return self
        else:
            return self[:n]

    def peek(self, n=3, name=None):
        """
        Show the first, middle and last n rows of the table.

        :param n: number of rows to show in each part.
        :param name: name or message to print with the table
        :return: the printed string
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

        Example:
        mutate(table, a=3, b=[4,5,6], c=lambda v: v["a"]+v["b"], d = mean)

        Note: all changes are applied over the input table and do not take into account the other changes.

        :param kwargs: named arguments with the changes to apply.
        The values can be:
         - a single value which will be applied to each row
         - a list with all the values of the column
         - a function to apply to the row.
        :return: a table
        """
        return Table(mutate(self, **kwargs))

    def group_by(self, col) -> SuperDict:
        """
        Group the rows of a table by the value fo some columns

        :param col: single name of list of columns to use to group the rows
        :return a SuperDict
        """
        return group_by(self, col)

    def group_mutate(self, group_by, **func) -> "Table":
        """
        Group by the given columns and apply the given functions to the others.

        :param group_by: name of the columns to group.
        :param func: function to apply to the named column. ex: a = first, b = mean
        :return: a table (Tuplist of dict).
        """
        return Table(group_mutate(self, group_by, **func))

    def sum_all(self, group_by=None) -> "Table":
        """
        Group by the given columns and sum the others.

        :param group_by: name of the columns to group.
        :return: a table (Tuplist of dict)
        """
        return Table(sum_all(self, group_by))

    def summarise(self, group_by=None, default: Callable = None, **func) -> "Table":
        """
        Group by the given columns and apply the given functions to the others.

        :param group_by: name of the columns to group.
        :param default: default function to apply to non-grouped columns.
        :param func: function to apply to the named column. ex: a = first, b = mean
        :return: a table (Tuplist of dict).
        """
        return Table(summarise(self, group_by, default=default, **func))

    def join(self, table2, by=None, suffix=None, jtype="full", empty=None) -> "Table":
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
        :return: a Table
        """
        return Table(join(self, table2, by=by, suffix=suffix, jtype=jtype, empty=empty))

    def left_join(self, table2, by=None, suffix=None, empty=None) -> "Table":
        """
        Shortcut to join(type="left")
        """
        return Table(left_join(self, table2, by=by, suffix=suffix, empty=empty))

    def right_join(self, table2, by=None, suffix=None, empty=None) -> "Table":
        """
        Shortcut to join(type="right")
        """
        return Table(right_join(self, table2, by=by, suffix=suffix, empty=empty))

    def full_join(self, table2, by=None, suffix=None, empty=None) -> "Table":
        """
        Shortcut to join(type="full")
        """
        return Table(full_join(self, table2, by=by, suffix=suffix, empty=empty))

    def inner_join(self, table2, by=None, suffix=None, empty=None) -> "Table":
        """
        Shortcut to join(type="inner")
        """
        return Table(inner_join(self, table2, by=by, suffix=suffix, empty=empty))

    def auto_join(self, by=None, suffix=None, empty=None) -> "Table":
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
        return Table(auto_join(self, by=None, suffix=None, empty=None))

    def select(self, *args) -> "Table":
        """
        Select columns from a table

        :param args: names of the columns to select
        :return: a table (Tuplist) with the selected columns.
        """
        return Table(select(self, *args))

    def drop(self, *args) -> "Table":
        """
        Drop columns from a table

        :param args: names of the columns to drop
        :return: a table without the selected columns.
        """
        return Table(drop(self, *args))

    def rename(self, **kwargs) -> "Table":
        """
        Rename columns from a table

        :param kwargs: names of the columns to rename and new names old_name=new_name
        :return: a table without the selected columns.
        """
        return Table(rename(self, **kwargs))

    def filter(self, func) -> "Table":
        """
        Filter a table.

        :param func: function to use to filter
        :return: the filtered table.
        """
        if not self.len():
            return self

        return Table(self.vfilter(func))

    def get_col_names(self) -> "Table":
        """
        Get the names of the column of the table.

        :return: a list of keys
        """
        return get_col_names(self)

    def to_columns(self) -> "SuperDict":
        """
        Create a dict with a list of values for each column of the table.

        :return: a dict
        """
        if self.len() == 0:
            return SuperDict()
        table = self.replace_empty(None)
        return SuperDict({col: table.take(col) for col in self.get_col_names()})

    @staticmethod
    def from_columns(dct) -> "Table":
        """
        Create a table from a dict of list (columns)

        Example:
        dic = dict(a=[1,2,3], b=[4,5,6])
        result= Table.from_column(dic)
        result: [{'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}]

        :param dct: a dict of list
        :return: a Table
        """
        return Table(to_dictlist(dct))

    def get_index(self, cond) -> "list":
        """
        Get row number for rows which respect a condition.

        :param cond: condition/filter to apply to the rows
        :return: a list of row numbers
        """
        return [i for i, v in enumerate(self) if cond(v)]

    def replace(self, replacement=None, to_replace=None) -> "Table":
        """
        Fill missing values of a tuplist.

        :param replacement: a single value or a dict of columns and values to use as replacement.
        :param to_replace: a single value or a dict of columns and values to replace.

        :return: the table with missing values filled.
        """
        return Table(replace(self, replacement, to_replace))

    def replace_empty(self, replacement=None) -> "Table":
        """
        Fill empty values of a tuplist.

        :param replacement: a single value or a dict of columns and values to use as replacement.
        :return: the table with empty values filled.
        """
        return Table(replace_empty(self, replacement))

    def replace_nan(self, replacement=None) -> "Table":
        """
        Fill nan values of a tuplist.

        :param replacement: a single value or a dict of columns and values to use as replacement.
        :return: the table with nan values filled.
        """
        return Table(replace_nan(self, replacement))

    def pivot_longer(self, cols, names_to="variable", value_to="value") -> "Table":
        """
        pivot_longer() "lengthens" data, increasing the number of rows and decreasing the number of columns.
        The inverse transformation is pivot_wider()

        :param cols: a list of columns to pivot
        :param names_to: the name of the new names column
        :param value_to: the name of the new value column
        :return: the table with the new columns
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

        :return: the table with the new columns
        """
        return Table(pivot_wider(self, names_from, value_from, id_cols, values_fill))

    def drop_empty(self, cols=None) -> "Table":
        """
        Drop rows with empty values of a tuplist.

        :return: the table with empty values dropped.
        """
        return Table(drop_empty(self, cols))

    def lag_col(self, col, i=1, replace=False) -> "Table":
        """
        Lag a column by i steps.

        :param col: the name of the column to lag
        :param i: the number of steps to lag
        :param replace: replace the former value of the column. If not, create a new column lag_{col}_i
        :return: the table with the new column
        """
        return Table(lag_col(self, col, i, replace))

    def distinct(self, columns) -> "Table":
        """
        Only keeps unique combinations values of the selected columns.
        When there are rows with duplicated values, the first one is kept.

        :param columns: names of the columns.
        :return: a Table (list of dict) with unique data.
        """
        return Table(distinct(self, columns))

    def order_by(self, columns, reverse=False) -> "Table":
        """
        Reorder the table according to the given columns.

        :param columns: names of the columns to use to sort the table.
        :param reverse:  if True, the sorted list is sorted in descending order.
        :return: the sorted Table
        """
        return Table(order_by(self, columns=columns, reverse=reverse))

    def drop_nested(self) -> "Table":
        """
        Drop any nested column from a table.
        Nested value are dict or lists nested as dict values in the table.
        This function assume df structure is homogenous and only look at the first row to find nested values.

        :return: the table without nested values.
        """
        return Table(drop_nested(self))

    def check_type(self):
        """
        Check that the table is a list of dict.
        """
        if not isinstance(self, list):
            raise TypeError("set_a Table must be a list of dict, not %s" % type(self))
        if not self._all_rows_are_dict():
            if self[0]:
                raise TypeError(
                    "set_a Table must be a list of dict, not list of %s" % type(self[0])
                )

    def _row_is_dict(self, i=0):
        return isinstance(self[i], dict)

    def _all_rows_are_dict(self):
        return all(self._row_is_dict(i) for i in range(len(self)))

    def to_set2(self, columns) -> "TupList":
        """
        Create a list of unique value from some columns of the table

        :param columns: Columns to select to create the set.
        :return: a tuplist with unique values
        """
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
        :return: True if the combination of values of the columns is unique.
        """
        len_unique = self.distinct(columns).len()
        return len_unique == self.len()

    def add_row(self, **kwargs):
        """
        Add a row to the table.
        Missing columns are filled with value None.

        :param kwargs: values of the column in the format column_name=value
        :return: the table with added row.
        """
        result = self + [{**kwargs}]
        return result.replace(replacement=None, to_replace=None)

    def rbind(self, table: TupList):
        """
        Bind two tables by rows.

        :param table: another table
        :return: the complete table.
        """
        return (self + Table(table)).replace(replacement=None, to_replace=None)

    @staticmethod
    def format_dataset(dataset):
        """
        Format an entire data instance applying Table() to every table.
        Leave dict as they are.

        :param dataset: a data instance in dict/json format.
        :return: a dict of Tables
        """
        return {
            k: Table(str_key_tl(v)) if isinstance(v, list) else v
            for k, v in dataset.items()
        }

    @classmethod
    def dataset_from_json(self, path):
        """
        Load a json file and format it applying Table() to every table.

        :param path: path of the json file
        :return: a dict of Tables
        """
        data = load_json(path)
        return self.format_dataset(data)

    # Save and load functions
    @staticmethod
    def from_pandas(df):
        """
        Create a table from a pandas dataframe.

        :param df: a pandas dataframe
        :return: a Table
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Pandas is not present in your system. Try: pip install pandas"
            )
        return Table(pd.DataFrame(df).to_dict(orient="records"))

    def to_pandas(self):
        """
        Create a pandas dataframe from a table.

        :return: a pandas dataframe.
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

    @staticmethod
    def pk_load(file):
        """
        Load a Table from a pickle file.

        :param file: path of the file
        :return: The table
        """
        import pickle as pk

        if ".pickle" not in file:
            file = file + ".pickle"
        with open(file, "rb") as f:
            data = pk.load(f)
        return Table(data)

    def to_json(self, path):
        """
        Export the table to a json file.
        If any column name is a number, transform it into string.

        :param path: path of the json file
        :return: nothing
        """
        write_json(str_key_tl(self), path)

    @staticmethod
    def from_json(path):
        """
        Create a table from a json file.

        :param path: path to json file
        :return: a Table containing the data.
        """
        return Table(load_json(path))

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
