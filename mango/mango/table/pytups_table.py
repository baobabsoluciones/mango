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
    group_by,
    auto_join,
)
from .table_tools import is_subset


class Table(TupList):
    """
    Enhanced table class extending TupList with data manipulation capabilities.

    A Table is a list of dictionaries that provides a rich set of methods for
    data manipulation, filtering, joining, and transformation operations.
    Inspired by R's dplyr and data.table packages.

    :param data: Initial data as a list of dictionaries or None for empty table
    :type data: list[dict], optional
    :param check: Whether to validate that all rows are dictionaries
    :type check: bool

    Example:
        >>> # Create empty table
        >>> table = Table()
        >>>
        >>> # Create table from list of dicts
        >>> data = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
        >>> table = Table(data)
        >>> print(table)
        Table (2 rows, 2 columns):
        0 {'name': 'Alice', 'age': 30}
        1 {'name': 'Bob', 'age': 25}
    """

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
        Apply a function to each element (row) of the table.

        Maps the given function to each row in the table, returning a new
        table with the transformed values.

        :param func: Function to apply to each row
        :type func: Callable
        :param args: Additional positional arguments for the function
        :param kwargs: Additional keyword arguments for the function
        :return: New table with transformed values
        :rtype: Table

        Example:
            >>> table = Table([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
            >>> result = table.vapply(lambda row: {"sum": row["x"] + row["y"]})
            >>> print(result)
            [{'sum': 3}, {'sum': 7}]
        """
        return Table(super().vapply(func, *args, **kwargs))

    def kapply(self, func, *args, **kwargs) -> "Table":
        """
        Apply a function to each key (column name) of the table.

        Maps the given function to each column name in the table, returning a new
        table with transformed column names.

        :param func: Function to apply to each column name
        :type func: Callable
        :param args: Additional positional arguments for the function
        :param kwargs: Additional keyword arguments for the function
        :return: New table with transformed column names
        :rtype: Table

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> result = table.kapply(lambda key: key.upper())
            >>> print(result)
            [{'NAME': 'Alice', 'AGE': 30}]
        """
        return Table(super().kapply(func, *args, **kwargs))

    def kvapply(self, func, *args, **kwargs) -> "Table":
        """
        Apply a function to each element with its index.

        Maps the given function to each row in the table along with its index,
        returning a new table with transformed values.

        :param func: Function to apply to each (index, row) pair
        :type func: Callable
        :param args: Additional positional arguments for the function
        :param kwargs: Additional keyword arguments for the function
        :return: New table with transformed values
        :rtype: Table

        Example:
            >>> table = Table([{"x": 1}, {"x": 2}])
            >>> result = table.kvapply(lambda idx, row: {"index": idx, "value": row["x"]})
            >>> print(result)
            [{'index': 0, 'value': 1}, {'index': 1, 'value': 2}]
        """
        return Table(super().kvapply(func, *args, **kwargs))

    def copy_shallow(self) -> "Table":
        """
        Create a shallow copy of the table.

        Creates a new table with the same structure but shares references
        to the same row objects. Changes to row contents will affect both tables.

        :return: New table with shallow copy of the data
        :rtype: Table

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> copy = table.copy_shallow()
            >>> copy[0]["age"] = 31
            >>> print(table[0]["age"])  # Also changed to 31
            31
        """
        return Table(super().copy_shallow())

    def copy_deep(self) -> "Table":
        """
        Create a deep copy of the table.

        Creates a new table with completely independent copies of all data.
        Changes to the original table will not affect the copy.

        :return: New table with deep copy of the data
        :rtype: Table

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> copy = table.copy_deep()
            >>> copy[0]["age"] = 31
            >>> print(table[0]["age"])  # Still 30
            30
        """
        return Table(super().copy_deep())

    def vfilter(self, function: Callable) -> "Table":
        """
        Filter rows based on a condition (internal method).

        Keeps only the rows for which the provided function returns True.
        This is the internal implementation used by the public filter method.

        :param function: Function that takes a row (dict) and returns a boolean
        :type function: Callable[[dict], bool]
        :return: New table containing only rows that satisfy the condition
        :rtype: Table
        """
        return Table(super().vfilter(function))

    def unique(self) -> "Table":
        """
        Remove duplicate rows from the table.

        Eliminates rows with identical values across all columns.
        When duplicates are found, the first occurrence is kept.

        :return: New table with duplicate rows removed
        :rtype: Table
        :raises NotImplementedError: If table structure is not compatible

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "Bob", "age": 25},
            ...     {"name": "Alice", "age": 30}  # Duplicate
            ... ])
            >>> result = table.unique()
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
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
        Remove duplicate rows from the table (alternative implementation).

        Alternative implementation of unique() that eliminates rows with
        identical values across all columns. When duplicates are found,
        the first occurrence is kept.

        :return: New table with duplicate rows removed
        :rtype: Table
        :raises NotImplementedError: If table structure is not compatible

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "Bob", "age": 25},
            ...     {"name": "Alice", "age": 30}  # Duplicate
            ... ])
            >>> result = table.unique2()
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
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
        Sort the table according to a custom key function.

        Sorts the table using a custom key function. This method is limited
        to simple data types and may not work with complex table structures.

        :param kwargs: Arguments for the sorted function
        :type kwargs: dict
        :return: New table with sorted rows
        :rtype: Table
        :raises NotImplementedError: If table structure is not compatible

        Example:
            >>> table = Table([{"value": 3}, {"value": 1}, {"value": 2}])
            >>> result = table.sorted(key=lambda x: x["value"])
            >>> print(result)
            [{'value': 1}, {'value': 2}, {'value': 3}]
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
        Extract values from specified columns of the table.

        Returns a TupList containing tuples of values from the specified columns.
        Each tuple represents the values from one row for the selected columns.

        :param args: Names of columns to extract
        :type args: str
        :param use_numpy: Whether to use numpy methods for extraction
        :type use_numpy: bool
        :return: TupList of tuples containing column values
        :rtype: TupList

        Example:
            >>> table = Table([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
            >>> result = table.take("a", "b")
            >>> print(result)
            [(1, 2), (3, 4)]
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
        Display specific rows of the table as a formatted string.

        Shows the specified row or range of rows with their indices.
        Used internally by the string representation of the table.

        :param n1: Starting row index
        :type n1: int
        :param n2: Ending row index (if None, shows only row n1)
        :type n2: int, optional
        :return: Formatted string showing the specified rows
        :rtype: str

        Example:
            >>> table = Table([{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}])
            >>> result = table.show_rows(0, 2)
            >>> print(result)
            0 {'name': 'Alice'}
            1 {'name': 'Bob'}
        """
        if n2 is None:
            n2 = n1 + 1
        return "".join(f"{i} {self[i]}\n" for i in range(n1, n2))

    def head(self, n=10) -> "Table":
        """
        Return the first n rows of the table.

        Creates a new table containing only the first n rows.
        If the table has fewer than n rows, returns the entire table.

        :param n: Number of rows to return (default: 10)
        :type n: int
        :return: New table with the first n rows
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"},
            ...     {"name": "David"}, {"name": "Eve"}
            ... ])
            >>> result = table.head(3)
            >>> print(result)
            [{'name': 'Alice'}, {'name': 'Bob'}, {'name': 'Charlie'}]
        """
        if self.len() < n:
            return self
        else:
            return self[:n]

    def peek(self, n=3, name=None) -> "Table":
        """
        Display a preview of the table showing first, middle, and last rows.

        Shows the first n rows, middle n rows, and last n rows of the table.
        Useful for getting an overview of large tables without printing everything.

        :param n: Number of rows to show in each section (default: 3)
        :type n: int
        :param name: Optional name or message to display with the table
        :type name: str, optional
        :return: The original table (unchanged)
        :rtype: Table

        Example:
            >>> table = Table([{"id": i, "value": i*2} for i in range(10)])
            >>> table.peek(2, "Sample Data")
            Sample Data: Table (10 rows, 2 columns):
            0 {'id': 0, 'value': 0}
            1 {'id': 1, 'value': 2}
            ...
            4 {'id': 4, 'value': 8}
            5 {'id': 5, 'value': 10}
            ...
            8 {'id': 8, 'value': 16}
            9 {'id': 9, 'value': 18}
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
        Add or modify columns in the table.

        Creates new columns or modifies existing ones using various methods:
        - Single values applied to all rows
        - Lists of values for each row
        - Functions that operate on row data

        Note: All changes are applied to the original table structure and
        do not take into account other changes made in the same call.

        :param kwargs: Named arguments with column names and their values
        :type kwargs: dict
        :return: New table with modified columns
        :rtype: Table

        Example:
            >>> table = Table([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
            >>> result = table.mutate(
            ...     z=10,  # Constant value
            ...     sum=lambda row: row["x"] + row["y"],  # Function
            ...     product=[2, 12]  # List of values
            ... )
            >>> print(result)
            [{'x': 1, 'y': 2, 'z': 10, 'sum': 3, 'product': 2},
             {'x': 3, 'y': 4, 'z': 10, 'sum': 7, 'product': 12}]
        """
        return Table(mutate(self, **kwargs))

    def group_by(self, col) -> SuperDict:
        """
        Group rows of the table by specified column values.

        Groups the table rows based on the values in the specified column(s).
        Returns a SuperDict where keys are the unique values and values are
        lists of rows that have that value.

        :param col: Column name or list of column names to group by
        :type col: str or list[str]
        :return: SuperDict with grouped data
        :rtype: SuperDict

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "city": "Madrid"},
            ...     {"name": "Bob", "city": "Barcelona"},
            ...     {"name": "Charlie", "city": "Madrid"}
            ... ])
            >>> result = table.group_by("city")
            >>> print(result)
            {'Madrid': [{'name': 'Alice', 'city': 'Madrid'}, {'name': 'Charlie', 'city': 'Madrid'}],
             'Barcelona': [{'name': 'Bob', 'city': 'Barcelona'}]}
        """
        return group_by(self, col)

    def group_mutate(self, group_by, **func) -> "Table":
        """
        Group by specified columns and apply functions to other columns.

        Groups the table by the specified columns and applies aggregation
        functions to the remaining columns. Similar to SQL GROUP BY with
        aggregate functions.

        :param group_by: Column name or list of column names to group by
        :type group_by: str or list[str]
        :param func: Functions to apply to columns (e.g., a=sum, b=mean)
        :type func: dict[str, Callable]
        :return: New table with grouped and aggregated data
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"city": "Madrid", "sales": 100},
            ...     {"city": "Madrid", "sales": 150},
            ...     {"city": "Barcelona", "sales": 200}
            ... ])
            >>> result = table.group_mutate("city", sales=sum)
            >>> print(result)
            [{'city': 'Madrid', 'sales': 250}, {'city': 'Barcelona', 'sales': 200}]
        """
        return Table(group_mutate(self, group_by, **func))

    def sum_all(self, group_by=None) -> "Table":
        """
        Group by specified columns and sum all numeric columns.

        Groups the table by the specified columns and sums all numeric
        columns in each group. Non-numeric columns are ignored.

        :param group_by: Column name or list of column names to group by
        :type group_by: str or list[str], optional
        :return: New table with grouped and summed data
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"category": "A", "value1": 10, "value2": 20},
            ...     {"category": "A", "value1": 15, "value2": 25},
            ...     {"category": "B", "value1": 5, "value2": 10}
            ... ])
            >>> result = table.sum_all("category")
            >>> print(result)
            [{'category': 'A', 'value1': 25, 'value2': 45},
             {'category': 'B', 'value1': 5, 'value2': 10}]
        """
        return Table(sum_all(self, group_by))

    def summarise(self, group_by=None, default: Callable = None, **func) -> "Table":
        """
        Group by specified columns and apply aggregation functions.

        Groups the table by specified columns and applies custom aggregation
        functions to other columns. More flexible than group_mutate as it
        allows specifying a default function for non-explicitly handled columns.

        :param group_by: Column name or list of column names to group by
        :type group_by: str or list[str], optional
        :param default: Default function to apply to columns not explicitly specified
        :type default: Callable, optional
        :param func: Functions to apply to specific columns
        :type func: dict[str, Callable]
        :return: New table with grouped and summarized data
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"category": "A", "value": 10, "count": 1},
            ...     {"category": "A", "value": 15, "count": 2},
            ...     {"category": "B", "value": 5, "count": 1}
            ... ])
            >>> result = table.summarise("category", value=sum, count=sum)
            >>> print(result)
            [{'category': 'A', 'value': 25, 'count': 3},
             {'category': 'B', 'value': 5, 'count': 1}]
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
        Join two tables using various join types.

        Performs table joins inspired by R's dplyr join functions. Supports
        different join types and flexible column matching strategies.

        :param table2: Second table to join with
        :type table2: Table or list[dict]
        :param by: Column specification for joining
        :type by: list, dict, or None
        :param suffix: Suffixes for disambiguating column names
        :type suffix: list[str], optional
        :param jtype: Type of join ("full", "left", "right", "inner")
        :type jtype: str
        :param empty: Value to use for empty cells created by the join
        :param if_empty_table_1: Replacement if table 1 is empty
        :param if_empty_table_2: Replacement if table 2 is empty
        :return: New table containing the joined data
        :rtype: Table

        Example:
            >>> table1 = Table([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> table2 = Table([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
            >>> result = table1.join(table2, by="id", jtype="left")
            >>> print(result)
            [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': None}]
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
        Perform a left join with another table.

        Returns all rows from the left table (self) and matching rows from
        the right table (table2). Rows from the left table without matches
        will have None values for columns from the right table.

        :param table2: Second table to join with
        :type table2: Table or list[dict]
        :param by: Column specification for joining
        :type by: list, dict, or None
        :param suffix: Suffixes for disambiguating column names
        :type suffix: list[str], optional
        :param empty: Value to use for empty cells created by the join
        :param if_empty_table_1: Replacement if table 1 is empty
        :param if_empty_table_2: Replacement if table 2 is empty
        :return: New table containing the left join result
        :rtype: Table

        Example:
            >>> table1 = Table([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> table2 = Table([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
            >>> result = table1.left_join(table2, by="id")
            >>> print(result)
            [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': None}]
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
        Perform a right join with another table.

        Returns all rows from the right table (table2) and matching rows from
        the left table (self). Rows from the right table without matches
        will have None values for columns from the left table.

        :param table2: Second table to join with
        :type table2: Table or list[dict]
        :param by: Column specification for joining
        :type by: list, dict, or None
        :param suffix: Suffixes for disambiguating column names
        :type suffix: list[str], optional
        :param empty: Value to use for empty cells created by the join
        :param if_empty_table_1: Replacement if table 1 is empty
        :param if_empty_table_2: Replacement if table 2 is empty
        :return: New table containing the right join result
        :rtype: Table

        Example:
            >>> table1 = Table([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> table2 = Table([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
            >>> result = table1.right_join(table2, by="id")
            >>> print(result)
            [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 3, 'name': None, 'age': 25}]
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
        Perform a full outer join with another table.

        Returns all rows from both tables, with None values where there are
        no matches. This is the default join type and combines left and right joins.

        :param table2: Second table to join with
        :type table2: Table or list[dict]
        :param by: Column specification for joining
        :type by: list, dict, or None
        :param suffix: Suffixes for disambiguating column names
        :type suffix: list[str], optional
        :param empty: Value to use for empty cells created by the join
        :param if_empty_table_1: Replacement if table 1 is empty
        :param if_empty_table_2: Replacement if table 2 is empty
        :return: New table containing the full join result
        :rtype: Table

        Example:
            >>> table1 = Table([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> table2 = Table([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
            >>> result = table1.full_join(table2, by="id")
            >>> print(result)
            [{'id': 1, 'name': 'Alice', 'age': 30},
             {'id': 2, 'name': 'Bob', 'age': None},
             {'id': 3, 'name': None, 'age': 25}]
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
        Perform an inner join with another table.

        Returns only rows that have matching values in both tables.
        Rows without matches in either table are excluded from the result.

        :param table2: Second table to join with
        :type table2: Table or list[dict]
        :param by: Column specification for joining
        :type by: list, dict, or None
        :param suffix: Suffixes for disambiguating column names
        :type suffix: list[str], optional
        :param empty: Value to use for empty cells created by the join
        :param if_empty_table_1: Replacement if table 1 is empty
        :param if_empty_table_2: Replacement if table 2 is empty
        :return: New table containing only matching rows
        :rtype: Table

        Example:
            >>> table1 = Table([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> table2 = Table([{"id": 1, "age": 30}, {"id": 3, "age": 25}])
            >>> result = table1.inner_join(table2, by="id")
            >>> print(result)
            [{'id': 1, 'name': 'Alice', 'age': 30}]
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
        Join a table with itself to create combinations.

        Performs a self-join to create all possible combinations of rows.
        Useful for creating Cartesian products or finding relationships
        within the same table.

        :param by: Column specification for the self-join
        :type by: list, dict, or None
        :param suffix: Suffixes to add to column names to distinguish them
        :type suffix: list[str], optional
        :param empty: Value to use for empty cells created by the join
        :return: New table with all combinations
        :rtype: Table

        Example:
            >>> table = Table([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> result = table.auto_join()
            >>> print(result)
            [{'id': 1, 'name': 'Alice', 'id_2': 1, 'name_2': 'Alice'},
             {'id': 1, 'name': 'Alice', 'id_2': 2, 'name_2': 'Bob'},
             {'id': 2, 'name': 'Bob', 'id_2': 1, 'name_2': 'Alice'},
             {'id': 2, 'name': 'Bob', 'id_2': 2, 'name_2': 'Bob'}]
        """
        return Table(auto_join(self, by=by, suffix=suffix, empty=empty))

    def select(self, *args) -> "Table":
        """
        Select specific columns from the table.

        Creates a new table containing only the specified columns.
        Maintains the original row order.

        :param args: Names of columns to select
        :type args: str
        :return: New table with only the selected columns
        :rtype: Table
        :raises KeyError: If any specified column doesn't exist

        Example:
            >>> table = Table([
            ...     {"id": 1, "name": "Alice", "age": 30, "city": "Madrid"},
            ...     {"id": 2, "name": "Bob", "age": 25, "city": "Barcelona"}
            ... ])
            >>> result = table.select("name", "age")
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        """
        return Table(select(self, *args))

    def drop(self, *args) -> "Table":
        """
        Remove specific columns from the table.

        Creates a new table with the specified columns removed.
        Maintains the original row order.

        :param args: Names of columns to remove
        :type args: str
        :return: New table without the specified columns
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"id": 1, "name": "Alice", "age": 30, "city": "Madrid"},
            ...     {"id": 2, "name": "Bob", "age": 25, "city": "Barcelona"}
            ... ])
            >>> result = table.drop("id", "city")
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        """
        return Table(drop(self, *args))

    def rename(self, **kwargs) -> "Table":
        """
        Rename columns in the table.

        Changes column names using a mapping of old names to new names.
        Maintains the original row order and data.

        :param kwargs: Mapping of old column names to new names
        :type kwargs: dict[str, str]
        :return: New table with renamed columns
        :rtype: Table
        :raises KeyError: If any old column name doesn't exist

        Example:
            >>> table = Table([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> result = table.rename(id="user_id", name="full_name")
            >>> print(result)
            [{'user_id': 1, 'full_name': 'Alice'}, {'user_id': 2, 'full_name': 'Bob'}]
        """
        return Table(rename(self, **kwargs))

    def filter(self, func) -> "Table":
        """
        Filter rows based on a condition.

        Keeps only the rows for which the provided function returns True.
        Returns an empty table if the original table is empty.

        :param func: Function that takes a row (dict) and returns a boolean
        :type func: Callable[[dict], bool]
        :return: New table containing only rows that satisfy the condition
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "Bob", "age": 25},
            ...     {"name": "Charlie", "age": 35}
            ... ])
            >>> result = table.filter(lambda row: row["age"] > 28)
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Charlie', 'age': 35}]
        """
        if not self.len():
            return self

        return Table(self.vfilter(func))

    def get_col_names(self, fast=False) -> "list":
        """
        Get the names of all columns in the table.

        Returns a list of column names. By default, scans all rows to ensure
        all possible columns are included. Use fast=True for better performance
        if you're certain the first row contains all columns.

        :param fast: If True, only check the first row for column names
        :type fast: bool
        :return: List of column names
        :rtype: list[str]
        :raises IndexError: If table is empty

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
            >>> columns = table.get_col_names()
            >>> print(columns)
            ['name', 'age']
        """
        return get_col_names(self, fast)

    def to_columns(self) -> "SuperDict":
        """
        Convert table to column-oriented format.

        Transforms the table from row-oriented (list of dicts) to column-oriented
        (dict of lists) format. Each column becomes a key with a list of values.

        :return: SuperDict with column names as keys and lists of values
        :rtype: SuperDict

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "Bob", "age": 25}
            ... ])
            >>> result = table.to_columns()
            >>> print(result)
            {'name': ['Alice', 'Bob'], 'age': [30, 25]}
        """
        if self.len() == 0:
            return SuperDict()
        table = self.replace_empty(None)
        return SuperDict({col: table.take(col) for col in self.get_col_names()})

    @classmethod
    def from_columns(cls, dct) -> "Table":
        """
        Create a table from a column-oriented dictionary.

        Transforms a dictionary of lists (column-oriented) into a table
        (row-oriented list of dictionaries).

        :param dct: Dictionary with column names as keys and lists of values
        :type dct: dict[str, list]
        :return: New table with row-oriented data
        :rtype: Table

        Example:
            >>> data = {"name": ["Alice", "Bob"], "age": [30, 25]}
            >>> result = Table.from_columns(data)
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        """
        return cls(to_dictlist(dct))

    def get_index(self, cond) -> "list":
        """
        Get indices of rows that satisfy a condition.

        Returns a list of row indices where the condition function returns True.
        Useful for identifying which rows match specific criteria.

        :param cond: Function that takes a row (dict) and returns a boolean
        :type cond: Callable[[dict], bool]
        :return: List of row indices that satisfy the condition
        :rtype: list[int]

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "Bob", "age": 25},
            ...     {"name": "Charlie", "age": 35}
            ... ])
            >>> indices = table.get_index(lambda row: row["age"] > 28)
            >>> print(indices)
            [0, 2]
        """
        return [i for i, v in enumerate(self) if cond(v)]

    def replace(self, replacement=None, to_replace=None, fast=False) -> "Table":
        """
        Replace specific values in the table.

        Replaces specified values with new values in the table. Can replace
        values across all columns or target specific columns.

        :param replacement: New values to use as replacements
        :type replacement: any or dict[str, any]
        :param to_replace: Values to be replaced
        :type to_replace: any or dict[str, any]
        :param fast: If True, assume first row contains all columns
        :type fast: bool
        :return: New table with replaced values
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"age": 25, "city": "Madrid"},
            ...     {"age": 30, "city": "Barcelona"}
            ... ])
            >>> result = table.replace(
            ...     replacement={"age": 35, "city": "Paris"},
            ...     to_replace={"age": 25, "city": "Madrid"}
            ... )
            >>> print(result)
            [{'age': 35, 'city': 'Paris'}, {'age': 30, 'city': 'Barcelona'}]
        """
        return Table(replace(self, replacement, to_replace, fast))

    def replace_empty(self, replacement=None, fast=False) -> "Table":
        """
        Replace empty values in the table.

        Replaces empty values (None, empty strings, etc.) with specified
        replacement values. Can use different replacements for different columns.

        :param replacement: Values to use for replacing empty values
        :type replacement: any or dict[str, any]
        :param fast: If True, assume first row contains all columns
        :type fast: bool
        :return: New table with empty values replaced
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": None},
            ...     {"name": "", "age": 25}
            ... ])
            >>> result = table.replace_empty(replacement={"name": "Unknown", "age": 0})
            >>> print(result)
            [{'name': 'Alice', 'age': 0}, {'name': 'Unknown', 'age': 25}]
        """
        return Table(replace_empty(self, replacement, fast))

    def replace_nan(self, replacement=None) -> "Table":
        """
        Replace NaN values in the table.

        Replaces NaN (Not a Number) values with specified replacement values.
        Useful for cleaning numeric data with missing values.

        :param replacement: Values to use for replacing NaN values
        :type replacement: any or dict[str, any]
        :return: New table with NaN values replaced
        :rtype: Table

        Example:
            >>> import math
            >>> table = Table([
            ...     {"value": 10.5, "score": math.nan},
            ...     {"value": math.nan, "score": 85.0}
            ... ])
            >>> result = table.replace_nan(replacement=0)
            >>> print(result)
            [{'value': 10.5, 'score': 0}, {'value': 0, 'score': 85.0}]
        """
        return Table(replace_nan(self, replacement))

    def pivot_longer(self, cols, names_to="variable", value_to="value") -> "Table":
        """
        Transform table from wide to long format.

        "Lengthens" data by increasing the number of rows and decreasing
        the number of columns. The inverse transformation of pivot_wider().

        :param cols: List of column names to pivot
        :type cols: list[str]
        :param names_to: Name for the new column containing variable names
        :type names_to: str
        :param value_to: Name for the new column containing values
        :type value_to: str
        :return: New table in long format
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"id": 1, "Q1": 100, "Q2": 150, "Q3": 200},
            ...     {"id": 2, "Q1": 120, "Q2": 180, "Q3": 220}
            ... ])
            >>> result = table.pivot_longer(["Q1", "Q2", "Q3"], "quarter", "sales")
            >>> print(result)
            [{'id': 1, 'quarter': 'Q1', 'sales': 100},
             {'id': 1, 'quarter': 'Q2', 'sales': 150},
             {'id': 1, 'quarter': 'Q3', 'sales': 200},
             {'id': 2, 'quarter': 'Q1', 'sales': 120},
             {'id': 2, 'quarter': 'Q2', 'sales': 180},
             {'id': 2, 'quarter': 'Q3', 'sales': 220}]
        """
        return Table(pivot_longer(self, cols, names_to, value_to))

    def pivot_wider(
        self, names_from="variable", value_from="value", id_cols=None, values_fill=None
    ) -> "Table":
        """
        Transform table from long to wide format.

        "Widens" data by increasing the number of columns and decreasing
        the number of rows. The inverse transformation of pivot_longer().

        :param names_from: Name of the column containing variable names
        :type names_from: str
        :param value_from: Name of the column containing values
        :type value_from: str
        :param id_cols: Columns that uniquely identify each observation
        :type id_cols: list[str], optional
        :param values_fill: Value or dict to fill missing values
        :type values_fill: any or dict, optional
        :return: New table in wide format
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"id": 1, "quarter": "Q1", "sales": 100},
            ...     {"id": 1, "quarter": "Q2", "sales": 150},
            ...     {"id": 2, "quarter": "Q1", "sales": 120}
            ... ])
            >>> result = table.pivot_wider("quarter", "sales", "id")
            >>> print(result)
            [{'id': 1, 'Q1': 100, 'Q2': 150}, {'id': 2, 'Q1': 120, 'Q2': None}]
        """
        return Table(pivot_wider(self, names_from, value_from, id_cols, values_fill))

    def drop_empty(self, cols=None, fast=False) -> "Table":
        """
        Remove rows with empty values in specified columns.

        Drops rows where the specified columns contain empty values
        (None, empty strings, etc.).

        :param cols: Column name(s) to check for empty values
        :type cols: str or list[str], optional
        :param fast: If True, assume first row contains all columns
        :type fast: bool
        :return: New table with empty rows removed
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "", "age": 25},
            ...     {"name": "Bob", "age": None}
            ... ])
            >>> result = table.drop_empty("name")
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': None}]
        """
        return Table(drop_empty(self, cols, fast))

    def lag_col(self, col, i=1, replace=False) -> "Table":
        """
        Create a lagged version of a column.

        Shifts the values of a column by a specified number of steps.
        Useful for time series analysis and creating features from previous values.

        :param col: Name of the column to lag
        :type col: str
        :param i: Number of steps to lag (default: 1)
        :type i: int
        :param replace: If True, replace original column; if False, create new column
        :type replace: bool
        :return: New table with lagged column
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"date": "2023-01", "sales": 100},
            ...     {"date": "2023-02", "sales": 150},
            ...     {"date": "2023-03", "sales": 200}
            ... ])
            >>> result = table.lag_col("sales", 1)
            >>> print(result)
            [{'date': '2023-01', 'sales': 100, 'lag_sales_1': None},
             {'date': '2023-02', 'sales': 150, 'lag_sales_1': 100},
             {'date': '2023-03', 'sales': 200, 'lag_sales_1': 150}]
        """
        return Table(lag_col(self, col, i, replace))

    def distinct(self, columns) -> "Table":
        """
        Keep only unique combinations of values in specified columns.

        Removes duplicate rows based on the values in the specified columns.
        When duplicates are found, the first occurrence is kept.

        :param columns: Column name(s) to check for uniqueness
        :type columns: str or list[str]
        :return: New table with duplicate rows removed
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "city": "Madrid"},
            ...     {"name": "Bob", "city": "Barcelona"},
            ...     {"name": "Alice", "city": "Madrid"}  # Duplicate
            ... ])
            >>> result = table.distinct("name")
            >>> print(result)
            [{'name': 'Alice', 'city': 'Madrid'}, {'name': 'Bob', 'city': 'Barcelona'}]
        """
        return Table(distinct(self, columns))

    def order_by(self, columns, reverse=False) -> "Table":
        """
        Sort the table by specified columns.

        Reorders the table rows based on the values in the specified columns.
        Supports both ascending and descending order.

        :param columns: Column name(s) to sort by
        :type columns: str or list[str]
        :param reverse: If True, sort in descending order
        :type reverse: bool
        :return: New table with sorted rows
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"name": "Charlie", "age": 35},
            ...     {"name": "Alice", "age": 30},
            ...     {"name": "Bob", "age": 25}
            ... ])
            >>> result = table.order_by("age")
            >>> print(result)
            [{'name': 'Bob', 'age': 25},
             {'name': 'Alice', 'age': 30},
             {'name': 'Charlie', 'age': 35}]
        """
        return Table(order_by(self, columns=columns, reverse=reverse))

    def drop_nested(self) -> "Table":
        """
        Remove columns containing nested data structures.

        Drops columns that contain nested values (dictionaries, lists, or tuples).
        Assumes homogeneous table structure and checks only the first row.

        :return: New table with nested columns removed
        :rtype: Table

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "age": 30, "hobbies": ["reading", "swimming"]},
            ...     {"name": "Bob", "age": 25, "hobbies": ["gaming"]}
            ... ])
            >>> result = table.drop_nested()
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
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
        Validate that the table is a list of dictionaries.

        Checks that the table structure is correct (list of dictionaries).
        Raises TypeError if the structure is invalid.

        :return: None
        :rtype: None
        :raises TypeError: If table is not a list of dictionaries

        Example:
            >>> table = Table([{"name": "Alice"}, {"name": "Bob"}])
            >>> table.check_type()  # No error
            >>>
            >>> invalid_table = Table([1, 2, 3])
            >>> invalid_table.check_type()  # Raises TypeError
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
        Check if a specific row is a dictionary.

        Internal method to validate that a row at the specified index
        is a dictionary.

        :param i: Row index to check
        :type i: int
        :return: True if the row is a dictionary
        :rtype: bool
        """
        return isinstance(self[i], dict)

    def _all_rows_are_dict(self):
        """
        Check if all rows in the table are dictionaries.

        Internal method to validate that all rows in the table
        are dictionaries.

        :return: True if all rows are dictionaries
        :rtype: bool
        """
        return all(self._row_is_dict(i) for i in range(len(self)))

    def to_set2(self, columns) -> "TupList":
        """
        Create a list of unique combinations from specified columns.

        Extracts unique combinations of values from the specified columns
        and returns them as a TupList.

        :param columns: Column name(s) to extract unique combinations from
        :type columns: str or list[str]
        :return: TupList containing unique combinations
        :rtype: TupList
        :raises KeyError: If specified columns don't exist in the table

        Example:
            >>> table = Table([
            ...     {"name": "Alice", "city": "Madrid"},
            ...     {"name": "Bob", "city": "Barcelona"},
            ...     {"name": "Alice", "city": "Madrid"}  # Duplicate
            ... ])
            >>> result = table.to_set2(["name", "city"])
            >>> print(result)
            [('Alice', 'Madrid'), ('Bob', 'Barcelona')]
        """
        if len(self) == 0:
            return TupList()
        table_col = set(self[0].keys())
        if not is_subset(columns, table_col):
            raise KeyError(f"key(s) {columns} are not in table.")
        return self.take(columns).unique2()

    def to_param(self, keys, value, is_list=False) -> "SuperDict":
        """
        Create a parameter dictionary from specified columns.

        Creates a SuperDict using specified columns as keys and values.
        Useful for creating lookup dictionaries or parameter mappings.

        :param keys: Column name(s) to use as dictionary keys
        :type keys: str or list[str]
        :param value: Column name to use as dictionary values
        :type value: str
        :param is_list: If True, values can be lists; if False, expects unique keys
        :type is_list: bool
        :return: SuperDict with keys and values from specified columns
        :rtype: SuperDict
        :raises KeyError: If specified columns don't exist
        :raises ValueError: If keys are not unique and is_list=False

        Example:
            >>> table = Table([
            ...     {"id": 1, "name": "Alice", "age": 30},
            ...     {"id": 2, "name": "Bob", "age": 25}
            ... ])
            >>> result = table.to_param("id", "name")
            >>> print(result)
            {1: 'Alice', 2: 'Bob'}
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
        Check if combinations of values in specified columns are unique.

        Determines whether the combination of values in the specified columns
        forms a unique key for each row.

        :param columns: Column name(s) to check for uniqueness
        :type columns: str or list[str]
        :return: True if all combinations are unique
        :rtype: bool

        Example:
            >>> table = Table([
            ...     {"id": 1, "name": "Alice"},
            ...     {"id": 2, "name": "Bob"},
            ...     {"id": 1, "name": "Charlie"}  # Duplicate id
            ... ])
            >>> print(table.is_unique("id"))
            False
            >>> print(table.is_unique("name"))
            True
        """
        len_unique = self.distinct(columns).len()
        return len_unique == self.len()

    def add_row(self, **kwargs) -> "Table":
        """
        Add a new row to the table.

        Adds a new row with the specified values. Missing columns are
        filled with None values to maintain table structure.

        :param kwargs: Column values in the format column_name=value
        :type kwargs: dict
        :return: New table with the added row
        :rtype: Table

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> result = table.add_row(name="Bob", age=25)
            >>> print(result)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
        """
        result = self + [{**kwargs}]
        return result.replace(replacement=None, to_replace=None)

    def rbind(self, table: list) -> "Table":
        """
        Bind another table by rows (row bind).

        Combines the current table with another table by stacking rows.
        Missing columns are filled with None values.

        :param table: Another table to bind (list of dicts or Table)
        :type table: list[dict] or Table
        :return: New table with combined rows
        :rtype: Table

        Example:
            >>> table1 = Table([{"name": "Alice", "age": 30}])
            >>> table2 = [{"name": "Bob", "city": "Madrid"}]
            >>> result = table1.rbind(table2)
            >>> print(result)
            [{'name': 'Alice', 'age': 30, 'city': None},
             {'name': 'Bob', 'age': None, 'city': 'Madrid'}]
        """
        return (self + Table(table)).replace(replacement=None, to_replace=None)

    def col_apply(self, columns, func: Callable, **kwargs) -> "Table":
        """
        Apply a function to specified columns.

        Applies a function to the values in specified columns, transforming
        the data according to the function logic.

        :param columns: Column name(s) to apply the function to
        :type columns: str or list[str]
        :param func: Function to apply to column values
        :type func: Callable
        :param kwargs: Additional keyword arguments for the function
        :return: New table with transformed columns
        :rtype: Table

        Example:
            >>> table = Table([{"value": 10}, {"value": 20}])
            >>> result = table.col_apply("value", lambda x: x * 2)
            >>> print(result)
            [{'value': 20}, {'value': 40}]
        """
        result = self
        for col in as_list(columns):
            result = result.mutate(**{col: lambda v: func(v[col], **kwargs)})
        return result

    @classmethod
    def format_dataset(cls, dataset) -> dict:
        """
        Convert dataset dictionary to use Table objects for list values.

        Processes a dataset dictionary, converting any list values to Table
        objects while keeping other values unchanged.

        :param dataset: Dictionary containing data in various formats
        :type dataset: dict
        :return: Dictionary with list values converted to Table objects
        :rtype: dict

        Example:
            >>> data = {
            ...     "users": [{"name": "Alice"}, {"name": "Bob"}],
            ...     "config": {"setting": "value"}
            ... }
            >>> result = Table.format_dataset(data)
            >>> print(type(result["users"]))
            <class 'mango.table.pytups_table.Table'>
        """
        return {
            k: cls(str_key_tl(v)) if isinstance(v, list) else v
            for k, v in dataset.items()
        }

    @classmethod
    def dataset_from_json(cls, path, **kwargs) -> dict:
        """
        Load a JSON file and convert list values to Table objects.

        Loads a JSON file and processes it using format_dataset to convert
        any list values to Table objects.

        :param path: Path to the JSON file
        :type path: str
        :param kwargs: Additional arguments for load_json
        :return: Dictionary with list values converted to Table objects
        :rtype: dict

        Example:
            >>> # Assuming data.json contains: {"users": [{"name": "Alice"}]}
            >>> result = Table.dataset_from_json("data.json")
            >>> print(type(result["users"]))
            <class 'mango.table.pytups_table.Table'>
        """
        data = load_json(path, **kwargs)
        return cls.format_dataset(data)

    # Save and load functions
    @classmethod
    def from_pandas(cls, df) -> "Table":
        """
        Create a Table from a pandas DataFrame.

        Converts a pandas DataFrame to a Table object, preserving all data
        and column names. Requires pandas to be installed.

        :param df: Pandas DataFrame to convert
        :type df: pandas.DataFrame
        :return: New Table object with the DataFrame data
        :rtype: Table
        :raises ImportError: If pandas is not installed

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [30, 25]})
            >>> table = Table.from_pandas(df)
            >>> print(table)
            [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
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
        Convert the Table to a pandas DataFrame.

        Creates a pandas DataFrame from the Table data, preserving all
        columns and rows. Requires pandas to be installed.

        :return: Pandas DataFrame with the table data
        :rtype: pandas.DataFrame
        :raises ImportError: If pandas is not installed

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
            >>> df = table.to_pandas()
            >>> print(df)
               name  age
            0  Alice   30
            1    Bob   25
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
        Save the table to a pickle file.

        Serializes the table object and saves it to a pickle file.
        Automatically adds .pickle extension if not present.

        :param file: Path to the pickle file
        :type file: str
        :return: None
        :rtype: None

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> table.pk_save("my_table")
            # Creates my_table.pickle file
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

        Deserializes a Table object from a pickle file.
        Automatically adds .pickle extension if not present.

        :param file: Path to the pickle file
        :type file: str
        :return: Table object loaded from the file
        :rtype: Table

        Example:
            >>> table = Table.pk_load("my_table")
            >>> print(table)
            [{'name': 'Alice', 'age': 30}]
        """
        import pickle as pk

        if ".pickle" not in file:
            file = file + ".pickle"
        with open(file, "rb") as f:
            data = pk.load(f)
        return cls(data)

    def to_json(self, path):
        """
        Export the table to a JSON file.

        Saves the table data to a JSON file. Numeric column names are
        automatically converted to strings for JSON compatibility.

        :param path: Path to the JSON file
        :type path: str
        :return: None
        :rtype: None

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> table.to_json("output.json")
            # Creates output.json with table data
        """
        write_json(str_key_tl(self), path)

    @classmethod
    def from_json(cls, path, **kwargs) -> "Table":
        """
        Create a table from a JSON file.

        Loads data from a JSON file and creates a Table object.
        The JSON file should contain a list of dictionaries.

        :param path: Path to the JSON file
        :type path: str
        :param kwargs: Additional arguments for load_json
        :return: New table with data from the JSON file
        :rtype: Table

        Example:
            >>> # Assuming data.json contains: [{"name": "Alice", "age": 30}]
            >>> table = Table.from_json("data.json")
            >>> print(table)
            [{'name': 'Alice', 'age': 30}]
        """
        return cls(load_json(path, **kwargs))

    def apply(self, func: Callable, *args, **kwargs):
        """
        Apply a function to the entire table.

        Passes the entire table as the first argument to the specified function.
        Useful for chaining custom functions that operate on the whole table.

        :param func: Function that takes the table as its first argument
        :type func: Callable
        :param args: Additional positional arguments for the function
        :param kwargs: Additional keyword arguments for the function
        :return: Result of the function call
        :rtype: any

        Example:
            >>> table = Table([{"value": 10}, {"value": 20}])
            >>> def double_table(t):
            ...     return t.mutate(value=lambda row: row["value"] * 2)
            >>> result = table.apply(double_table)
            >>> print(result)
            [{'value': 20}, {'value': 40}]
        """
        return func(self, *args, **kwargs)

    @classmethod
    def dataset_from_excel(cls, path, sheets=None) -> dict:
        """
        Load an Excel file and return a dictionary of Table objects.

        Reads an Excel file and converts each sheet to a Table object.
        Returns a dictionary with sheet names as keys and Table objects as values.

        :param path: Path to the Excel file
        :type path: str
        :param sheets: List of sheet names to read (all sheets if None)
        :type sheets: list[str], optional
        :return: Dictionary with sheet names and Table objects
        :rtype: dict

        Example:
            >>> # Assuming data.xlsx has sheets "users" and "orders"
            >>> result = Table.dataset_from_excel("data.xlsx")
            >>> print(list(result.keys()))
            ['users', 'orders']
        """
        data = load_excel_light(path, sheets)
        return cls.format_dataset(data)

    def to_excel(self, path, sheet_name=None):
        """
        Export the table to an Excel file.

        Saves the table data to an Excel file with the specified sheet name.
        If no sheet name is provided, uses "Sheet1" as default.

        :param path: Path to the Excel file
        :type path: str
        :param sheet_name: Name of the Excel sheet
        :type sheet_name: str, optional
        :return: None
        :rtype: None

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> table.to_excel("output.xlsx", "Users")
            # Creates output.xlsx with "Users" sheet
        """
        if sheet_name is None:
            sheet_name = "Sheet1"
        return write_excel_light(path, {sheet_name: self})

    @classmethod
    def from_csv(cls, path, sep=",", encoding=None) -> "Table":
        """
        Load a table from a CSV file.

        Reads a CSV file and creates a Table object from the data.
        Supports custom separators and encodings.

        :param path: Path to the CSV file
        :type path: str
        :param sep: Column separator (detected automatically if None)
        :type sep: str
        :param encoding: File encoding
        :type encoding: str, optional
        :return: New table with data from the CSV file
        :rtype: Table

        Example:
            >>> # Assuming data.csv contains: name,age\\nAlice,30\\nBob,25
            >>> table = Table.from_csv("data.csv")
            >>> print(table)
            [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]
        """
        data = load_csv_light(path, sep, encoding)
        return Table(data)

    def to_csv(self, path, sep=",", encoding=None):
        """
        Export the table to a CSV file.

        Saves the table data to a CSV file with the specified separator
        and encoding.

        :param path: Path to the CSV file
        :type path: str
        :param sep: Column separator for the CSV file
        :type sep: str
        :param encoding: File encoding
        :type encoding: str, optional
        :return: None
        :rtype: None

        Example:
            >>> table = Table([{"name": "Alice", "age": 30}])
            >>> table.to_csv("output.csv", sep=";")
            # Creates output.csv with semicolon separator
        """
        write_csv_light(path, self, sep, encoding)
