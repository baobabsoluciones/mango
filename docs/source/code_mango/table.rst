Table
------------

The **Table** class is used to represent and work on a table or dataframe as a list of dict. It derives from the pytups.Tuplist class and provides powerful data manipulation capabilities similar to pandas DataFrames but with a more lightweight and flexible approach.

Key Features
============

* Lightweight data structure based on Python dictionaries
* Powerful data manipulation methods inspired by R's dplyr
* Easy conversion to/from JSON format
* Integration with pandas DataFrames
* Support for various join operations
* Flexible data reshaping capabilities

Basic Example
=============

For example the following table:

======= =======
   a       b
======= =======
   1     False
   2     True
   3     False
======= =======

will be represented as:

.. code-block::

    [{"a": 1, "b": False}, {"a": 2, "b": True}, {"a": 3, "b": False}]

Common Operations
=================

Data Manipulation
~~~~~~~~~~~~~~~~~

The Table class provides various methods for data manipulation:

* **mutate()**: Add or modify columns
* **select()**: Choose specific columns
* **drop()**: Remove columns
* **filter()**: Filter rows based on conditions
* **group_by()**: Group data by columns
* **summarise()**: Aggregate data after grouping

.. code-block::

    # Example of data manipulation
    table = Table([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    
    # Add a new column 'c' as sum of 'a' and 'b'
    result = table.mutate(c=lambda v: v["a"] + v["b"])
    # Result: [{"a": 1, "b": 2, "c": 3}, {"a": 3, "b": 4, "c": 7}]

Join Operations
~~~~~~~~~~~~~~~

The class supports various types of joins similar to SQL:

* **left_join()**: Keep all rows from the left table
* **right_join()**: Keep all rows from the right table
* **inner_join()**: Keep only matching rows
* **full_join()**: Keep all rows from both tables
* **auto_join()**: Join a table with itself

.. code-block::

    # Example of join operation
    table1 = Table([{"id": 1, "value": "A"}, {"id": 2, "value": "B"}])
    table2 = Table([{"id": 1, "data": 100}, {"id": 3, "data": 300}])
    
    result = table1.left_join(table2, by="id")
    # Result: [{"id": 1, "value": "A", "data": 100}, {"id": 2, "value": "B", "data": None}]

Data Reshaping
~~~~~~~~~~~~~~~

Support for reshaping data between wide and long formats:

* **pivot_longer()**: Convert wide format to long format
* **pivot_wider()**: Convert long format to wide format

.. code-block::

    # Example of pivot operations
    table = Table([{"id": 1, "x": 10, "y": 20}, {"id": 2, "x": 30, "y": 40}])
    
    long = table.pivot_longer(["x", "y"], names_to="variable", value_to="value")
    # Result: [{"id": 1, "variable": "x", "value": 10}, {"id": 1, "variable": "y", "value": 20},
    #          {"id": 2, "variable": "x", "value": 30}, {"id": 2, "variable": "y", "value": 40}]

Import/Export
~~~~~~~~~~~~~~~

The Table class supports various formats:

* JSON files (to_json/from_json)
* Excel files (to_excel/from_excel)
* CSV files (to_csv/from_csv)
* Pandas DataFrames (to_pandas/from_pandas)

For more information on the underlying Tuplist class, see the `pytups documentation <https://pchtsp.github.io/pytups/code.html#module-pytups.tuplist/>`_.

Table class API
===============

.. autoclass:: mango.table.Table
      :members:
      :undoc-members:
      :show-inheritance:
      :exclude-members:
      :inherited-members:
