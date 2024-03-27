Table
------------


The **Table** class is used to represent and work on a table or dataframe as a list of dict.

For example the following table

======= =======
   a       b
======= =======
   1     False
   2     True
   3     False
======= =======

will be represented as

.. code-block::

    [{"a": 1, "b": False}, {"a": 2, "b": True}, {"a": 3, "b": False}]

This class is particularly useful for working with data in json format.

Table class
============

.. autoclass:: mango.table.Table
