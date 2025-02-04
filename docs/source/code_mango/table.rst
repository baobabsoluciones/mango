Table
------------


The **Table** class is used to represent and work on a table or dataframe as a list of dict. It derives from the pytups.Tuplist class.


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

and can be directly initialized with this list. For more information on the Tuplist class, see the `pytups documentation <https://pchtsp.github.io/pytups/code.html#module-pytups.tuplist/>`_.
This class is particularly useful for working with data in json format.

Table class
============

.. autoclass:: mango.table.Table
      :members:
      :undoc-members:
      :show-inheritance:
      :exclude-members:
      :inherited-members:
