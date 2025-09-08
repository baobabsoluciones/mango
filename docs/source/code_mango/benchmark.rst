Benchmarks
-----------

This module contains a comprehensive collection of benchmark functions for optimization problems. These functions are commonly used to test and evaluate the performance of optimization algorithms, including genetic algorithms, particle swarm optimization, and other metaheuristic methods.

The benchmark functions include:

* **Unimodal functions**: Functions with a single global optimum (e.g., Rosenbrock, Sphere)
* **Multimodal functions**: Functions with multiple local optima (e.g., Rastrigin, Ackley, Griewank)
* **Composition functions**: Complex functions combining multiple sub-functions
* **Real-world inspired functions**: Functions modeling practical optimization problems

Each function is provided in both standard and inverted versions, allowing for both minimization and maximization testing scenarios. These benchmarks serve as a standardized test suite for comparing algorithm performance and establishing baseline metrics before developing new optimization approaches. 

Optimization benchmarks
=======================

Ackley Function
~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.ackley.ackley
.. autofunction:: mango.benchmark.optimization.ackley.inverted_ackley

Bukin Function
~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.bukin.bukin_function_6
.. autofunction:: mango.benchmark.optimization.bukin.inverted_bukin_function_6

Cross-in-Tray Function
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.cross_in_tray.cross_in_tray
.. autofunction:: mango.benchmark.optimization.cross_in_tray.inverted_cross_in_tray

Dolan Function
~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.dolan.dolan_function_no2
.. autofunction:: mango.benchmark.optimization.dolan.inverted_dolan_function_no2

Drop-Wave Function
~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.drop_wave.drop_wave
.. autofunction:: mango.benchmark.optimization.drop_wave.inverted_drop_wave

Egg-Holder Function
~~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.egg_holder.egg_holder
.. autofunction:: mango.benchmark.optimization.egg_holder.inverted_egg_holder

Gramacy & Lee Function
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.gramacy_lee.gramacy_lee
.. autofunction:: mango.benchmark.optimization.gramacy_lee.inverted_gramacy_lee

Griewank Function
~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.griewank.griewank
.. autofunction:: mango.benchmark.optimization.griewank.inverted_griewank

Holder Table Function
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.holder_table.holder_table
.. autofunction:: mango.benchmark.optimization.holder_table.inverted_holder_table

Langermann Function
~~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.langermann.langermann
.. autofunction:: mango.benchmark.optimization.langermann.inverted_langermann

Levy Function
~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.levy.levy
.. autofunction:: mango.benchmark.optimization.levy.inverted_levy
.. autofunction:: mango.benchmark.optimization.levy.levy_function_no13
.. autofunction:: mango.benchmark.optimization.levy.inverted_levy_no13

Rastrigin Function
~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.rastrigin.rastrigin
.. autofunction:: mango.benchmark.optimization.rastrigin.inverted_rastrigin

Rosenbrock Function
~~~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.rosenbrock.rosenbrock
.. autofunction:: mango.benchmark.optimization.rosenbrock.inverted_rosenbrock

Schaffer Function
~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.schaffer.schaffer_function_no2
.. autofunction:: mango.benchmark.optimization.schaffer.inverted_schaffer_function_no2
.. autofunction:: mango.benchmark.optimization.schaffer.schaffer_function_no4
.. autofunction:: mango.benchmark.optimization.schaffer.inverted_schaffer_function_no4

Schwefel Function
~~~~~~~~~~~~~~~~~

.. autofunction:: mango.benchmark.optimization.schwefel.schwefel
.. autofunction:: mango.benchmark.optimization.schwefel.inverted_schwefel







