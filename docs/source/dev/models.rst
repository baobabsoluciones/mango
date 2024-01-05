Models
----------


Genetic
========

Here we have the code documentation for the main classes of the genetic algorithm: individual, population, configuration and problem.

.. autoclass:: mango.models.genetic.individual.Individual
    :members:
    :undoc-members:
    :private-members:
    :special-members: __hash__, __eq__


.. autoclass:: mango.models.genetic.population.Population
    :members:
    :undoc-members:
    :private-members:


.. autoclass:: mango.models.genetic.config.GeneticBaseConfig
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

.. autoclass:: mango.models.genetic.problem.Problem
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

There is also some other helper classes that are used by the genetic algorithm that are documented below.

.. autoclass:: mango.models.genetic.shared.exceptions.GeneticDiversity

.. autoclass:: mango.models.genetic.shared.exceptions.ConfigurationError


Neural Networks components
===========================

Activations
~~~~~~~~~~~~

.. automodule:: mango.models.activations

Neural networks
~~~~~~~~~~~~~~~~

.. automodule:: mango.models.neural_networks

Optimization
=============

Pyomo
~~~~~~

.. automodule:: mango.models.pyomo
