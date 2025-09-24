Genetic
--------

Here we have the code documentation for the main classes of the genetic algorithm: individual, population, configuration and problem.

.. autoclass:: mango_genetic.individual.Individual
    :members:
    :undoc-members:
    :private-members:
    :special-members: __hash__, __eq__


.. autoclass:: mango_genetic.population.Population
    :members:
    :undoc-members:
    :private-members:


.. autoclass:: mango_genetic.config.GeneticBaseConfig
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

.. autoclass:: mango_genetic.problem.Problem
    :members:
    :undoc-members:
    :private-members:
    :show-inheritance:

There is also some other helper classes that are used by the genetic algorithm that are documented below.

.. autoclass:: mango_genetic.shared.exceptions.GeneticDiversity

.. autoclass:: mango_genetic.shared.exceptions.ConfigurationError