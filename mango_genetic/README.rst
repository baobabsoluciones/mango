Mango Genetic
=============

A Python library for implementing genetic algorithms and evolutionary computation methods.

Overview
--------

Mango Genetic provides a comprehensive framework for building and running genetic algorithms. It includes implementations of various genetic operators such as selection, crossover, mutation, and replacement strategies.

Features
--------

**Individual Management**
- Base classes for representing individuals with different encoding types (real, binary, integer, categorical)

**Population Control**
- Population management with configurable size and generation limits

**Selection Methods**
- Multiple selection strategies including roulette wheel, tournament, rank-based, and elitism

**Crossover Operators**
- Various crossover methods like blend, one-split, two-split, linear, flat, gaussian, and mask

**Mutation Control**
- Configurable mutation rates with static, adaptive, gene-based, and population-based approaches

**Replacement Strategies**
- Different replacement methods including elitist, stochastic elitist, random, and offspring-only

**Configuration System**
- Flexible configuration management for all genetic algorithm parameters

Installation
------------

**Using uv:**

.. code-block:: bash

   uv add mango-genetic

**Using pip:**

.. code-block:: bash

   pip install mango-genetic

Dependencies
------------

- Python >= 3.10
- numpy >= 1.24.4
- mango[data] == 0.3.0a8

Quick Start
-----------

.. code-block:: python

   from mango_genetic.config import GeneticBaseConfig
   from mango_genetic.individual import Individual
   from mango_genetic.population import Population

   # Load configuration
   config = GeneticBaseConfig("config.cfg")

   # Create population and run genetic algorithm
   population = Population(config, fitness_function)
   population.run()

Documentation
-------------

For detailed documentation, visit the `Mango Documentation <https://baobabsoluciones.github.io/mango/>`_.

License
-------

This project is licensed under the Apache Software License.


Support
-------

For questions, issues, or contributions, please contact:

- Email: mango@baobabsoluciones.es
- Create an issue on the repository

---

Made with ❤️ by `baobab soluciones <https://baobabsoluciones.es/>`_