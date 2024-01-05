Configuration
--------------

The genetic algorith (GA) has a helper class that is used to parse `.yml` files to read the configuration that has to be used.

This configuration :class:`GeneticBaseConfig<mango.models.genetic.config.GeneticBaseConfig>` class can be extended and modified at will.

The parameters that can be modified are divided by sections.

Main section
============

In this section we have tha main controls of the genetic algorithm:

- **population_size**: The size of the population that will be used in the GA. It has to be an integer value and it defaults to 100.
- **max_generations**: The maximum number of generations that will be used in the GA. It has to be an integer value and it defaults to 500.
- **optimization_objective**: The objective of the optimization. It can be `max` or `min`. It has to be a string and can only be either `max` or `min`.
- **selection**: The selection method that will be used in the GA. It can be any of the following values: `random`, `elitism`, `rank`, `order`, `roulette` or `tournament`.
- **crossover**: The crossover method that will be used in the GA.It can be any of the following values: `one-split`, `two-split`, `mask`, `linear`, `flat`, `blend` or `gaussian`.
- **replacement**: The replacement method that will be used in the GA. It can be any of the following values: `random`, `only-offspring`, `elitist` or `elitist-stochastic`.
- **mutation_control**: The mutation control method that will be used in the GA. It can be `static`, `gene-based`, `population-based` or `adaptative`.
- **mutation_base_rate**: The base mutation rate that will be used in the GA. It can be any float number between 0 and 1 and it defaults to 0.1.

An example of the `.yml` file can be seen following:

.. code-block:: yaml

    [main]
    population_size = 100
    max_generations = 100
    optimization_objective = max
    selection = tournament
    crossover = one_split
    replacement = elitism
    mutation_control = adaptative
    mutation_base_rate = 0.1


Individual section
==================

The section for the individual has the following parameters:

- **encoding**: The encoding that will be used in the GA. It can be `binary`, `integer` or `real`.
- **gene_length**: The length of the genome that will be used in the GA. It has to be an integer value and it defaults to 0.
- **gene_min_value**: The minimum value that a gene can have. It has to be a float value.
- **gene_max_value**: The maximum value that a gene can have. It has to be a float value.

An example of the `.yml` file can be seen following:

.. code-block:: yaml

    [individual]
    encoding = binary
    gene_length = 10
    gene_min_value = 0.0
    gene_max_value = 1.0


Selection section
=================

The section for the selection has the following parameters:

- **elitism_size**: The size of the elitism that will be used in the GA. It has to be an integer value and it defaults to 20.
- **tournament_size**: The size of the tournament that will be used in the GA. It has to be an integer value and it defaults to 2.
- **rank_pressure**: The rank pressure that will be used in the GA. It has to be a float value between 1 and 2 and defaults to 2.

An example of the `.yml` file can be seen following:

.. code-block:: yaml

    [selection]
    elitism_size = 20
    tournament_size = 2
    rank_pressure = 2.0

Crossover section
=================

The section for the crossover has the following parameters:

- **offspring_size**: The size of the offspring that will be used in the GA. It has to be an integer value and it defaults to 100.
- **blend_expansion**: The blend expansion that will be used in the GA. It has to be a float value  and defaults to 0.5.

An example of the `.yml` file can be seen following:

.. code-block:: yaml

    [crossover]
    offspring_size = 100
    blend_expansion = 0.5

Mutation section
================

The section for the mutation has the following parameters:

- **generation_adaptative**: The generation adaptative that will be used in the GA. It has to be an integer value and it defaults to 10.

An example of the `.yml` file can be seen following:

.. code-block:: yaml

    [mutation]
    generation_adaptative = 10
