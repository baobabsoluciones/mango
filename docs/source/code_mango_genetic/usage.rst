Usage
--------

In this section of the documentation an small example of how to use the genetic algorithm (GA) module is given.

For the example we are going to use as the optimization objective the Ackley function (:meth:`ackley<mango.benchmark.optimization.ackley.ackley>`).

First we need to import the GA module and the Ackley function:

.. code-block:: python

    from mango.benchmark.optimization.ackley import ackley
    from mango_genetic.population import Population
    from mango_genetic.config import GeneticBaseConfig

Then we have to define the configuration that we are going to use on a `.yml` file. We can write the following configuration to a file called `ackley.cfg`:

.. code-block:: yaml

    [main]
    population_size         = 50
    max_generations         = 300
    optimization_objective  = min
    selection               = roulette
    crossover               = blend
    replacement             = elitist
    mutation_control        = static
    mutation_base_rate      = 0.2

    [individual]
    encoding        = real
    gene_length     = 100
    gene_min_value  = -32.768
    gene_max_value  = 32.768

    [crossover]
    blend_expansion   = 0.5

Then we can read the configuration:

.. code-block:: python

    config = GeneticBaseConfig('ackley.cfg')

Now we can create the population:

.. code-block:: python

    population = Population(config, ackley)

And we run the GA:

.. code-block:: python

    population.run()

Once the model has finished we can get the best individual, its fitness and solution:

.. code-block:: python

    best_individual = population.best.individual
    best_fitness = best_inidividual.fitness
    best_solution = best_individual.genes

As it can be seen running a genetic algorithm to optimize a function is very easy and requires a few lines of code. In order to use any other function we can just have a method that receives the genes (as the :meth:`ackley<mango.benchmark.optimization.ackley.ackley>` function does) or we can use the :class:`Problem<mango.models.genetic.problem.Problem>`, creating a subclass and defining the :meth:`calculate_fitness<mango.models.genetic.problem.Problem.calculate_fitness>` method.

This would be done as follows:

.. code-block:: python

    from mango.models.genetic.problem import Problem

    class MyProblem(Problem):
        def calculate_fitness(self, genes):
            # Calculate the fitness of the individual
            # ...
            return fitness
    problem = MyProblem()
    population = Population(config, problem)
    population.run()