Population
------------
In the context of genetic algorithms (GA), the concept of a population is fundamental to the simulation of evolutionary processes inspired by natural selection. A population in genetic algorithms represents a group of potential solutions, each encoded as a set of parameters or genes. These solutions undergo a process of evolution analogous to biological evolution, where the fittest individuals are more likely to survive and reproduce, passing their genetic information to subsequent generations.

The diversity and size of the population play crucial roles in the effectiveness of genetic algorithms. A diverse population ensures exploration of a broad solution space, while a sufficiently large population helps prevent premature convergence to suboptimal solutions. The genetic algorithm's iterative process involves the selection of individuals based on their fitness, followed by genetic operators such as crossover and mutation that mimic biological recombination and mutation. This interplay of selection and genetic operations facilitates the continuous improvement of the population over successive generations, ultimately converging towards better solutions to the given problem.

In essence, the population in genetic algorithms serves as the dynamic substrate upon which the algorithm acts, embodying a collective repository of potential solutions that adapt and evolve over time. The characteristics of the population heavily influence the algorithm's exploration-exploitation balance, determining its ability to navigate complex solution spaces and discover high-quality solutions to optimization problems.

Implementation
==============

The :class:`Population<mango.models.genetic.population.Population>` class is a container for a collection of :class:`Individual<mango.models.genetic.individual.Individual>` objects, each of which represents a potential solution to the given problem. The population is initialized with a list of individuals, which can be generated randomly or from a set of predefined solutions. The population size is fixed and cannot be changed after initialization.

From the existing :class:`Population<mango.models.genetic.population.Population>`, a new class can be created to override any of the methods that the class has to control the different aspects of the genetic algorithm, but we believe that it should not be necessary as a lot of different operators are already included and the main workflow for the algorithm is the most common one.

On main method that may be altered is the stop conditions that have to be evaluated on the :meth:`stop<mango.models.genetic.population.Population.stop>` as it is the one that decides when the algorithm has to stop. The default implementation is to stop when the maximum number of generations is reached or the genetic diversity between individuals in the population is too low (more information about it can be found on the :doc:`stop` section of the documentation).