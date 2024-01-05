Replacement
-------------

Replacement is the process by which individuals in the current population are updated with new candidate solutions generated through crossover and mutation operations. This mechanism is fundamental to the algorithm's ability to maintain diversity, explore the solution space efficiently, and converge towards optimal or near-optimal solutions.

The choice of replacement strategy significantly influences the performance and behavior of genetic algorithms (GA). There are various replacement schemes, each with its own advantages and drawbacks. Common replacement strategies include generational replacement, steady-state replacement, and elitism, each catering to specific optimization requirements and population dynamics.

Understanding the dynamics of replacement involves a delicate balance between exploration and exploitation. On one hand, the algorithm needs to explore different regions of the solution space by introducing diverse individuals. On the other hand, it must exploit promising solutions by allowing them to survive and propagate through generations.

Replacement precesses
=====================

In mango there is several replacement processes implemented that can be used.

Random replacement
~~~~~~~~~~~~~~~~~~

In this replacement process the population of the next generation is randomly selected among the current generation and the children.

It is implemented on the :meth:`random_replacement<mango.models.genetic.population.Population._random_replacement>` method.

Offspring replacement
~~~~~~~~~~~~~~~~~~~~~

In this replacement process the population of the next generation is composed by the children only.

It is implemented on the :meth:`offspring_replacement<mango.models.genetic.population.Population._offspring_replacement>` method.

Elitism replacement
~~~~~~~~~~~~~~~~~~~

In this replacement process the population of the next generation is composed by the best individuals of the current generation and the children.

It is implemented on the :meth:`elitism_replacement<mango.models.genetic.population.Population._elitist_replacement>` method.

Elitist stochastic replacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this replacement process the population of the next generation is selected randomly from the current generation and the children, but every individual has a probability of being selected proportional to its fitness.

The probability of being selected is calculated in the same method as the one used on the :ref:`roulette-wheel-selection-label`.

It is implemented on the :meth:`elistist_stochastic_replacement<mango.models.genetic.population.Population._elitist_stochastic_replacement>` method.