Selection
----------

The selection process in genetic algorithms (GA) is a pivotal mechanism that emulates the principles of natural selection, allowing the propagation of individuals with higher fitness scores while gradually phasing out less fit solutions. This mimics the survival-of-the-fittest concept from Darwinian evolution, promoting the convergence of the population toward optimal or near-optimal solutions.

One commonly employed selection method is proportional or roulette wheel selection, where individuals are assigned probabilities of being selected based on their fitness scores. Higher-fitness individuals have larger slices of the "roulette wheel," making them more likely to be chosen for reproduction. This probabilistic approach ensures that individuals with superior fitness contribute more genetic material to the next generation, mirroring the biological concept that successful traits are more likely to be inherited.

Another widely used selection mechanism is tournament selection, wherein individuals are randomly sampled from the population, and the one with the highest fitness is chosen as a parent for reproduction. This approach introduces an element of randomness and diversity, preventing the algorithm from fixating on a limited set of solutions.

The careful design of the selection process is crucial in balancing exploration and exploitation, allowing the genetic algorithm to efficiently navigate the solution space. Effective selection mechanisms contribute to the algorithm's ability to efficiently explore diverse regions early in the optimization process and converge towards promising solutions as generations progress. The interplay between selection and genetic operators ensures the continual refinement of the population, guiding it toward increasingly optimal solutions over successive generations.

Key concepts in selection implementation
=========================================

Mango's selection process has to be separated in two steps:

- First, we calculate the selection probabilities for each individual in the population. This is done by the selected selection process on the configuration of the algorithm. The different selection processes will be explained in the following section.
- Secondly, based on the probabilities, a method is called to select the parents needed for the crossover (typically 2). The method used here is the :meth:`_select_parents<mango.models.genetic.population.Population._select_parents>` method.

Selection processes
====================

Random selection
~~~~~~~~~~~~~~~~

In this selection process the probability of selection of each parent is the same (:math:`\frac{1}{n}` with :math:`n` being the population size).

.. caution::
    This selection process is not recommended as it does not take into account the fitness of the individuals and it can lead to a very slow convergence.

This selection process is implemented in the method: :meth:`_random_selection<mango.models.genetic.population.Population._random_selection>`.

Elitism selection
~~~~~~~~~~~~~~~~~

In this selection process the :math:`k` best individuals are assigned the same probability of selection (:math:`\frac{1}{k}`) while the rest of the individuals have a probability of zero.

.. caution::
    This selection process can reduce the genetic diversity of the population and it can lead to a premature convergence. If the value of :math:`k` to low it may happend that all possible choices of parent are actually the same individual causing the algorithm to stop.

    If :math:`k` is equal to the population size, this selection process is equivalent to the random selection process.

This selection process is implemented in the method: :meth:`_elitism_selection<mango.models.genetic.population.Population._elitism_selection>`.

Rank selection
~~~~~~~~~~~~~~

This selection process assigns a selection probability to each individual based on its rank in the population. This value is calculated as:

.. math::
    p_i = \frac{1}{n} * \left( s - (2s -2)\frac{i-1}{n-1} \right)

where :math:`s` is the selection pressure or rank pressure (usually 2), :math:`n` the size of the population and :math:`i` is the rank of the individual, a value between :math:`1` and :math:`n`, where the best individual has rank :math:`n` and the worst rank :math:`1`.

.. tip::
    The las individual in the population is going to have a probability of selection zero.

.. caution::
    If the rank pressure is set up to 1 this selection process es equivalent to the random selection process.

This selection process is implemented in the method: :meth:`_rank_selection<mango.models.genetic.population.Population._rank_selection>`.

The rank pressure is a parameter that can be set up in the configuration of the algorithm. The default value is 2. The name of the parameter is ``rank_pressure``.

Order selection
~~~~~~~~~~~~~~~

This selection process assigns a selection probability to each individual based on its order in the population. This value is calculated as:

.. math::
    p_i = \frac{i * 2}{n * (n+1)}

where :math:`n` the size of the population and :math:`i` is the order of the individual, a value between :math:`1` and :math:`n`, where the best individual has order :math:`n` and the worst order :math:`1`.

.. tip::
    The probabilities assigned to each individual are very similar to those calculated by the rank selection (with rank pressure 2) but the last individual does have a probability of being selected in this case.

This selection process is implemented in the method: :meth:`_order_selection<mango.models.genetic.population.Population._order_selection>`.

Roulette wheel selection
~~~~~~~~~~~~~~~~~~~~~~~~

This selection process is also known as proportional selection. In this selection process the probability of selection of each individual is proportional to its fitness.

This selection process has two different methods to calculate the selection probabilities depending if the problem is to minimize or maximize. For minimization problems:

.. math::
    p_i = \frac{max(f_i) - f_i}{\sum_{j=1}^{n} (max(f_j) - f_j)}

But for maximization problems:

.. math::
    p_i = \frac{f_i - min(f_i)}{\sum_{j=1}^{n} (f_j - min(f_j)}

where :math:`f_i` is the fitness of the individual :math:`i` and :math:`n` is the size of the population.

Based on these formulas the worst individual should not have a selection probability, but one is assigned based on the probability of the second worst.

Tournament selection
~~~~~~~~~~~~~~~~~~~~

PLACEHOLDER

Boltzmann selection
~~~~~~~~~~~~~~~~~~~

PLACEHOLDER

Parent selection
================

PLACEHOLDER