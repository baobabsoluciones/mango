Mutation
----------

Although most of the mutation is controlled from the :class:`Individual<mango.models.genetic.individual.Individual` class, there is some control that can also be done from the :class:`Population<mango.models.genetic.population.Population`> class.

If set in the configuration the attribute :attr:`mutation_control` to a value of `gene_based`, `population_based` or `adaptative` we can control how the mutation rate evolves through the generations.

Gene based mutation control
===========================

This control modifies the :attr:`mutation_rate` of each individual based on the number of genes the individuals have. It was proposed by Kenneth de Jong.

The mutation rate then will be calculated as follows:

.. math::

    mutation\_rate = \frac{1}{n}

being :math:`n` the number of genes the individual has.

This way the mutation rate will be inversely proportional to the number of genes the individual has.

Population based mutation control
=================================

This control modifies the :attr:`mutation_rate` of each individual based on the number of individuals the population has. It was proposed by Schaffer :cite:p:`schaffer1989study`.

The mutation rate then will be calculated as follows:

.. math::

    \text{mutation_rate} = \frac{1}{p^{0.9318} \cdot n^{0.4535}}

being :math:`p` the number of individuals the population has and :math:`n` the number of genes the individual has.


Adaptative mutation control
===========================

To control how the population is evolving and having a unit-less measure of the population we are using the coefficient of variation (CV), also known as Normalized Root-Mean-Square Deviation (NRMSD).

This coefficient is calculated as follows:

.. math::

    CV = \frac{\sigma}{\mu}

Where :math:`\\sigma` is the standard deviation and :math:`\\mu` is the mean of the population.

When the value of this coefficient decreases then the mutation rate is increased as it means the population is slowly converging towards one solution, where as if the value of the coefficient increases then the mutation rate is decreased as it means the population is diverging and we need to explore more the search space.

The parameter :attr:`generation_adaptative` controls after how many generations the mutation rate has to be modified based on the state of the coefficient.

That way we can control how the exploration and exploitation is done in the search space.