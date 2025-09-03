.. _crossover-label:

Crossover
----------

At its core, crossover emulates the biological concept of genetic recombination, where genetic material is exchanged between parent organisms to create offspring with a combination of traits from both parents. In the context of genetic algorithms (GA), crossover involves the recombination of genetic information encoded within individuals, typically represented as arrays of values.

The process begins by selecting pairs of individuals, often called parents, from the population based on their fitness – a measure of how well they perform in solving the given problem. Crossover then takes place, and specific segments of the parent chromosomes are exchanged, giving rise to new individuals known as offspring. The objective is to introduce diversity and potentially create solutions that inherit beneficial characteristics from both parents.

There are various crossover techniques employed in genetic algorithms, each with its own advantages and drawbacks. One-point crossover, two-point crossover, and uniform crossover are some common methods used to determine how genetic material is exchanged between parents. The choice of crossover method can significantly impact the algorithm's ability to explore the solution space effectively.

Crossover in genetic algorithms promotes the exploration of the solution space by combining different traits from parent individuals. This exploration mechanism allows the algorithm to move towards promising regions of the solution space, potentially converging to optimal or near-optimal solutions. However, the effectiveness of crossover is intertwined with other genetic operators like mutation, as well as parameters such as population size and selection mechanisms.

In conclusion, crossover is a fundamental component of genetic algorithms, playing a pivotal role in their ability to evolve solutions to complex problems. Its mimicry of biological recombination enables the algorithm to efficiently explore the solution space and discover novel, potentially superior solutions by combining the genetic information of parent individuals. The careful selection and tuning of crossover methods are essential for the success of genetic algorithms in finding optimal solutions in diverse application domains.

Crossover processes
===================

In mango there is several crossover processes implemented that can be used for the different types of encodings that the :class:`Individual<mango.models.genetic.individual.Individual>` class supports.

.. _one-split-label:

One-split crossover
~~~~~~~~~~~~~~~~~~~

In this crossover a split point is randomly selected and two children are created from two parents. The first child is created by taking the first part of the first parent and the second part of the second parent. The second child is created by taking the first part of the second parent and the second part of the first parent. An example can be seen in the following figure:

.. figure:: ../static/img/one_split.png
    :width: 500
    :align: center

    One split crossover

.. tip::
    This crossover can be used for all types of encodings.

This crossover process is implemented in the method: :meth:`one_split_crossover<mango_genetic.population.Population._one_split_crossover>`

.. _two-split-label:

Two-split crossover
~~~~~~~~~~~~~~~~~~~

In this crossover two split points are randomly selected and two children are created from two parents. The first child is created by taking the first part of the first parent, the second part of the second parent and the third part of the first parent. The second child is created by taking the first part of the second parent, the second part of the first parent and the third part of the second parent. An example can be seen in the following figure:

.. figure:: ../static/img/two_split.png
    :width: 500
    :align: center

    Two split crossover

.. tip::
    This crossover can be used for all types of encodings.

This crossover process is implemented in the method: :meth:`two_split_crossover<mango_genetic.population.Population._two_split_crossover>`

.. _mask-label:

Mask crossover
~~~~~~~~~~~~~~

In this crossover a mask is randomly generated and two children are created from two parents. The first child is created by taking the values of the first parent where the mask is 1 and the values of the second parent where the mask is 0. The second child is created by taking the values of the second parent where the mask is 1 and the values of the first parent where the mask is 0. An example can be seen in the following figure:

.. figure:: ../static/img/mask.png
    :width: 500
    :align: center

    Mask crossover

.. tip::
    This crossover can be used for all types of encodings.

This crossover process is implemented in the method: :meth:`mask_crossover<mango_genetic.population.Population._mask_crossover>`

.. _linear-label:

Linear crossover
~~~~~~~~~~~~~~~~

In this crossover a linear combination of the two parents is created and three children are created, it was proposed by Wright :cite:p:`wright1991genetic`. The linear combination is defined by the following formula:

.. math::
    \begin{align}
        \text{child}_1 &= \frac{(\text{parent}_1 +  \text{parent}_2 )}{2}\\
        \text{child}_2 &= 1.5 \cdot \text{parent}_1 - 0.5 \cdot \text{parent}_2\\
        \text{child}_3 &= -0.5 \cdot \text{parent}_1 + 1.5 \cdot \text{parent}_2
    \end{align}

The objective of this crossover is to handle both exploration and exploitation. The first child is the average of the two parents and is used for exploitation. The second and third child are used for exploration. An example can be seen in the following figure:

.. figure:: ../static/img/linear.png
    :width: 700
    :align: center

    Linear crossover

As it can be seen in the example the first child lies between both original parents while the second and third child are outside the range of the original parents. This is the reason why the second and third child are used for exploration.

.. warning::
    This crossover can only be used for real encodings as it will not work with binary or integer encodings where the linear combination is not possible

This crossover process is implemented in the method: :meth:`linear_crossover<mango_genetic.population.Population._linear_crossover>`

.. _flat-label:

Flat crossover
~~~~~~~~~~~~~~

This method is an implementation of Radcliffe's flat crossover :cite:p:`radcliffe1991equivalence`. In this crossover two children are created from two parents by taking a random value for each gene from a uniform distribution defined by the values of the father genes. An example can be seen in the following figure:

.. figure:: ../static/img/flat.png
    :width: 500
    :align: center

    Flat crossover

.. warning::
    This crossover can only be used with real encoding. In the future there will be an implementation for integer encoding as well.

This crossover process is implemented in the method: :meth:`flat_crossover<mango_genetic.population.Population._flat_crossover>`

.. _blend-label:

Blend crossover
~~~~~~~~~~~~~~~

This crossover is an implementation of Eshelman's blend crossover :cite:p:`eshelman1993real`. In this crossover two children are created from two parents by taking a random value for each gene from a uniform distribution defined by the interval defined by the parents.

Fist we calculate the interval for the genes as follows:

.. math::
    \text{interval} = abs(\text{parent}_1 - \text{parent}_2)

Then the first child gets generated from randomly sampling from a uniform distribution from the following interval:

.. math::
    [\text{min}(\text{parent}_1, \text{parent}_2) - \alpha \cdot \text{interval}, \text{max}(\text{parent}_1, \text{parent}_2) + \alpha \cdot \text{interval}]

Where :math:`\alpha` is a parameter that controls the expansion of the interval and is set to 0.5 by default. To change its value we have to change the parameter :attr:`blend_expansion<mango.models.genetic.population.Population.blend_expansion>`.

.. tip::
    If the parameter is set to 0 then the first child is generated in the same way as the flat crossover.

    If the parameter is set to 0.5 then the uniform distribution can generate numbers from the same interval that the linear crossover generates, but instead of having a linear combination of the two parents we have a random combination of the two parents.

Then, the second child is calculated as:

.. math::
    \text{child}_2 = \text{parent}_1  + \text{parent}_2 - \text{child}_1

An example can be seen in the following figure:

.. figure:: ../static/img/blend.png
    :width: 700
    :align: center

    Blend crossover

.. warning::
    This crossover can only be used with real encoding.

This crossover process is implemented in the method: :meth:`blend_crossover<mango_genetic.population.Population._blend_crossover>`

.. _gaussian-label:

Gaussian crossover
~~~~~~~~~~~~~~~~~~

This crossover is an implementation of Ono UNDX crossover method :cite:p:`ono2003real`. In this crossover two children are created from three parents.

This is probably the most complex crossover method implemented in mango. The explanation here will be as brief as possible, and if further explanation is needed please refer to the original paper.

The main idea is that from two parents we calculate the line defined by them, then we calculate the minimum distance between the third parent and said line. Then the children is calculated as the midpoint in said line and then modified by all the orthonormal vectors to the distance vector to the third parent multiplied by a random value from a normal distribution with mean 0 and standard deviation :math:`\sigma_{\eta} \cdot distance`.

First two values are calculated for the process, :math:`\sigma_{\eta}` and :math:`\sigma_{\xi}`. The first one fixed value of :math:`\frac{0.35}{\sqrt{n}}` being :math:`n` the number of genes, the second one gets the fixed value of :math:`\frac{1}{4}`.

The current implementation follows these steps:

1. Calculate the vector for the line defined by the first two parents.
2. Calculate the unit vector in said line.
3. Calculate the mid point in said line.
4. Calculate the distance vector from the third parent to the line and its norm.
5. Calculate the random vectors in the base from :math:`\sigma_{\eta}` and the norm calculated on step 4.
6. Substract from said vector the projection of the distance vector between the first two parents.
7. Add the paralell component calculated with :math:`\sigma_{\xi}` and the distance between the first two parents.
8. Calculate the two children by adding and substracting the calculated base from the mid point in the line.

.. warning::
    This crossover can only be used with real encoding.

This crossover process is implemented in the method: :meth:`gaussian_crossover<mango_genetic.population.Population._gaussian_crossover>`.

Morphology crossover
~~~~~~~~~~~~~~~~~~~~

This crossover method is not yet implemented on mango.



