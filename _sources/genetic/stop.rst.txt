Stop conditions
-----------------

The following conditions will cause the algorithm to stop:

- Reach the maximum number of generations set up by the configuration (:attr:`max_generations`).
- Having all the individuals with the same fitness value.
- Having all but one individual with the same fitness value.
- Detecting a genetic diversity problem (during parent selection).

.. note::
    The last two conditions have to be improved upon as two individuals with the same fitness value or phenotype does not mean they have the same genes or genotype so the genetic diversity does not have to be in danger.

In the future some other stopping conditions will be added:

- Stagnation: If the best fitness does not improve over a given number of generations the algorithm should stop.
- Time limit: If the algorithm has run for too long it should stop.
- Convergence to a fitness value: If a good enough fitness value is known and indicated convergence can stop one that said value is reached.

All these stopping criteria will be configurable and will be activated or deactivated at will.