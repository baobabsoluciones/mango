from collections import Counter
from math import sqrt
from random import randint
from typing import Union

import numpy as np
from mango_genetic.config import GeneticBaseConfig
from mango_genetic.individual import Individual
from mango_genetic.problem import Problem
from mango_genetic.shared.exceptions import GeneticDiversity, ConfigurationError


class Population:
    """
    Main population class for genetic algorithm operations.

    Manages a population of individuals through the complete genetic algorithm lifecycle,
    including initialization, selection, crossover, mutation, and replacement phases.
    Supports various selection methods, crossover operators, and mutation strategies.

    :param config: Configuration object containing genetic algorithm parameters
    :type config: GeneticBaseConfig
    :param evaluator: Function or class for evaluating individual fitness
    :type evaluator: Union[callable, Problem]
    """

    def __init__(self, config, evaluator):
        # General configuration
        self.config = config
        self.evaluator = evaluator
        self.generation = 1
        self.max_generations = self.config("max_generations")
        self.population_size = self.config("population_size")
        self._optimization_objective = self.config("optimization_objective")
        self.population = np.array([], dtype=Individual)
        self.selection_probs = []
        self.offspring = np.array([], dtype=Individual)
        self.best = None
        self._count_individuals = 0
        self._gene_length = self.config("gene_length")
        self._encoding = self.config("encoding")

        # selection params
        self._selection_type = self.config("selection")
        self._elitism_size = self.config("elitism_size")
        self._tournament_size = self.config("tournament_size")
        self._rank_pressure = self.config("rank_pressure")

        # crossover params
        self._crossover_type = self.config("crossover")
        self._offspring_size = self.config("offspring_size") or self.population_size
        self._blend_expansion = self.config("blend_expansion")
        self._morphology_parents = self.config("morphology_parents")

        # mutation params
        self._mutation_type = self.config("mutation_control")
        self._mutation_rate = self.config("mutation_base_rate")
        self._coefficient_variation = None
        self._generation_adaptative = self.config("generation_adaptative")

        if self._mutation_type == "gene-based":
            """
            Based on a proposal by Kenneth De Jong (1975)
            """
            self._mutation_rate = 1 / self._gene_length
        elif self._mutation_type == "population-based":
            """
            Based on the proposal by Schaffer (1989) on A Study of Control Parameters Affecting Online Performance
            of Genetic Algorithms for Function Optimization
            """
            self._mutation_rate = 1 / (
                pow(self.population_size, 0.9318) * pow(self._gene_length, 0.4535)
            )

        # replacement params
        self._replacement_type = self.config("replacement")

        if self._encoding != "real" and self._crossover_type in [
            "blend",
            "linear",
            "flat",
            "gaussian",
        ]:
            raise ConfigurationError(
                f"Encoding {self._encoding} not supported for crossover type {self._crossover_type}"
            )

    # -------------------
    # Property methods
    # -------------------
    @property
    def config(self):
        """
        Get the configuration object for the genetic algorithm.

        Returns the configuration object containing all genetic algorithm parameters
        such as population size, encoding type, selection method, etc.

        :return: Configuration object with genetic algorithm parameters
        :rtype: GeneticBaseConfig
        """
        return self._config

    @config.setter
    def config(self, value: GeneticBaseConfig):
        """
        Set the configuration object for the genetic algorithm.

        :param value: New configuration object
        :type value: GeneticBaseConfig
        """
        self._config = value

    @property
    def evaluator(self):
        """
        Get the fitness evaluation function or class.

        Returns the function or class responsible for evaluating the fitness
        of individuals in the population.

        :return: Evaluator function or Problem class instance
        :rtype: Union[callable, Problem]
        """
        return self._evaluator

    @evaluator.setter
    def evaluator(self, value: Union[callable, Problem]):
        """
        Set the fitness evaluation function or class.

        :param value: New evaluator function or Problem class instance
        :type value: Union[callable, Problem]
        """
        self._evaluator = value

    def run(self):
        """
        Run the complete genetic algorithm evolution process.

        Executes the genetic algorithm for the specified number of generations,
        performing selection, crossover, mutation, and replacement operations
        in each generation. The algorithm can be stopped early due to low
        genetic diversity or other stop conditions.

        Raises:
            GeneticDiversity: When genetic diversity becomes too low during
                parent selection or stop phase evaluation
        """

        if self.generation == 1:
            self.init_population()
            self.update_population()
            self._coefficient_variation = self._calculate_coefficient_variation()

        while self.generation <= self.max_generations:
            self.selection()
            self.crossover()
            self.mutate()
            self.update_population()
            self.replace()
            self.update_best()
            self.stop()
            self.generation += 1

    def continue_running(self, generations: int):
        """
        Continue running the genetic algorithm for additional generations.

        Extends the maximum generation limit and continues the evolution process
        from where it previously stopped. This method works best when the algorithm
        stopped due to reaching the generation limit rather than low genetic diversity.

        :param generations: Number of additional generations to run
        :type generations: int

        Raises:
            GeneticDiversity: When genetic diversity becomes too low during
                parent selection or stop phase evaluation
        """
        self.max_generations += generations
        self.run()

    def init_population(self):
        """
        Initialize the population with random individuals.

        Creates a population of random individuals with size equal to the
        configured population_size. Each individual is created using the
        Individual.create_random_individual method with the current configuration.
        """
        self.population = np.array(
            [
                Individual.create_random_individual(i + 1, self.config)
                for i in range(self.population_size)
            ],
            dtype=Individual,
        )

        self._count_individuals = len(self.population)

    def selection(self):
        """
        Execute the selection phase of the genetic algorithm.

        Creates selection probabilities for each individual based on the configured
        selection method. These probabilities determine how likely each individual
        is to be chosen as a parent during the crossover phase.

        Supported selection methods:
        - Random: Equal probability for all individuals
        - Elitism: Only the best k individuals can be selected
        - Rank: Probability based on individual ranking
        - Order: Probability based on fitness order
        - Roulette: Probability proportional to fitness
        - Tournament: Selection through tournament competition
        - Boltzmann: Temperature-based selection (not implemented)
        """
        if self._selection_type == "random":
            self._random_selection()
        elif self._selection_type == "elitism":
            self._elitism_selection()
        elif self._selection_type == "rank":
            self._rank_selection()
        elif self._selection_type == "order":
            self._order_selection()
        elif self._selection_type == "roulette":
            self._roulette_selection()
        elif self._selection_type == "tournament":
            self._tournament_selection()
        elif self._selection_type == "boltzmann":
            self._boltzmann_selection()
        else:
            raise NotImplementedError("Selection method not implemented")

    def crossover(self):
        """
        Execute the crossover phase of the genetic algorithm.

        Creates offspring individuals by combining genetic information from selected
        parents using the configured crossover method. The crossover operation
        generates new individuals that inherit traits from their parents.

        Supported crossover methods:
        - One-split: Single crossover point
        - Two-split: Two crossover points
        - Mask: Random binary mask selection
        - Linear: Linear combination of parents (real encoding only)
        - Flat: Random values within parent range (real encoding only)
        - Blend: Extended range crossover (real encoding only)
        - Gaussian: Gaussian distribution crossover (real encoding only)
        - Morphology: Morphological crossover (not implemented)
        """
        if self._crossover_type == "one-split":
            self._one_split_crossover()
        elif self._crossover_type == "two-split":
            self._two_split_crossover()
        elif self._crossover_type == "mask":
            self._mask_crossover()
        elif self._crossover_type == "linear":
            self._linear_crossover()
        elif self._crossover_type == "flat":
            self._flat_crossover()
        elif self._crossover_type == "blend":
            self._blend_crossover()
        elif self._crossover_type == "gaussian":
            self._gaussian_crossover()
        elif self._crossover_type == "morphology":
            self._morphology_crossover()
        else:
            raise NotImplementedError("Crossover method not implemented")

    def mutate(self):
        """
        Execute the mutation phase of the genetic algorithm.

        Applies mutation to offspring individuals based on the configured mutation
        strategy. The actual mutation implementation is handled by individual
        objects as it depends on the encoding type. For adaptive mutation,
        the mutation rate is updated based on population diversity before
        applying mutations.
        """
        if self._mutation_type in ["static", "gene-based", "population-based"]:
            self._base_mutation()
        elif self._mutation_type == "adaptative":
            self._adaptative_mutation()

    def update_population(self):
        """
        Update fitness values for all individuals in the population.

        Ensures that all individuals in both the main population and offspring
        have their fitness values calculated using the configured evaluator.
        """
        self._update_fitness(np.concatenate((self.population, self.offspring)))

    def replace(self):
        """
        Execute the replacement phase of the genetic algorithm.

        Determines which individuals from the current population and offspring
        will survive to the next generation based on the configured replacement
        strategy.

        Supported replacement methods:
        - Random: Random selection from population and offspring
        - Only-offspring: Replace entire population with offspring
        - Elitist: Keep best individuals from combined population
        - Elitist-stochastic: Probabilistic selection based on fitness
        """
        if self._replacement_type == "random":
            self._random_replacement()
        elif self._replacement_type == "only-offspring":
            self._offspring_replacement()
        elif self._replacement_type == "elitist":
            self._elitist_replacement()
        elif self._replacement_type == "elitist-stochastic":
            self._elitist_stochastic_replacement()
        else:
            raise NotImplementedError("Replacement method not implemented")

    def update_best(self):
        """
        Update the best individual in the population.

        Identifies and stores the best individual based on the optimization
        objective (maximization or minimization) and their fitness value.
        """
        if self._optimization_objective == "max":
            current_best = max(self.population, key=lambda x: x.fitness)
        else:
            current_best = min(self.population, key=lambda x: x.fitness)

        if self.best is None:
            self.best = current_best
        else:
            if self._optimization_objective == "max":
                if current_best.fitness > self.best.fitness:
                    self.best = current_best
            else:
                if current_best.fitness < self.best.fitness:
                    self.best = current_best

    def stop(self):
        """
        Check stop conditions for the genetic algorithm.

        Evaluates whether the algorithm should stop based on population
        characteristics. Currently checks for low genetic diversity by
        examining fitness value distribution in the population.

        Returns:
            bool: False (algorithm continues) or raises GeneticDiversity exception

        Raises:
            GeneticDiversity: When genetic diversity becomes too low
        """
        fitness_diversity = Counter([ind.fitness for ind in self.population])
        if len(fitness_diversity) == 1:
            raise GeneticDiversity
        if len(fitness_diversity) == 2:
            if fitness_diversity.most_common(1)[0][1] == self.population_size - 1:
                raise GeneticDiversity

        return False

    # -------------------
    # Internal methods.
    # -------------------
    # Selection
    # -------------------
    def _random_selection(self):
        """
        Perform random selection with equal probabilities.

        Sets equal selection probability for all individuals in the population,
        making each individual equally likely to be chosen as a parent.
        """
        self.selection_probs = np.ones(self.population_size) / self.population_size

    def _elitism_selection(self):
        """
        Perform elitism selection based on fitness ranking.

        Selects only the k-best individuals (where k is the elitism_size parameter)
        for crossover. The best k individuals have equal selection probability,
        while all others have zero probability.

        Note: This method may reduce genetic diversity if k is small compared
        to the population size.
        """
        temp = np.argsort(
            np.array([self.extract_fitness(obj) for obj in self.population])
        )

        if self._optimization_objective == "max":
            temp = self.population_size - 1 - temp

        self.population = self.population[temp]

        self.selection_probs = np.concatenate(
            (
                np.ones(self._elitism_size),
                np.zeros(self.population_size - self._elitism_size),
            ),
        )

        self.selection_probs = self.selection_probs / sum(self.selection_probs)

    def _rank_selection(self):
        """
        Perform rank-based selection.

        Assigns selection probabilities based on the rank of individuals
        in the population, with higher-ranked individuals having higher
        selection probability.
        """
        fitness_values = np.array(
            [self.extract_fitness(obj) for obj in self.population]
        )
        temp = np.argsort(fitness_values)

        if self._optimization_objective == "min":
            temp = self.population_size - 1 - temp

        self.population = self.population[temp]

        self.selection_probs = (
            self._rank_pressure
            - (2 * self._rank_pressure - 2)
            * np.arange(self.population_size)
            / (self.population_size - 1)
        ) / self.population_size

    def _order_selection(self):
        """
        Perform order-based selection.

        Assigns selection probabilities based on the order of individuals
        in the population, with better individuals having higher probability.
        """
        temp = np.argsort(
            np.array([self.extract_fitness(obj) for obj in self.population])
        )

        if self._optimization_objective == "min":
            temp = self.population_size - 1 - temp

        self.selection_probs = (temp + 1) / (
            len(self.population) * (len(self.population) + 1) / 2
        )

    def _roulette_selection(self):
        """
        Perform roulette wheel selection.

        Assigns selection probabilities proportional to fitness values,
        with better individuals having higher probability of selection.
        """
        fitness_values = np.array(
            [self.extract_fitness(obj) for obj in self.population]
        )
        temp = np.argsort(fitness_values)
        fitness_values = fitness_values[temp]

        self.population = self.population[temp]

        max_fitness = self.population.item(-1).fitness
        min_fitness = self.population.item(0).fitness

        if self._optimization_objective == "max":
            ref_fitness = min_fitness
            self.selection_probs = fitness_values - ref_fitness

        else:
            ref_fitness = max_fitness
            self.selection_probs = ref_fitness - fitness_values

        # The last one will have probability zero, but we want to add some small probability.
        # It is going to be half of the second to last
        self.selection_probs[np.where(self.selection_probs == 0)] = (
            self.selection_probs[
                np.argmin(self.selection_probs[np.where(self.selection_probs > 0)])
            ]
            / 2
        )

        # Then we make sure that it adds up to one
        self.selection_probs = self.selection_probs / sum(self.selection_probs)

    def _tournament_selection(self):
        """
        Perform tournament selection.

        Simulates tournaments where individuals compete against each other,
        with the winner being selected based on fitness. Selection probability
        is proportional to the number of tournament wins.
        """
        self.selection_probs = np.zeros(self.population_size)
        fitness_values = np.array(
            [self.extract_fitness(obj) for obj in self.population]
        )
        for _ in range(self._offspring_size * 2):
            temp = np.zeros(self.population_size)
            individuals = np.random.choice(
                self.population_size, size=self._tournament_size, replace=False
            )

            temp[individuals] = 1
            temp = temp * fitness_values

            if self._optimization_objective == "max":
                temp[np.where(temp == 0)] = -np.inf
                winner = np.argmax(temp)
            else:
                temp[np.where(temp == 0)] = np.inf
                winner = np.argmin(temp)

            self.selection_probs[winner] += 1

        self.selection_probs = self.selection_probs / np.sum(self.selection_probs)

    def _boltzmann_selection(self):
        """
        Perform Boltzmann selection (not implemented).

        This method is not yet implemented. Boltzmann selection uses
        temperature-based probability distribution for selection.
        """
        pass

    def _select_parents(self, n: int = 2) -> tuple:
        """
        Select n parents from the population based on selection probabilities.

        Selects parents using the probabilities established during the selection phase.
        Ensures that different parents are selected to maintain genetic diversity.

        :param n: Number of parents to select (default: 2)
        :type n: int
        :return: Tuple containing the selected parent individuals
        :rtype: tuple

        Raises:
            GeneticDiversity: When unable to select different parents due to low diversity
        """

        parents = np.random.choice(
            self.population, size=1, replace=False, p=self.selection_probs
        )

        for i in range(n - 1):
            next_parent = np.random.choice(
                self.population, size=1, replace=False, p=self.selection_probs
            )
            k = 0

            while next_parent in parents:
                k += 1
                next_parent = np.random.choice(
                    self.population, size=1, replace=False, p=self.selection_probs
                )

                if k >= self.population_size:
                    raise GeneticDiversity

            parents = np.concatenate((parents, next_parent))

        return tuple(parents)

    # -------------------
    # Crossover
    # -------------------
    def _one_split_crossover(self):
        """
        Perform one-point crossover operation.

        Creates two offspring by selecting a random split point and exchanging
        genetic material between two parents at that point. The first offspring
        gets genes from parent 1 before the split and from parent 2 after the split,
        while the second offspring gets the opposite combination.
        """
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            split = randint(1, self._gene_length - 2)
            offspring_1 = np.concatenate((p1.genes[:split], p2.genes[split:]))
            offspring_2 = np.concatenate((p2.genes[:split], p1.genes[split:]))

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _two_split_crossover(self):
        """
        Perform two-point crossover operation.

        Creates two offspring by selecting two random split points and exchanging
        genetic material between two parents in the middle segment. The first
        offspring gets genes from parent 1 in the outer segments and from parent 2
        in the middle segment, while the second offspring gets the opposite combination.
        """
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            split_1 = randint(1, self._gene_length - 3)
            split_2 = randint(split_1 + 1, self._gene_length - 2)

            split_1, split_2 = sorted([split_1, split_2])

            offspring_1 = np.concatenate(
                (p1.genes[:split_1], p2.genes[split_1:split_2], p1.genes[split_2:])
            )
            offspring_2 = np.concatenate(
                (p2.genes[:split_1], p1.genes[split_1:split_2], p2.genes[split_2:])
            )

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _mask_crossover(self):
        """
        Perform uniform crossover using a random mask.

        Creates two offspring using a random binary mask to determine which
        parent contributes each gene position. The first offspring gets genes
        from parent 1 where the mask is 1 and from parent 2 where the mask is 0.
        The second offspring gets the opposite combination.
        """
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            random_indices = np.random.choice([0, 1], size=self._gene_length)
            offspring_1 = np.where(random_indices, p1.genes, p2.genes)
            random_indices = 1 - random_indices
            offspring_2 = np.where(random_indices, p1.genes, p2.genes)

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _linear_crossover(self):
        """
        Perform linear crossover as proposed by Wright (1991).

        Creates three offspring individuals using linear combinations of parent genes:
        - First offspring: Midpoint between parents (p1 + p2) / 2
        - Second offspring: 1.5 * p1 - 0.5 * p2
        - Third offspring: -0.5 * p1 + 1.5 * p2

        Note: This method is only applicable for real-valued encoding.
        """

        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            offspring_1 = (p1.genes + p2.genes) / 2
            offspring_2 = 1.5 * p1.genes - 0.5 * p2.genes
            offspring_3 = -0.5 * p1.genes + 1.5 * p2.genes

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)
            self._add_offspring(offspring_3, p1, p2)

    def _flat_crossover(self):
        """
        Implement Radcliffe's flat crossover (1990).

        Creates two offspring where each gene is randomly generated within the
        interval defined by the corresponding genes of the two parents.
        Each gene value is uniformly distributed between the minimum and maximum
        values of the parent genes at that position.

        Note: This method is only applicable for real-valued encoding.
        """
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            max_c = np.maximum(p1.genes, p2.genes)
            min_c = np.minimum(p1.genes, p2.genes)

            offspring_1 = np.random.uniform(min_c, max_c)
            offspring_2 = np.random.uniform(min_c, max_c)

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _blend_crossover(self):
        """
        Implement Blend crossover as proposed by Eshelman and Schaffer (1993).

        Creates two offspring using an extended interval around the parent genes.
        The blend_expansion parameter controls the expansion: 0 equals flat crossover,
        0.5 equals linear crossover intervals (with random selection).

        The first offspring is randomly generated in the expanded interval,
        while the second offspring is calculated to maintain the average of the parents.

        Note: This method is only applicable for real-valued encoding.
        """
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            max_c = np.maximum(p1.genes, p2.genes)
            min_c = np.minimum(p1.genes, p2.genes)
            interval = max_c - min_c

            offspring_1 = np.random.uniform(
                min_c - interval * self._blend_expansion,
                max_c + interval * self._blend_expansion,
            )

            offspring_2 = p1.genes + p2.genes - offspring_1

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _gaussian_crossover(self):
        """
        Perform Gaussian crossover (UNDX) as proposed by Ono (2003).

        Implements the Unimodal Normal Distribution Crossover using three parents
        to create offspring with Gaussian-distributed genes. The first two parents
        define the primary search space, while the third parent creates the secondary
        search space with the midpoint between the first two parents.

        Note: This method is only applicable for real-valued encoding.
        """

        while len(self.offspring) < len(self.population):
            # First we select three parents. The first two are going to be the primary search space,
            # while the third is going to create the secondary search space with the midpoint between parents 1 and 2
            p1, p2, p3 = self._select_parents(n=3)

            # This are parameters for the randomness of the gaussian distribution used in the crossover.
            sigma_eta = 0.35 / sqrt(self._gene_length)
            sigma_xi = 1 / 4

            # We calculate the distance vector, the unit distance vector and the mean point in the primary search space
            dist_p1p2 = p2.genes - p1.genes
            dist_p1p2_norm = np.linalg.norm(dist_p1p2)
            dist_p1p2_unit = dist_p1p2 / dist_p1p2_norm
            mean_p1p2 = (p1.genes + p2.genes) / 2

            # We calculate the distance vector between the third parent and the first one
            dist_p3p1 = p3.genes - p1.genes

            # This proportion is used to calculate the vector between the third parent and the primary search space
            proportion = np.dot(dist_p1p2, dist_p3p1) / np.dot(dist_p1p2, dist_p1p2)

            # Formula for the distance vector from the third parent to the primary search line.
            # This formula should be able to be applied to any dimension space and this distance vector is orthogonal
            # to the distance vector between the first and second parent.
            distance_vector = dist_p3p1 - proportion * dist_p1p2

            # Distance
            distance = np.linalg.norm(distance_vector)

            # Random vector based in the basis of the distance between the third parent and the primary search space
            # then we substract the component of the primary search
            t = np.random.normal(0, (distance * sigma_eta) ** 2, self._gene_length)
            t = t - np.dot(t, dist_p1p2_unit) * dist_p1p2

            # and we add the parallel component
            t = t + np.random.normal(0, sigma_xi) * dist_p1p2

            # We create two children
            offspring_1 = mean_p1p2 + t
            offspring_2 = mean_p1p2 - t

            # And we add them to the population
            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _morphology_crossover(self):
        """
        Perform morphological crossover (not implemented).

        This method is not yet implemented. Morphological crossover would
        use morphological operations to combine parent genetic information.
        """
        raise NotImplementedError("Morphology crossover not implemented")

    def _add_offspring(self, genes: np.array, *parents: tuple):
        """
        Add a new offspring individual to the offspring array.

        Creates a new individual with the specified genes and parent information,
        assigns it a unique index, and adds it to the offspring population.

        :param genes: Genetic information for the new offspring
        :type genes: numpy.ndarray
        :param parents: Parent individuals of the offspring
        :type parents: tuple
        """
        self._count_individuals += 1

        offspring = Individual(
            genes=genes,
            idx=self._count_individuals,
            parents=parents,
            config=self.config,
        )

        self.offspring = np.concatenate((self.offspring, np.array([offspring])))

    # -------------------
    # Mutation
    # -------------------

    def _base_mutation(self):
        """
        Apply base mutation to all offspring individuals.

        Applies mutation to each offspring individual using the configured
        mutation rate, which can be static, gene-based, or population-based.
        """
        for ind in self.offspring:
            ind.mutate(mutation_prob=self._mutation_rate)

    def _adaptative_mutation(self):
        """
        Implement adaptive mutation with dynamic rate adjustment.

        Adjusts the mutation rate based on the coefficient of variation of
        population fitness. If diversity decreases, mutation rate increases;
        if diversity increases, mutation rate decreases. The mutation rate
        is capped at 0.9 to prevent excessive mutation.
        """

        if self.generation % self._generation_adaptative == 0:
            temp_cv = self._calculate_coefficient_variation()

            if temp_cv < self._coefficient_variation:
                self._mutation_rate *= 1.1
            elif temp_cv > self._coefficient_variation:
                self._mutation_rate /= 1.1

            self._coefficient_variation = temp_cv

        if self._mutation_rate > 1:
            self._mutation_rate = 0.9

        self._base_mutation()

    def _calculate_coefficient_variation(self):
        """
        Calculate the coefficient of variation of population fitness.

        Computes the ratio of standard deviation to mean fitness, which
        indicates the relative variability in fitness values across the population.

        :return: Coefficient of variation (std/mean)
        :rtype: float
        """
        mean_fitness = sum([ind.fitness for ind in self.population]) / len(
            self.population
        )
        std_fitness = sqrt(
            sum([(ind.fitness - mean_fitness) ** 2 for ind in self.population])
            / len(self.population)
        )

        return std_fitness / mean_fitness

    # -------------------
    # Update fitness
    # -------------------
    def _update_fitness(self, population):
        """
        Update fitness values for individuals in the given population.

        Evaluates fitness for all individuals that don't already have a fitness value,
        using the configured evaluator function or class.

        :param population: Array of individuals to evaluate
        :type population: numpy.ndarray
        """
        for ind in population:
            if ind.fitness is None:
                ind.fitness = self.evaluator(ind.genes)

    # -------------------
    # Replacement
    # -------------------
    def _random_replacement(self):
        """
        Implement random replacement strategy.

        Combines the current population with offspring and randomly selects
        individuals to form the next generation, maintaining the population size.
        """
        self.population = np.concatenate((self.population, self.offspring))

        self.population = np.random.choice(
            self.population, size=self.population_size, replace=False
        )

        self.offspring = np.array([], dtype=Individual)

    def _offspring_replacement(self):
        """
        Implement offspring-only replacement strategy.

        Completely replaces the current population with the offspring,
        discarding all individuals from the previous generation.
        """
        self.population = self.offspring
        self.offspring = np.array([], dtype=Individual)

    def _elitist_replacement(self):
        """
        Implement elitist replacement strategy.

        Selects the best individuals from the combined population and offspring
        to form the next generation, ensuring the best solutions are preserved.
        """
        self.population = np.concatenate((self.population, self.offspring))
        temp = np.argsort(
            np.array([self.extract_fitness(obj) for obj in self.population])
        )

        if self._optimization_objective == "max":
            temp = self.population_size - 1 - temp

        self.population = self.population[temp][: self.population_size]

        self.offspring = np.array([], dtype=Individual)

    def _elitist_stochastic_replacement(self):
        """
        Implement elitist stochastic replacement strategy.

        Assigns replacement probabilities to individuals based on their fitness values,
        with better individuals having higher probability of survival to the next generation.
        """
        self.population = np.concatenate((self.population, self.offspring))

        fitness_values = np.array(
            [self.extract_fitness(obj) for obj in self.population]
        )
        temp = np.argsort(fitness_values)
        fitness_values = fitness_values[temp]

        self.population = self.population[temp]

        max_fitness = self.population.item(-1).fitness
        min_fitness = self.population.item(0).fitness

        if self._optimization_objective == "max":
            ref_fitness = min_fitness
            self._replacement_probs = fitness_values - ref_fitness

        else:
            ref_fitness = max_fitness
            self._replacement_probs = ref_fitness - fitness_values

        self._replacement_probs[np.where(self._replacement_probs == 0)] = (
            self._replacement_probs[
                np.argmin(
                    self._replacement_probs[np.where(self._replacement_probs > 0)]
                )
            ]
            / 2
        )

        self._replacement_probs = self._replacement_probs / sum(self._replacement_probs)

        self.population = np.random.choice(
            self.population,
            size=self.population_size,
            replace=False,
            p=self._replacement_probs,
        )

        self.offspring = []

    # -------------------
    # Sort function
    # -------------------
    @staticmethod
    def extract_fitness(obj):
        """
        Extract fitness value from an individual.

        Helper method to extract the fitness value from an individual object,
        used for sorting and comparison operations.

        :param obj: Individual object
        :type obj: Individual
        :return: Fitness value of the individual
        :rtype: float
        """
        return obj.fitness
