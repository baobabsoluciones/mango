from collections import Counter
from math import sqrt
from random import randint

import numpy as np

from mango.models.genetic.individual import Individual
from mango.models.genetic.shared.exceptions import GeneticDiversity, ConfigurationError


class Population:
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

    def run(self):
        """
        Main method to run the Genetic Algorithm.
        """
        try:
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

        except GeneticDiversity:
            print(
                f"Exited due to low genetic diversity on generation: {self.generation + 1}"
            )

    def continue_running(self, generations: int):
        """
        Method to rerun the Genetic Algorithm from the point it stopped for more generations.

        :param generations: Number of generations to run the Genetic Algorithm.
        """
        self.max_generations += generations
        self.run()

    def init_population(self):
        """
        Method to initialize the population in the Genetic Algorithm
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
        Method to run the selection phase of the Genetic Algorithm.
        Selection method actually just creates the list of probabilities of a given individual
        to be chosen on the crossover phase. The values of said probabilities depend on the selection method.

        Currently, there is four selection methods implemented:
            - Roulette
            - Rank
            - Random
            - Tournament
        """
        if self._selection_type == "roulette":
            self._roulette_selection()
        elif self._selection_type == "elitism":
            self._elitism_selection()
        elif self._selection_type == "random":
            self._random_selection()
        elif self._selection_type == "tournament":
            self._tournament_selection()
        elif self._selection_type == "rank":
            self._rank_selection()
        elif self._selection_type == "order":
            self._order_selection()
        else:
            raise NotImplementedError("Selection method not implemented")

    def crossover(self):
        """
        Method to run the crossover phase of the Genetic Algorithm

        Currently, there is one crossover method implemented:
            - Mask
        """
        if self._crossover_type == "mask":
            self._mask_crossover()
        elif self._crossover_type == "blend":
            self._blend_crossover()
        elif self._crossover_type == "one-split":
            self._one_split_crossover()
        elif self._crossover_type == "two-split":
            self._two_split_crossover()
        elif self._crossover_type == "linear":
            self._linear_crossover()
        elif self._crossover_type == "flat":
            self._flat_crossover()
        elif self._crossover_type == "gaussian":
            self._gaussian_crossover()
        else:
            raise NotImplementedError("Crossover method not implemented")

    def mutate(self):
        """
        Method to run the mutation phase of the Genetic Algorithm.

        The actual implementation of the mutation is done in the individual as it is heavily dependent on the encoding.
        """
        if self._mutation_type in ["none", "gene-based", "population-based"]:
            self._base_mutation()
        elif self._mutation_type == "adaptative":
            self._adaptative_mutation()

    def update_population(self):
        """
        Method to make sure that all individuals in the population have a fitness value.
        """
        self._update_fitness(np.concatenate((self.population, self.offspring)))

    def replace(self):
        """
        Method to run the replacement phase of the Genetic Algorithm.

        Currently, there is one replacement method implemented:
            - Elitist
        """
        if self._replacement_type == "elitist":
            self._elitist_replacement()
        elif self._replacement_type == "only-offspring":
            self._offspring_replacement()
        elif self._replacement_type == "random":
            self._random_replacement()
        elif self._replacement_type == "elitist-stochastic":
            self._elitist_stochastic_replacement()
        else:
            raise NotImplementedError("Replacement method not implemented")

    def update_best(self):
        """
        Method to update the best adapted individual on the population.
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
        Method to implement stop conditions based on information about the population or the genetic diversity.
        """
        # TODO: add more stop conditions
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
        Method to perform random selection.
        In this case the probability to select any individual is equal.
        """
        self.selection_probs = np.ones(self.population_size) / self.population_size

    def _elitism_selection(self):
        """
        Method to perform elitism selection.

        Based on the config parameter rank_size (k), the k-best individuals are selected for crossover.
        That means that k individuals will have the same selection probability and the rest will have zero.

        This selection method could create a diversity problem on the long run
        if k is small compared to the population size.
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
        Method to perform rank selection.
        """
        fitness_values = np.array(
            [self.extract_fitness(obj) for obj in self.population]
        )
        temp = np.argsort(fitness_values)

        if self._optimization_objective == "min":
            temp = self.population_size - 1 - temp

        self.population = self.population[temp]

        self.selection_probs = (
            1
            / self.population_size
            * (2 * self._rank_pressure - 2)
            * np.arange(self.population_size)
            / (self.population_size - 1)
        )

    def _order_selection(self):
        """
        Method to perform order selection
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
        Method to perform roulette selection.
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
        pass

    def _select_parents(self, n: int = 2) -> tuple:
        """
        Method to select n parents from the population based on the selection probabilities
        established on the selection phase.

        :param int n: number of parents to select from the population. It defaults to two,
        but it has the capability to be a different number.
        :return: a tuple with the selected parents
        :rtype: tuple
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
    def _mask_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            random_indices = np.random.choice([0, 1], size=self._gene_length)
            offspring_1 = np.where(random_indices, p1.genes, p2.genes)
            random_indices = 1 - random_indices
            offspring_2 = np.where(random_indices, p1.genes, p2.genes)

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _flat_crossover(self):
        """
        This method implements Radcliffe's flat crossover (1990).

        It creates two child individuals with each gene created randomly inside the interval defined by the parents.
        """
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            max_c = np.maximum(p1.genes, p2.genes)
            min_c = np.minimum(p1.genes, p2.genes)

            offspring_1 = np.random.uniform(min_c, max_c)
            offspring_2 = np.random.uniform(min_c, max_c)

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _linear_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            offspring_1 = (p1.genes + p2.genes) / 2
            offspring_2 = 1.5 * p1.genes - 0.5 * p2.genes
            offspring_3 = -0.5 * p1.genes + 1.5 * p2.genes

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)
            self._add_offspring(offspring_3, p1, p2)

    def _blend_crossover(self):
        """
        This method implements the Blend crossover proposed by Eshelman and Schaffer (1993).

        The alpha parameter defined for the crossover is defined as blend_expansion parameter.
        With a value of 0 it is equal toa a Radcliffe's flat crossover while a value of 0.5 is
        equal to Wright's linear crossover intervals (but the values are randomly selected).

        The main difference with Wrights linear crossover is that only two childs are created instead of two
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

            offspring_2 = interval - offspring_1

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _one_split_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            split = randint(1, self._gene_length - 2)
            offspring_1 = np.concatenate((p1.genes[:split], p2.genes[split:]))
            offspring_2 = np.concatenate((p2.genes[:split], p1.genes[split:]))

            self._add_offspring(offspring_1, p1, p2)
            self._add_offspring(offspring_2, p1, p2)

    def _two_split_crossover(self):
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

    def _gaussian_crossover(self):
        """
        Method to perform the Gaussian crossover, also known as Unimodal Normal Distribution Crossover
        or UNDX, proposed by Ono (2003).

        This method is only implemented for real encoding.
        """

        # TODO: review bad results for huge genomes
        while len(self.offspring) < len(self.population):
            # First we select three parents. The first two are going to be the primary search space,
            # while the third is going to create the secondary search space with the midpoint between parents 1 and 2
            p1, p2, p3 = self._select_parents(n=3)

            # TODO: this could be changed by config in the future
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
        """ """
        raise NotImplementedError("Morphology crossover not implemented")

    def _add_offspring(self, genes, *parents):
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
        This is the based mutation applied for the base mutation rate or
        the ones calculated based on gene length or population size.
        """
        for ind in self.offspring:
            ind.mutate(mutation_prob=self._mutation_rate)

    def _adaptative_mutation(self):
        """
        Method to implement adaptative mutation operator.

        For the calculation of this
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
        Method to calculate the coefficient of variation of the fitness of the population.
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
        for ind in population:
            if ind.fitness is None:
                ind.fitness = self.evaluator(ind.genes)

    # -------------------
    # Replacement
    # -------------------
    def _elitist_replacement(self):
        """
        The best adapted individuals on the population are the ones that pass to the next generation
        """
        self.population = np.concatenate((self.population, self.offspring))
        temp = np.argsort(
            np.array([self.extract_fitness(obj) for obj in self.population])
        )

        if self._optimization_objective == "max":
            temp = self.population_size - 1 - temp

        self.population = self.population[temp][: self.population_size]

        self.offspring = np.array([], dtype=Individual)

    def _offspring_replacement(self):
        """
        The population gets completely substituted by the offspring.
        """
        self.population = self.offspring
        self.offspring = np.array([], dtype=Individual)

    def _random_replacement(self):
        """
        Method to implement a random replacement operator.
        """
        self.population = np.concatenate((self.population, self.offspring))

        self.population = np.random.choice(
            self.population, size=self.population_size, replace=False
        )

        self.offspring = np.array([], dtype=Individual)

    def _elitist_stochastic_replacement(self):
        """
        Method to implement an elitist stochastic replacement operator.

        In this case each individual has a replacement probability based on their fitness.
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
        return obj.fitness
