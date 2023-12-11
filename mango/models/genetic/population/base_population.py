from collections import Counter
from math import sqrt
from random import choices, choice, sample, randint, uniform, gauss

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
        self.population = []
        self.selection_probs = []
        self.offspring = []
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
        self._blend_expansion = self.config("generalized_expansion")
        self._morphology_parents = self.config("morphology_parents")

        # mutation params
        self._mutation_type = self.config("mutation_control")
        self._mutation_rate = self.config("mutation_base_rate")

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
        self.population = [
            Individual.create_random_individual(i + 1, self.config)
            for i in range(self.population_size)
        ]

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
        self._update_fitness(self.offspring + self.population)

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
        self.selection_probs = [1 for _ in self.population]

    def _elitism_selection(self):
        """
        Method to perform elitism selection.

        Based on the config parameter rank_size (k), the k-best individuals are selected for crossover.
        That means that k individuals will have the same selection probability and the rest will have zero.

        This selection method could create a diversity problem on the long run
        if k is small compared to the population size.
        """
        if self._optimization_objective == "max":
            temp = sorted(self.population, key=lambda x: x.fitness, reverse=True)[
                : self._elitism_size
            ]
        else:
            temp = sorted(self.population, key=lambda x: x.fitness)[
                : self._elitism_size
            ]

        self.selection_probs = [1 if ind in temp else 0 for ind in self.population]

    def _rank_selection(self):
        """
        Method to perform rank selection.
        """
        if self._optimization_objective == "max":
            temp = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        else:
            temp = sorted(self.population, key=lambda x: x.fitness)

        self.selection_probs = [
            1
            / len(self.population)
            * (
                self._rank_pressure
                - (2 * self._rank_pressure - 2)
                * ((temp.index(ind)) / (len(self.population) - 1))
            )
            for ind in self.population
        ]

    def _order_selection(self):
        """
        Method to perform order selection
        """
        if self._optimization_objective == "max":
            temp = sorted(self.population, key=lambda x: x.fitness)
        else:
            temp = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        self.selection_probs = [
            (temp.index(ind) + 1)
            / (len(self.population) * (len(self.population) + 1) / 2)
            for ind in self.population
        ]

    def _roulette_selection(self):
        """
        Method to perform roulette selection.
        """
        max_fitness = max(self.population, key=lambda x: x.fitness).fitness
        min_fitness = min(self.population, key=lambda x: x.fitness).fitness

        if self._optimization_objective == "max":
            ref_fitness = min_fitness

            self.selection_probs = [
                ind.fitness - ref_fitness for ind in self.population
            ]

        else:
            ref_fitness = max_fitness
            self.selection_probs = [
                ref_fitness - ind.fitness for ind in self.population
            ]

        # self.selection_probs = [
        #     x + (max_fitness - min_fitness) / 1000 if x == 0 else x
        #     for x in self.selection_probs
        # ]

    def _tournament_selection(self):
        wins = dict()
        for _ in range(self._offspring_size * 2):
            individuals = sample(self.population, k=self._tournament_size)
            if self._optimization_objective == "max":
                winner = max(individuals, key=lambda x: x.fitness)
            else:
                winner = min(individuals, key=lambda x: x.fitness)
            wins[winner.idx] = wins.get(winner.idx, 0) + 1

        self.selection_probs = [wins.get(ind.idx, 0) for ind in self.population]

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

        p1 = choices(self.population, weights=self.selection_probs)[0]
        parents = [p1 for _ in range(n)]

        for i in range(1, n):
            k = 0
            pi = parents[i]
            while pi in parents:
                pi = choices(self.population, weights=self.selection_probs)[0]
                k += 1
                if k >= 2 * self.population_size:
                    raise GeneticDiversity

            parents[i] = pi

        return tuple(parents)

    # -------------------
    # Crossover
    # -------------------
    def _mask_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            genes_offspring = [choice([g1, g2]) for g1, g2 in zip(p1.genes, p2.genes)]

            self._add_offspring(genes_offspring, p1, p2)

    def _flat_crossover(self):
        """
        This method implements Radcliffe's flat crossover (1990).

        It creates two child individuals with each gene created randomly inside the interval defined by the parents.
        """
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            max_c = [max(g1, g2) for g1, g2 in zip(p1.genes, p2.genes)]
            min_c = [min(g1, g2) for g1, g2 in zip(p1.genes, p2.genes)]

            genes_offspring_1 = [uniform(min_c[i], max_c[i]) for i in range(len(max_c))]
            genes_offspring_2 = [uniform(min_c[i], max_c[i]) for i in range(len(max_c))]

            self._add_offspring(genes_offspring_1, p1, p2)
            self._add_offspring(genes_offspring_2, p1, p2)

    def _linear_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            genes_offspring_1 = [
                (p1.genes[i] + p2.genes[i]) / 2 for i in range(len(p1.genes))
            ]

            genes_offspring_2 = [
                1.5 * p1.genes[i] - 0.5 * p2.genes[i] for i in range(len(p1.genes))
            ]

            genes_offspring_3 = [
                -0.5 * p1.genes[i] + 1.5 * p2.genes[i] for i in range(len(p1.genes))
            ]

            self._add_offspring(genes_offspring_1, p1, p2)
            self._add_offspring(genes_offspring_2, p1, p2)
            self._add_offspring(genes_offspring_3, p1, p2)

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

            max_c = [max(g1, g2) for g1, g2 in zip(p1.genes, p2.genes)]
            min_c = [min(g1, g2) for g1, g2 in zip(p1.genes, p2.genes)]
            interval = [max_c[i] - min_c[i] for i in range(len(max_c))]

            genes_offspring_1 = [
                uniform(
                    min_c[i] - interval[i] * self._blend_expansion,
                    max_c[i] + interval[i] * self._blend_expansion,
                )
                for i in range(len(max_c))
            ]
            genes_offspring_2 = [
                interval[i] - genes_offspring_1[i] for i in range(len(max_c))
            ]

            self._add_offspring(genes_offspring_1, p1, p2)
            self._add_offspring(genes_offspring_2, p1, p2)

    def _one_split_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            split = randint(1, self._gene_length - 2)
            genes_offspring = p1.genes[:split] + p2.genes[split:]

            self._add_offspring(genes_offspring, p1, p2)

    def _two_split_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            split_1 = randint(1, self._gene_length - 2)
            split_2 = randint(1, self._gene_length - 2)
            while split_1 == split_2:
                split_2 = randint(1, self._gene_length - 2)

            split_1, split_2 = sorted([split_1, split_2])

            genes_offspring = (
                p1.genes[:split_1] + p2.genes[split_1:split_2] + p1.genes[split_2:]
            )

            self._add_offspring(genes_offspring, p1, p2)

    def _gaussian_crossover(self):
        """
        Method to perform the Gaussian crossover, also known as Unimodal Normal Distribution Crossover
        or UNDX, proposed by Ono (2003).

        This method is only implemented for real encoding.
        """
        while len(self.offspring) < len(self.population):
            # First we select three parents. The first two are going to be the primary search space,
            # while the third is going to create the secondary search space with the midpoint between parents 1 and 2
            p1, p2, p3 = self._select_parents(n=3)

            # TODO: this could be changed by config in the future
            # This are parameters for the randomness of the gaussian distribution used in the crossover.
            sigma_eta = 0.35 / sqrt(self._gene_length)
            sigma_xi = 1 / 4

            # We calculate the distance vector, the unit distance vector and the mean point in the primary search space
            dist_p1p2 = [p2.genes[i] - p1.genes[i] for i in range(self._gene_length)]
            dist_p1p2_norm = sqrt(
                sum([dist_p1p2[i] * dist_p1p2[i] for i in range(self._gene_length)])
            )

            dist_p1p2_unit = [
                dist_p1p2[i] / dist_p1p2_norm for i in range(self._gene_length)
            ]

            mean_p1p2 = [
                (p1.genes[i] + p2.genes[i]) / 2 for i in range(self._gene_length)
            ]

            # We calculate the distance vector between the third parent and the first one
            dist_p3p1 = [p3.genes[i] - p1.genes[i] for i in range(self._gene_length)]

            # This proportion is used to calculate the vector between the third parent and the primary search space
            proportion = sum(
                [dist_p1p2[i] * dist_p3p1[i] for i in range(self._gene_length)]
            ) / sum([dist_p1p2[i] * dist_p1p2[i] for i in range(self._gene_length)])

            # Formula for the distance vector from the third parent to the primary search line.
            # This formula should be able to be applied to any dimension space and this distance vector is orthogonal
            # to the distance vector between the first and second parent.
            distance_vector = [
                dist_p3p1[i] - proportion * dist_p1p2[i]
                for i in range(self._gene_length)
            ]

            # Distance
            distance = sqrt(
                sum(
                    [
                        distance_vector[i] * distance_vector[i]
                        for i in range(self._gene_length)
                    ]
                )
            )

            # Random vector based in the basis of the distance between the third parent and the primary search space
            # then we substract the component of the primary search
            t = [
                gauss(0, (distance * sigma_eta) ** 2) for _ in range(self._gene_length)
            ]

            t_dot_parents = sum(
                [t[i] * dist_p1p2_unit[i] for i in range(self._gene_length)]
            )

            t = [t[i] - t_dot_parents * dist_p1p2[i] for i in range(self._gene_length)]

            # and we add the parallel component
            t = [
                t[i] + gauss(0, sigma_xi) * dist_p1p2[i]
                for i in range(self._gene_length)
            ]

            # We create two children
            offspring_1 = [mean_p1p2[i] + t[i] for i in range(self._gene_length)]
            offspring_2 = [mean_p1p2[i] - t[i] for i in range(self._gene_length)]

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

        self.offspring.append(offspring)

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

        In this case the mutation rate is evaluated every X generations.
        If the genetic diversity has decreased then the mutation rate is increased,
        if the genetic diversity has decreased then the mutation rate is lowered.
        This should help the algorithm converge faster in some cases but
        keep some exploration alive in order to avoid local optima.
        """
        raise NotImplementedError("Adaptative mutation not implemented")

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
        if self._optimization_objective == "max":
            self.population = sorted(
                self.population + self.offspring, key=lambda x: x.fitness, reverse=True
            )
        else:
            self.population = sorted(
                self.population + self.offspring, key=lambda x: x.fitness
            )

        self.population = self.population[: self.population_size]
        self.offspring = []

    def _offspring_replacement(self):
        """
        The population gets completely substituted by the offspring.
        """
        self.population = list(self.offspring)
        self.offspring = []

    def _random_replacement(self):
        """
        Method to implement a random replacement operator.
        """
        self.population = sample(
            self.population + self.offspring, k=self.population_size
        )
        self.offspring = []

    def _elitist_stochastic_replacement(self):
        """
        Method to implement an elitist stochastic replacement operator.

        In this case each individual has a replacement probability based on their fitness.
        """
        self.population = self.population + self.offspring

        max_fitness = max(self.population, key=lambda x: x.fitness).fitness
        min_fitness = min(self.population, key=lambda x: x.fitness).fitness

        if self._optimization_objective == "max":
            self.population = sorted(
                self.population + self.offspring, key=lambda x: x.fitness, reverse=True
            )

            ref_fitness = min_fitness

            self._replacement_probs = [
                ind.fitness - ref_fitness for ind in self.population
            ]

        else:
            self.population = sorted(
                self.population + self.offspring, key=lambda x: x.fitness
            )

            ref_fitness = max_fitness

            self._replacement_probs = [
                ref_fitness - ind.fitness for ind in self.population
            ]

        temp = []
        for _ in range(self.population_size):
            individual = choices(self.population, weights=self._replacement_probs)[0]
            temp.append(individual)
            position = self.population.index(individual)
            del self._replacement_probs[position]
            self.population.remove(individual)

        self.population = list(temp)
        self.offspring = []
