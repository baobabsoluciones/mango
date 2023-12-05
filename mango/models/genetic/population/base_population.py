import logging
from random import choices, choice, sample

from mango.models.genetic.individual import Individual


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

        # selection params
        self._selection_type = self.config("selection")
        self._rank_size = self.config("rank_size")
        self._tournament_size = self.config("tournament_size")
        self._tournament_winnings = self.config("tournament_winnings")

        # crossover params
        self._crossover_type = self.config("crossover")
        self._offspring_size = self.config("offspring_size") or self.population_size

        # replacement params
        self._replacement_type = self.config("replacement")

    def run(self):
        """
        Main method to run the Genetic Algorithm.
        """
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
        elif self._selection_type == "rank":
            self._rank_selection()
        elif self._selection_type == "random":
            self._random_selection()
        elif self._selection_type == "tournament":
            self._tournament_selection()
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
        else:
            raise NotImplementedError("Crossover method not implemented")

    def mutate(self):
        """
        Method to run the mutation phase of the Genetic Algorithm.

        The actual implementation of the mutation is done in the individual as it is heavily dependent on the encoding.
        """
        for ind in self.offspring:
            ind.mutate()

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
        pass

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

    def _rank_selection(self):
        """
        Method to perform rank selection.

        Based on the config parameter rank_size (k), the k-best individuals are selected for crossover.
        That means that k individuals will have the same selection probability and the rest will have zero.

        This selection method could create a diversity problem on the long run
        if k is small compared to the population size.
        """
        if self._optimization_objective == "max":
            temp = sorted(self.population, key=lambda x: x.fitness, reverse=True)[
                : self._rank_size
            ]
        else:
            temp = sorted(self.population, key=lambda x: x.fitness)[: self._rank_size]

        self.selection_probs = [1 if ind in temp else 0 for ind in self.population]

    def _stochastic_rank_selection(self):
        pass

    def _roulette_selection(self):
        if self._optimization_objective == "max":
            ref_fitness = min(self.population, key=lambda x: x.fitness).fitness
            self.selection_probs = [
                ind.fitness - ref_fitness + 1 for ind in self.population
            ]
        else:
            ref_fitness = max(self.population, key=lambda x: x.fitness).fitness
            self.selection_probs = [
                ref_fitness - ind.fitness + 1 for ind in self.population
            ]

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

    def _select_parents(self):
        p1 = choices(self.population, weights=self.selection_probs)[0]
        p2 = p1

        k = 0
        while p1 == p2:
            p2 = choices(self.population, weights=self.selection_probs)[0]
            k += 1
            if k >= 2 * self.population_size:
                # TODO: add custom exception here
                raise Exception("Could not find two different parents")
        return p1, p2

    # -------------------
    # Crossover
    # -------------------
    def _mask_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            genes_offspring = [choice([g1, g2]) for g1, g2 in zip(p1.genes, p2.genes)]

            self._count_individuals += 1

            offspring = Individual(
                genes=genes_offspring,
                idx=self._count_individuals,
                parents=(p1, p2),
                config=self.config,
            )

            self.offspring.append(offspring)

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
