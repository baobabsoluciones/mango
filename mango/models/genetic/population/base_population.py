from collections import Counter
from collections import Counter
from random import choices, choice, sample, randint, uniform

from mango.models.genetic.individual import Individual
from mango.models.genetic.shared.exceptions import GeneticDiversity


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

        # selection params
        self._selection_type = self.config("selection")
        self._elitism_size = self.config("elitism_size")
        self._tournament_size = self.config("tournament_size")
        self._tournament_winnings = self.config("tournament_winnings")

        # crossover params
        self._crossover_type = self.config("crossover")
        self._offspring_size = self.config("offspring_size") or self.population_size
        self._generalized_expansion = self.config("generalized_expansion")

        # replacement params
        self._replacement_type = self.config("replacement")

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
        elif self._selection_type == "rank":
            self._elitism_selection()
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
        elif self._crossover_type == "generalized":
            self._generalized_crossover()
        elif self._crossover_type == "one-split":
            self._one_split_crossover()
        elif self._crossover_type == "two-split":
            self._two_split_crossover()
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
        pass

    def _roulette_selection(self):
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

    def _select_parents(self):
        p1 = choices(self.population, weights=self.selection_probs)[0]
        p2 = p1

        k = 0
        while p1 == p2:
            p2 = choices(self.population, weights=self.selection_probs)[0]
            k += 1
            if k >= 2 * self.population_size:
                # TODO: add custom exception here
                raise GeneticDiversity
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

    def _generalized_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            max_c = [max(g1, g2) for g1, g2 in zip(p1.genes, p2.genes)]
            min_c = [min(g1, g2) for g1, g2 in zip(p1.genes, p2.genes)]
            interval = [max_c[i] - min_c[i] for i in range(len(max_c))]

            genes_offspring = [
                uniform(
                    min_c[i] - interval[i] * self._generalized_expansion,
                    max_c[i] + interval[i] * self._generalized_expansion,
                )
                for i in range(len(max_c))
            ]

            self._count_individuals += 1

            offspring = Individual(
                genes=genes_offspring,
                idx=self._count_individuals,
                parents=(p1, p2),
                config=self.config,
            )

            self.offspring.append(offspring)

    def _one_split_crossover(self):
        while len(self.offspring) < len(self.population):
            p1, p2 = self._select_parents()

            split = randint(1, self._gene_length - 2)
            genes_offspring = p1.genes[:split] + p2.genes[split:]

            self._count_individuals += 1

            offspring = Individual(
                genes=genes_offspring,
                idx=self._count_individuals,
                parents=(p1, p2),
                config=self.config,
            )

            self.offspring.append(offspring)

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
