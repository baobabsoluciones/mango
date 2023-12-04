import logging
from random import choices, choice

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
        self.offspring = []
        self._count_individuals = 0
        self.best = None

        # selection params
        self._selection_type = self.config("selection")
        self.selection_probs = []

        # crossover params
        self._crossover_type = self.config("crossover")

        # replacement params
        self._replacement_type = self.config("replacement")

    def run(self):
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

    def init_population(self):
        self.population = [
            Individual.create_random_individual(i + 1, self.config)
            for i in range(self.population_size)
        ]

        self._count_individuals = len(self.population)

    def selection(self):
        if self._selection_type == "roulette":
            self._roulette_selection()
        else:
            raise NotImplementedError("Selection method not implemented")

    def crossover(self):
        if self._crossover_type == "mask":
            self._mask_crossover()
        else:
            raise NotImplementedError("Crossover method not implemented")

    def mutate(self):
        for ind in self.offspring:
            ind.mutate()

    def update_population(self):
        self._update_fitness(self.offspring + self.population)

    def replace(self):
        if self._replacement_type == "elitist":
            self._elitist_replacement()
        else:
            raise NotImplementedError("Replacement method not implemented")

    def update_best(self):
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
        pass

    # -------------------
    # Internal methods.
    # -------------------
    # Selection
    # -------------------
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
