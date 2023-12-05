from mango.config import BaseConfig, ConfigParameter
from mango.models.genetic.const import (
    INDIVIDUAL_ENCODINGS,
    SELECTION_METHODS,
    CROSSOVER_METHODS,
    REPLACEMENT_METHODS,
)


class GeneticBaseConfig(BaseConfig):
    __params = {
        "main": [
            ConfigParameter("population_size", int, 100),
            ConfigParameter("max_generations", int, 500),
            ConfigParameter("optimization_objective", str, validate=["max", "min"]),
            ConfigParameter("selection", str, validate=SELECTION_METHODS),
            ConfigParameter("crossover", str, validate=CROSSOVER_METHODS),
            ConfigParameter("replacement", str, validate=REPLACEMENT_METHODS),
            ConfigParameter("mutation_rate", float, 0.1),
        ],
        "individual": [
            ConfigParameter("encoding", str, validate=INDIVIDUAL_ENCODINGS),
            ConfigParameter("gene_length", int, default=0),
            ConfigParameter("gene_min_value", float),
            ConfigParameter("gene_max_value", float),
        ],
        "selection": [
            ConfigParameter("elitism_size", int, 20, required=False),
            ConfigParameter("tournament_size", int, 2, required=False),
            ConfigParameter("tournament_winnings", float, 0.5, required=False),
        ],
        "crossover": [
            ConfigParameter("offspring_size", int, 100, required=False),
        ],
    }

    def __init__(self, filename):
        super().__init__(filename, self.__params)
