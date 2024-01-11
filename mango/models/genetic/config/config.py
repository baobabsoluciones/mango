from mango.config import BaseConfig, ConfigParameter
from mango.models.genetic.const import (
    INDIVIDUAL_ENCODINGS,
    SELECTION_METHODS,
    CROSSOVER_METHODS,
    REPLACEMENT_METHODS,
    MUTATION_CONTROL_METHODS,
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
            ConfigParameter(
                "mutation_control",
                str,
                validate=MUTATION_CONTROL_METHODS,
                default="none",
            ),
            ConfigParameter(
                "mutation_base_rate",
                float,
                0.1,
                required=False,
                min_value=0,
                max_value=1,
            ),
            ConfigParameter(
                "checkpoint_save_path",
                str,
                None,
                required=False,
            ),
            ConfigParameter(
                "checkpoint_save_period",
                int,
                None,
                required=False,
                min_value=0,
            ),
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
            ConfigParameter(
                "rank_pressure",
                float,
                2.0,
                required=False,
                min_value=1.0,
                max_value=2.0,
            ),
        ],
        "crossover": [
            ConfigParameter("offspring_size", int, 100, required=False),
            ConfigParameter("blend_expansion", float, 0.1, required=False),
        ],
        "mutation": [
            ConfigParameter("generation_adaptative", int, default=10, required=False),
        ],
    }

    def __init__(self, filename):
        super().__init__(filename, self.__params)
        # Add validation for checkpoint_save_path and checkpoint_save_rate
        if (self("checkpoint_save_path") is None) != (
            self("checkpoint_save_period") is None
        ):
            raise ValueError(
                "Both 'checkpoint_save_path' and 'checkpoint_save_period' must be set together."
            )
