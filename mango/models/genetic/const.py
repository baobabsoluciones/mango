""" This file contains the constants used to validate the configuration parameters for the config """

INDIVIDUAL_ENCODINGS = ["real", "binary", "integer", "non-negative-real"]
SELECTION_METHODS = ["roulette", "tournament", "rank", "random", "stochastic_rank"]
CROSSOVER_METHODS = ["mask", "generalized", "one-split", "two-split"]

REPLACEMENT_METHODS = [
    "elitist",
    "elitist-pseudo-random",
    "random",
    "only-offspring",
]
