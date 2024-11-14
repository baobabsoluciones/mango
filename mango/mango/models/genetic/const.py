""" This file contains the constants used to validate the configuration parameters for the config """

INDIVIDUAL_ENCODINGS = ["real", "binary", "integer", "categorical"]
SELECTION_METHODS = ["roulette", "tournament", "rank", "random", "elitism", "order"]
CROSSOVER_METHODS = [
    "mask",
    "blend",
    "one-split",
    "two-split",
    "linear",
    "flat",
    "gaussian",
]

MUTATION_CONTROL_METHODS = ["static", "adaptative", "gene-based", "population-based"]

REPLACEMENT_METHODS = [
    "elitist",
    "elitist-stochastic",
    "random",
    "only-offspring",
]
