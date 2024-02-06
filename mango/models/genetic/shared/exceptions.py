class GeneticDiversity(Exception):
    """
    Exception raised when the genetic diversity of the population is too low.
    """

    pass


class ConfigurationError(Exception):
    """
    Exception raised when the configuration of the genetic algorithm is not valid.
    """

    pass
