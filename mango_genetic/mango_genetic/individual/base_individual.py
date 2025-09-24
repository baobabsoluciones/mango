from random import uniform, randint

import numpy as np

from mango_genetic.config import GeneticBaseConfig


class Individual:
    """
    Base class representing an individual in a genetic algorithm.

    An individual contains genes (genetic information), fitness value, and metadata
    about its parents and configuration. The class supports different encoding types
    (real, binary, integer) and provides methods for mutation and genetic operations.

    :param genes: Initial genetic information as numpy array
    :type genes: numpy.ndarray, optional
    :param idx: Unique identifier for the individual
    :type idx: int
    :param parents: Tuple of parent individuals
    :type parents: tuple, optional
    :param config: Configuration object containing genetic algorithm parameters
    :type config: GeneticBaseConfig, optional
    """

    def __init__(self, genes=None, idx: int = 0, parents: tuple = None, config=None):
        self.fitness = None
        self.idx = idx
        self.config = config
        self.parents_idx = None
        self.parents = parents

        # Info to get from config file
        self.encoding = self.config("encoding")
        self.min_bound = self.config("gene_min_value")
        self.max_bound = self.config("gene_max_value")
        self.gene_length = self.config("gene_length")

        self.genes = genes

    @property
    def genes(self) -> np.ndarray:
        """
        Get the genetic information of the individual.

        Returns the numpy array containing the individual's genes. If bounds are set,
        gene values are automatically clipped to stay within the specified range.

        :return: Array containing the individual's genetic information
        :rtype: numpy.ndarray
        """
        return self._genes

    @genes.setter
    def genes(self, value: np.ndarray = None):
        """
        Set the genetic information of the individual.

        Sets the genes array and automatically clips values to bounds if they are defined.
        Updates the individual's hash after setting new genes.

        :param value: New genetic information as numpy array
        :type value: numpy.ndarray, optional
        """
        if (
            value is not None
            and self.max_bound is not None
            and self.min_bound is not None
        ):
            value[np.greater(value, self.max_bound)] = self.max_bound
            value[np.less(value, self.min_bound)] = self.min_bound
        self._genes = value
        if self._genes is not None:
            self._hash = self.__hash__()

    @property
    def idx(self) -> int:
        """
        Get the unique identifier of the individual.

        Returns the internal index used to distinguish between different individuals
        in the population.

        :return: Unique identifier for the individual
        :rtype: int
        """
        return self._idx

    @idx.setter
    def idx(self, value: int):
        """
        Set the unique identifier of the individual.

        :param value: New unique identifier
        :type value: int
        """
        self._idx = value

    @property
    def config(self) -> GeneticBaseConfig:
        """
        Get the configuration object for the individual.

        Returns the configuration object containing genetic algorithm parameters
        such as encoding type, bounds, and gene length.

        :return: Configuration object with genetic algorithm parameters
        :rtype: GeneticBaseConfig
        """
        return self._config

    @config.setter
    def config(self, value: GeneticBaseConfig):
        """
        Set the configuration object for the individual.

        :param value: New configuration object
        :type value: GeneticBaseConfig
        """
        self._config = value

    @property
    def parents(self) -> tuple:
        """
        Get the parent individuals of this individual.

        Returns a tuple containing the parent individuals. When parents are set,
        the parents_idx attribute is automatically updated with their indices.

        :return: Tuple of parent individuals
        :rtype: tuple
        """
        return self._parents

    @parents.setter
    def parents(self, value: tuple):
        """
        Set the parent individuals of this individual.

        Sets the parent individuals and automatically updates the parents_idx
        attribute with their indices.

        :param value: Tuple of parent individuals
        :type value: tuple
        """
        self._parents = value
        if self._parents is not None:
            self.parents_idx = tuple([p.idx for p in self.parents])

    @property
    def encoding(self) -> str:
        """
        Get the encoding type for the individual's genes.

        Returns the encoding type used for the genetic representation.
        Supported types include 'real', 'binary', and 'integer'.

        :return: Encoding type string
        :rtype: str
        """
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        """
        Set the encoding type for the individual's genes.

        :param value: Encoding type ('real', 'binary', or 'integer')
        :type value: str
        """
        self._encoding = value

    @property
    def fitness(self) -> float:
        """
        Get the fitness value of the individual.

        Returns the fitness value representing how well this individual
        performs according to the optimization objective.

        :return: Fitness value of the individual
        :rtype: float
        """
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        """
        Set the fitness value of the individual.

        :param value: New fitness value
        :type value: float
        """
        self._fitness = value

    @classmethod
    def create_random_individual(
        cls, idx: int, config: GeneticBaseConfig
    ) -> "Individual":
        """
        Create a new individual with randomly generated genes.

        Factory method that creates a new individual with the specified index
        and configuration, then generates random genes based on the encoding type.

        :param idx: Unique identifier for the new individual
        :type idx: int
        :param config: Configuration object containing genetic algorithm parameters
        :type config: GeneticBaseConfig
        :return: New individual with random genes
        :rtype: Individual

        Example:
            >>> config = GeneticBaseConfig()
            >>> individual = Individual.create_random_individual(idx=0, config=config)
        """
        ind = cls(idx=idx, config=config)

        ind.create_random_genes()
        return ind

    def create_random_genes(self):
        """
        Generate random genes for the individual based on the encoding type.

        Creates random genetic information according to the individual's encoding
        type and configuration bounds. The generated genes are stored in the
        genes attribute.

        Supported encoding types:
        - 'real': Continuous values between min_bound and max_bound
        - 'binary': Binary values (0 or 1)
        - 'integer': Integer values between min_bound and max_bound (inclusive)
        """
        if self.encoding == "real":
            self.genes = np.random.uniform(
                self.min_bound, self.max_bound, self.gene_length
            )

        elif self.encoding == "binary":
            self.genes = np.random.randint(0, 2, self.gene_length)

        elif self.encoding == "integer":
            self.genes = np.random.randint(
                self.min_bound, self.max_bound + 1, self.gene_length
            )

    def mutate(self, mutation_prob: float = None):
        """
        Mutate the individual's genes based on the encoding type.

        Applies mutation to the individual's genes with the specified probability.
        The mutation process continues until the probability check fails, meaning
        an individual can have multiple genes mutated in a single call.

        :param mutation_prob: Probability of mutation (0.0 to 1.0)
        :type mutation_prob: float

        Raises:
            NotImplementedError: If encoding type is not supported

        Example:
            >>> individual.mutate(mutation_prob=0.1)  # 10% chance of mutation
        """
        keep = True
        while keep:
            chance = uniform(0, 1)
            if chance <= mutation_prob:
                if self.encoding == "real":
                    self._mutate_real()
                elif self.encoding == "binary":
                    self._mutate_binary()
                elif self.encoding == "integer":
                    self._mutate_integer()
                else:
                    raise NotImplemented("Only real encoding is implemented for now")
                self._hash = self.__hash__()
            else:
                keep = False

    def _mutate_binary(self):
        """
        Mutate a binary-encoded gene by flipping its value.

        Performs a simple bit flip mutation on a randomly selected gene position.
        Changes 0 to 1 or 1 to 0.
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = 1 - self.genes[gene_position]

    def _mutate_integer(self):
        """
        Mutate an integer-encoded gene with a random value.

        Replaces a randomly selected gene with a new random integer value
        between the minimum and maximum bounds (inclusive).
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = randint(self.min_bound, self.max_bound)

    def _mutate_real(self):
        """
        Mutate a real-encoded gene with a random value.

        Replaces a randomly selected gene with a new random real value
        between the minimum and maximum bounds.
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = uniform(self.min_bound, self.max_bound)

    def dominates(self, other):
        """
        Check if this individual dominates another individual.

        This method should implement the logic to determine if one individual
        dominates another in multi-objective optimization problems.

        A solution dominates another when:
        - It is equal or better in all objectives
        - It is strictly better in at least one objective

        :param other: The other individual to compare against
        :type other: Individual
        :return: True if this individual dominates the other, False otherwise
        :rtype: bool

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplemented

    def copy(self) -> "Individual":
        """
        Create a deep copy of the individual.

        Creates a new individual with the same genetic information, index,
        configuration, and parent information as this individual.

        :return: A new individual that is a copy of this one
        :rtype: Individual

        Example:
            >>> original = Individual(genes=np.array([1, 2, 3]), idx=0)
            >>> copy = original.copy()
            >>> copy.genes[0] = 5
            >>> print(original.genes[0])  # Still 1
        """
        return Individual(
            genes=self.genes.copy(),
            idx=self.idx,
            config=self.config,
            parents=self.parents,
        )

    def __hash__(self):
        """
        Calculate the hash of the individual based on its genetic information.

        The hash is calculated from the byte representation of the individual's genes,
        ensuring that only individuals with identical genetic information have the same hash.
        This enables efficient comparison and storage in hash-based data structures.

        :return: Hash value based on the individual's genes
        :rtype: int
        """
        return hash(self.genes.tobytes())

    def __eq__(self, other) -> bool:
        """
        Compare two individuals for equality based on their genetic information.

        Uses pre-generated hash values for fast comparison. The comparison is based
        on genotype (genetic information) rather than phenotype (fitness/behavior),
        as individuals with different genotypes should be considered distinct for
        genetic diversity and crossover purposes.

        :param other: The other individual to compare against
        :type other: Individual
        :return: True if individuals have identical genetic information, False otherwise
        :rtype: bool
        """
        if isinstance(other, self.__class__):
            return self._hash == other._hash

        return False

    def __repr__(self):
        """
        Return a string representation of the individual.

        :return: String representation showing the individual's index
        :rtype: str
        """
        return f"Individual {self.idx}"
