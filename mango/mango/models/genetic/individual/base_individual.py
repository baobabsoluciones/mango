from random import uniform, randint

import numpy as np
from mango.models.genetic.config import GeneticBaseConfig


class Individual:
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
        Property that stores the :class:`numpy.ndarray` of genes in the individual

        If max_bound or min_bound are set, the genes values are clipped to the bounds.
        After setting the value for the genes the hash is updated.

        :param value: the new value for the genes (used on setting up the property)
        :type value: :class:`numpy.ndarray`
        :return: the genes value
        """
        return self._genes

    @genes.setter
    def genes(self, value: np.ndarray = None):
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
        Property that stores the internal idx of the Individual. This value is just to help
        distinguish between individuals.

        :param value: the new value for the idx (used on setting up the property)
        :type value: int
        :return: the idx value
        """
        return self._idx

    @idx.setter
    def idx(self, value: int):
        self._idx = value

    @property
    def config(self) -> GeneticBaseConfig:
        """
        Property that stores the config object for the individual

        :param value: the new value for the config (used on setting up the property)
        :type value: :class:`mango.models.genetic.config.GeneticBaseConfig`
        :return: the config value
        """
        return self._config

    @config.setter
    def config(self, value: GeneticBaseConfig):
        self._config = value

    @property
    def parents(self) -> tuple:
        """
        Property that stores the parents of the individual and their idx.

        During the setter the value for attribute parents_idx
        is updated as well as a tuple of the idx of all the parents

        :param value: the new value for the parents (used on setting up the property)
        :type value: tuple
        :return: the parents value
        """
        return self._parents

    @parents.setter
    def parents(self, value: tuple):
        self._parents = value
        if self._parents is not None:
            self.parents_idx = tuple([p.idx for p in self.parents])

    @property
    def encoding(self) -> str:
        """
        Property that stores the encoding type for the individual

        :param value: the new value for the encoding (used on setting up the property)
        :type value: str
        :return: the encoding value
        :rtype: str
        """
        return self._encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    @property
    def fitness(self) -> float:
        """
        Property that stores the fitness value for the individual

        :param value: the new value for the fitness (used on setting up the property)
        :type value: float
        :return: the fitness value
        """
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        self._fitness = value

    @classmethod
    def create_random_individual(
        cls, idx: int, config: GeneticBaseConfig
    ) -> "Individual":
        """
        Class method that creates a new individual with random genes.

        :param idx: the idx of the new individual
        :type idx: int
        :param config: the config object for the individual
        :type config: :class:`mango.models.genetic.config.GeneticBaseConfig`
        :return: the new individual
        """
        ind = cls(idx=idx, config=config)

        ind.create_random_genes()
        return ind

    def create_random_genes(self):
        """
        Method to create random genes for an individual based on the encoding type.

        The genes are stored in the :attr:`genes` attribute
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
        Method to mutate the individual based on the encoding type.

        The mutation probability is used to determine if the individual will be mutated or not.
        An individual can have more than one gene that gets mutated. The

        :param mutation_prob: the probability of mutation
        :type mutation_prob: float
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
        Mutation method for the binary values encoding.

        This mutation is a simple bit flip on the calculated gene position
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = 1 - self.genes[gene_position]

    def _mutate_integer(self):
        """
        Mutation method for the integer values encoding.

        This mutation is a simple random value between the min and max bounds (randint)
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = randint(self.min_bound, self.max_bound)

    def _mutate_real(self):
        """
        Mutation method for the real values encoding

        This mutation is a simple random value between the min and max bounds
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = uniform(self.min_bound, self.max_bound)

    def dominates(self, other):
        """
        This method should implement the logic to check if one individual dominates another one

        The domination concept is mainly used in multiobjective optimization problems.
        A solution dominates another when the first one is equal or better in all objectives and one objective
        is at least better than the other solution
        """
        raise NotImplemented

    def copy(self) -> "Individual":
        """
        Method to create a copy of the individual
        """
        return Individual(
            genes=self.genes.copy(),
            idx=self.idx,
            config=self.config,
            parents=self.parents,
        )

    def __hash__(self):
        """
        Method to calculate the hash of the individual and store it in the :attr:`_hash` attribute

        The hash is calculated based on the byte representation of the genes of the individual so that only individuals
        with the same genome have the hash
        """
        return hash(self.genes.tobytes())

    def __eq__(self, other) -> bool:
        """
        Method to compare two individuals

        In order to make this comparison as fast as possible it used the pre-generated hash to compare the individuals.
        So the comparison is made on the genotype instead of the phenotype, as two individuals with different
        genotypes can have the same phenotype, so they should be considered as not equal
        for crossover and genetic diversity purposes.

        :param other: the other individual to compare
        :return: True if the individuals are equal, False otherwise
        :rtype: bool
        """
        if isinstance(other, self.__class__):
            return self._hash == other._hash

        return False

    def __repr__(self):
        return f"Individual {self.idx}"
