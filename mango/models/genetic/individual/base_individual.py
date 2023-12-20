import hashlib
from random import uniform, randint
import numpy as np


class Individual:
    def __init__(self, genes=None, idx: int = 0, parents: tuple = None, config=None):
        self.fitness = None
        self.idx = idx
        self.config = config
        self.parents = parents

        # Info to get from config file
        self.encoding = self.config("encoding")
        self.min_bound = self.config("gene_min_value")
        self.max_bound = self.config("gene_max_value")
        self.gene_length = self.config("gene_length")

        self.genes = genes

        self.parents_idx = None
        if self.parents is not None:
            self.parents_idx = tuple([p.idx for p in self.parents])

    @classmethod
    def create_random_individual(cls, idx, config):
        ind = cls(idx=idx, config=config)

        ind.create_random_genes()
        return ind

    def create_random_genes(self):
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

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, value: np.ndarray = None):
        if value is not None:
            value[np.greater(value, self.max_bound)] = self.max_bound
            value[np.less(value, self.min_bound)] = self.min_bound
        self._genes = value
        if self._genes is not None:
            self._hash = self.__hash__()

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        # might not be needed
        self._idx = value

    def mutate(self, mutation_prob: float = None):
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

    def _mutate_real(self):
        """
        Mutation method for the real values encoding
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = uniform(self.min_bound, self.max_bound)

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
        """
        gene_position = randint(0, self.gene_length - 1)
        self.genes[gene_position] = randint(self.min_bound, self.max_bound)

    def dominates(self, other):
        """
        This method should implement the logic to check if one individual dominates another one

        The domination concept is mainly used in multiobjective optimization problems.
        A solution dominates another when the first one is equal or better in all objectives and one objective
        is at least better than the other solution
        """
        raise NotImplemented

    def copy(self):
        return Individual(
            genes=self.genes.copy(),
            idx=self.idx,
            config=self.config,
            parents=self.parents,
        )

    def __hash__(self):
        return hash(self.genes.tobytes())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._hash == other._hash

            # for i, j in zip(self.genes.flat, other.genes.flat):
            #     if i != j:
            #         return False
            # return True

            # return np.array_equal(self.genes, other.genes)
        return False

    def __repr__(self):
        return f"Individual {self.idx}"
