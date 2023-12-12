from random import uniform, randint


class Individual:
    def __init__(self, genes=None, idx: int = 0, parents: tuple = None, config=None):
        self.fitness = None
        self.idx = idx

        # Info to get from config file
        self.encoding = config("encoding")
        self.min_bound = config("gene_min_value")
        self.max_bound = config("gene_max_value")
        self.gene_length = config("gene_length")

        self.genes = genes

        self.parents_idx = None
        if parents is not None:
            self.parents_idx = tuple([p.idx for p in parents])

        self._hash = self.__hash__()

    @classmethod
    def create_random_individual(cls, idx, config):
        ind = cls(idx=idx, config=config)

        ind.create_random_genes()
        return ind

    def create_random_genes(self):
        if self.encoding == "real":
            self.genes = [
                uniform(self.min_bound, self.max_bound) for _ in range(self.gene_length)
            ]
            self._hash = self.__hash__()
        elif self.encoding == "binary":
            self.genes = [randint(0, 1) for _ in range(self.gene_length)]

        elif self.encoding == "integer":
            self.genes = [
                randint(self.min_bound, self.max_bound) for _ in range(self.gene_length)
            ]

        self._hash = self.__hash__()

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, value: list = None):
        if value is not None:
            value = [self.min_bound if x < self.min_bound else x for x in value]
            value = [self.max_bound if x > self.max_bound else x for x in value]
        self._genes = value

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

    def __hash__(self):
        if self.genes is None:
            return None
        return hash(tuple(self.genes))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._hash == other._hash
        return False

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self._hash < other._hash
        raise NotImplemented("The objects do not share the same class")

    def __le__(self, other):
        if isinstance(other, self.__class__):
            return self._hash <= other._hash
        raise NotImplemented("The objects do not share the same class")

    def _eq_genotype(self, other):
        if isinstance(other, self.__class__):
            return self.genes == other.genes
        raise NotImplemented("The objects do not share the same class")

    def __repr__(self):
        return f"Individual {self.idx}"