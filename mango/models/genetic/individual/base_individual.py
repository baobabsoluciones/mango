from abc import ABC, abstractmethod


class BaseIndividual(ABC):
    def __init__(self, genes=None, idx: int = 0, parents: tuple = None):
        self._genes = genes
        self.fitness = None
        self._idx = idx
        self._hash = self.__hash__()
        self.gene_len = len(self.genes)

        self.parents_idx = None
        if parents is not None:
            p1, p2 = parents
            self.parents_idx = (p1.idx, p2.idx)

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, value: list = None):
        self._genes = value

    @genes.deleter
    def genes(self):
        del self._genes

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        # might not be needed
        self._idx = value

    @idx.deleter
    def idx(self):
        del self._idx

    @abstractmethod
    def mutate(self):
        """This method should implement the logic to mutate the genotype of one individual"""
        # Implemented on inherited classes
        raise NotImplemented

    @abstractmethod
    def dominates(self, other):
        """This method should implement the logic to check if one individual dominates another one"""
        # Implemented on inherited classes
        raise NotImplemented

    def __hash__(self):
        return hash(self.idx)

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
