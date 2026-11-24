from abc import ABC, abstractmethod

class SearchSpace(ABC):
    def __init__(self, name):
        self.name = name

    def sample(self, genotype=True):
        if genotype:
            return self.encode(self._sample())
        return self._sample()

    def is_valid(self, network):
        return True

    @abstractmethod
    def return_available_ops(self, idx):
        pass

    @abstractmethod
    def _sample(self):
        pass

    @abstractmethod
    def encode(self, network):
        pass

    @abstractmethod
    def decode(self, network):
        pass