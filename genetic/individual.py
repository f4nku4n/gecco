import copy

class Individual:
    """
    - 'Phenotype' is the encoded skeleton, X is genotype.
    - 'Network' is built upon the phenotype
    - F is fitness score (e.g., accuracy)
    - 'Info' consists of training epochs and training time (in seconds).
    """
    def __init__(self):
        self.X = None
        self.hashX = None
        self.F = None
        self.phenotype = None
        self.model = None
        self.add_data = {}

    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.add_data[key] = value

    def copy(self):
        idv = copy.copy(self)
        idv.add_data = self.add_data.copy()
        return idv

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self.add_data:
            return self.add_data[key]
        return None

    def __call__(self):
        print(f'Genotype: {self.X} - Score: {self.F}')

if __name__ == '__main__':
    # x = Individual()
    # x()
    pass