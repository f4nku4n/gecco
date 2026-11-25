from utils.general_utils import set_seed


class Environment:
    def __init__(self, seed, device, search_space, problem, algorithm, evaluator):
        self.seed = seed
        self.device = device

        self.search_space = search_space
        self.problem = problem
        self.algorithm = algorithm
        self.evaluator = evaluator

    def setup(self):
        pass

    def run(self):
        set_seed(self.seed)
        self.algorithm.do()