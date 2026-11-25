from genetic import Individual

class RandomSearch:
    def __init__(self, problem, evaluator):
        self.problem = problem  # Search space
        self.evaluator = evaluator  # Fitness evaluator

    def do(self):
        list_hashX = []
        best_idv = None
        while self.evaluator.nEvals < self.problem.maxEvals:
            idv = Individual()
            idv.X = self.problem.search_space.sample()
            idv.hashX = ''.join([str(x) for x in idv.X])
            if idv.hashX in list_hashX:
                continue
            list_hashX.append(idv.hashX)
            idv.phenotype = self.problem.search_space.decode(idv.X)
            idv.model = self.problem.search_space.get_model(idv.X, taskcla=self.problem.taskcla)
            self.evaluator.evaluate(idv, 'AIA')

            if best_idv is None or idv.F > best_idv.F:
                best_idv = idv
        return best_idv