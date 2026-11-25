from utils.general_utils import set_seed
import numpy as np


class Environment:
    def __init__(self, seed, device, search_space, problem, algorithm, evaluator,
                 fully_train_iepochs=None, fully_train_seed=None):
        self.seed = seed
        self.device = device
        self.search_space = search_space
        self.problem = problem
        self.algorithm = algorithm
        self.evaluator = evaluator
        self.fully_train_iepochs = ([20] * problem.nTasks if fully_train_iepochs is None
                                   else list(fully_train_iepochs))
        self.fully_train_seed = seed if fully_train_seed is None else fully_train_seed
        if len(self.fully_train_iepochs) != problem.nTasks or any(e <= 0 for e in self.fully_train_iepochs):
            raise ValueError("fully_train_iepochs must contain one positive epoch count per task")

    def setup(self):
        pass

    def run(self):
        set_seed(self.seed)
        best_idv = self.algorithm.do()
        fitness_check = self.evaluator.verify_search_result(best_idv)
        print(f"Search {fitness_check['metric']}: {best_idv.F:.6f} (verified)")
        fully_trained_model = self.evaluator.fully_train(
            best_idv.X, self.fully_train_iepochs, self.fully_train_seed,
        )
        test_accuracies = self.problem.evaluate_full_tasks(fully_trained_model)
        test_aa = float(np.mean(list(test_accuracies.values())))
        print(f"Fully trained test AA: {test_aa:.6f}")
        return {
            'best_idv': best_idv,
            'fitness_check': fitness_check,
            'fully_trained_model': fully_trained_model,
            'test_accuracies': test_accuracies,
            'test_aa': test_aa,
        }
