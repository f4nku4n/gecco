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
        best_idv = self.algorithm.do()

        # Check
        _list_val_accs = []
        for tid in range(self.problem.nTasks):
            _val_loader = torch.utils.data.DataLoader(list_val_sets[tid], batch_size=self.val_batch_size, shuffle=False)
            val_results = self.eval_per_task(
                idv.model, val_loader=_val_loader, tid=tid,
                unique_labels=torch.tensor(self.problem.task_classes[tid], dtype=torch.long),
            )
            _list_val_accs.append(float(val_results['acc']))
        matrix_val_accs.append(_list_val_accs)
        CL_metrics = compute_cl_metrics(matrix_val_accs)

