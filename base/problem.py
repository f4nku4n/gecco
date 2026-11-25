import torch
from utils.general_utils import create_TIL_benchmark


class Problem:
    """
    TIL supported only
    """
    def __init__(self, problem_name, search_space, nTasks, maxEvals, data_root, device):
        self.problem_name = problem_name
        self.search_space = search_space
        self.nTasks = nTasks
        self.train_stream, self.val_stream, self.test_stream = None, None, None
        self.data_root = data_root
        self.maxEvals = maxEvals
        self.device = device

    def setup(self):
        benchmark, total_classes = create_TIL_benchmark(benchmark=self.problem_name, resize=False,
                                                        n_experiences=self.nTasks, datadir=self.data_root)
        # task_classes = [sorted(experience.classes_in_this_experience) for experience in benchmark.train_stream]
        # taskcla = [(tid, len(classes)) for tid, classes in enumerate(task_classes)]
        # label_maps = [{label: local_id for local_id, label in enumerate(classes)} for classes in task_classes]

        self.train_stream = benchmark.train_stream
        try:
            self.val_stream = benchmark.val_stream
        except AttributeError:
            self.val_stream = None
        self.test_stream = benchmark.test_stream

    def eval_per_task(self, model, dataloader, tid=None, unique_labels=None, **kwargs):
        model.eval()
        correct, total = 0, 0

        if unique_labels is None:
            all_labels = torch.cat([batch[1].reshape(-1).cpu() for batch in dataloader])
            unique_labels = torch.unique(all_labels, sorted=True)
        unique_labels = unique_labels.to(self.device)

        with torch.no_grad():
            for batch in dataloader:
                X, y = batch[:2]  # works for (X, y) and (X, y, task)
                X, y = X.to(self.device), y.to(self.device)

                logits = model(X, t=tid)
                y_pred = unique_labels[torch.argmax(logits, dim=1)]
                correct += (y_pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        results = {'correct': correct, 'total': total, 'acc': acc}
        return results

    def evaluate_full_tasks(self, model):
        res = {}
        for tid in range(self.nTasks):
            test_set = self.test_stream[tid].dataset
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

            test_results = self.eval_per_task(model=model, dataloader=test_loader, device=self.device, tid=tid)
            test_accuracy = round(test_results['acc'], 4)
            print(f'Test accuracy (task {tid + 1}): {test_accuracy * 100:.2f}%')
            res[f'Task-{tid + 1}'] = test_accuracy
        print('-' * 20)
        return res

