import numpy as np

from tqdm import tqdm

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset

def compute_cl_metrics(accuracies):
    N = len(accuracies)

    AA_per_stage = np.array([np.mean(row) for row in accuracies])
    AA = AA_per_stage[-1]
    AIA = AA_per_stage.mean()

    BWT_per_task = np.array([accuracies[-1][j] - accuracies[j][j] for j in range(N - 1)])
    BWT = BWT_per_task.mean() if N > 1 else np.nan

    forgetting_per_task = np.array([max(accuracies[i][j] for i in range(j, N)) - accuracies[-1][j] for j in range(N - 1)])
    AF = forgetting_per_task.mean() if N > 1 else np.nan

    return {
        "AA": AA,
        "AIA": AIA,
        "BWT": BWT,
        "AF": AF,
        "AA_per_stage": AA_per_stage,
        "BWT_per_task": BWT_per_task,
        "Forgetting_per_task": forgetting_per_task,
    }

class Evaluator:
    """
    - Only for CL-NAS (TaskIL setting)
    - Input: optimizer, list iepochs, train_stream, val_stream
    - Output: list of validation accuracies
    """
    def __init__(self, env_seed, problem, optimizer, lr, list_iepochs, train_batch_size, val_batch_size, val_ratio, device):
        self.seed = env_seed
        self.problem = problem
        self.list_iepochs = list_iepochs

        self.optimizer = optimizer
        self.lr = lr
        self.train_batch_size, self.val_batch_size = train_batch_size, val_batch_size

        self.device = device
        self.nEvals = 0

        self.val_ratio = val_ratio
        self.cache = {}
        if len(list_iepochs) != problem.nTasks or any(e < 0 for e in list_iepochs):
            raise ValueError("list_iepochs must contain one non-negative epoch count per task")
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")

    def task_loaders(self, tid):
        """Use the same deterministic holdout for search, verification and retraining."""
        train_set = self.problem.train_stream[tid].dataset
        if self.problem.val_stream is None:
            n = len(train_set)
            if n < 2:
                raise ValueError(f"Task {tid}: need at least two samples for train/validation")
            indices = np.random.default_rng(self.seed + tid).permutation(n)
            val_size = max(1, min(n - 1, int(n * self.val_ratio)))
            eval_set = train_set.with_transforms("eval") if hasattr(train_set, "with_transforms") else train_set
            val_set = Subset(eval_set, indices[:val_size])
            train_set = Subset(train_set, indices[val_size:])
        else:
            val_set = self.problem.val_stream[tid].dataset
            if hasattr(val_set, "with_transforms"):
                val_set = val_set.with_transforms("eval")
        if not len(train_set) or not len(val_set):
            raise ValueError(f"Task {tid}: train and validation datasets must be nonempty")
        return (
            torch.utils.data.DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True),
            torch.utils.data.DataLoader(val_set, batch_size=self.val_batch_size, shuffle=False),
        )

    def eval_per_task(self, model, val_loader, tid, unique_labels):
        model.eval()
        correct, total = 0, 0

        if unique_labels is None:
            all_labels = torch.cat([batch[1].reshape(-1).cpu() for batch in val_loader])
            unique_labels = torch.unique(all_labels, sorted=True)
        unique_labels = unique_labels.to(self.device)

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch[:2]  # works for (X, y) and (X, y, task)
                X, y = X.to(self.device), y.to(self.device)
                logits = model(X, t=tid)

                if logits.ndim != 2 or logits.size(1) != unique_labels.numel():
                    raise ValueError(f"Task {tid}: model output must match the task class count")
                y_pred = unique_labels[torch.argmax(logits, dim=1)]
                correct += (y_pred == y).sum().item()
                total += y.size(0)

        if total == 0:
            raise ValueError(f"Task {tid}: evaluation dataset is empty")
        acc = correct / total
        results = {'correct': correct, 'total': total, 'acc': acc}
        return results

    def train_per_task(self, model, train_loader, val_loader, maxEpoch, tid):
        model = model.to(self.device)
        model.train()

        if self.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
        else:
            raise ValueError
        criterion = nn.CrossEntropyLoss()

        unique_labels = torch.tensor(self.problem.task_classes[tid], dtype=torch.long)
        unique_labels = unique_labels.to(self.device)

        val_results = self.eval_per_task(model, val_loader=val_loader, tid=tid, unique_labels=unique_labels)
        best_val_acc = val_results['acc']
        best_state_dict = copy.deepcopy(model.state_dict())

        pbar = tqdm(range(maxEpoch), desc=f"Task {tid}", dynamic_ncols=True)

        for _ in pbar:
            model.train()
            total_loss = 0.0
            total_samples = 0

            for batch in train_loader:
                X, y = batch[:2]
                X, y = X.to(self.device), y.to(self.device)
                # Global labels -> Local labels
                y_mapped = torch.searchsorted(unique_labels, y).long()

                logits = model(X, t=tid)
                loss = criterion(logits, y_mapped)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                total_loss += loss.item() * y.size(0)
                total_samples += y.size(0)

            train_loss = total_loss / total_samples
            val_results = self.eval_per_task(model, val_loader=val_loader, tid=tid, unique_labels=unique_labels)
            val_acc = val_results['acc']

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state_dict = copy.deepcopy(model.state_dict())

            pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "best_acc": f"{best_val_acc:.4f}",
            })

        model.load_state_dict(best_state_dict)
        return {
            "best_val_acc": best_val_acc,
        }

    def evaluate(self, idv, metric='AIA'):
        self.nEvals += 1
        if idv.hashX not in self.cache:
            val_loaders = []
            matrix = []
            for cur_tid in range(self.problem.nTasks):
                train_loader, val_loader = self.task_loaders(cur_tid)
                val_loaders.append(val_loader)
                self.train_per_task(
                    idv.model, train_loader, val_loader,
                    maxEpoch=self.list_iepochs[cur_tid], tid=cur_tid,
                )
                row = []
                for tid, loader in enumerate(val_loaders):
                    result = self.eval_per_task(
                        idv.model, loader, tid,
                        torch.tensor(self.problem.task_classes[tid], dtype=torch.long),
                    )
                    row.append(float(result['acc']))
                matrix.append(row)
            entry = compute_cl_metrics(matrix)
            entry['matrix_val_accs'] = copy.deepcopy(matrix)
            self.cache[idv.hashX] = entry
        entry = self.cache[idv.hashX]
        idv.add_data['matrix_val_accs'] = copy.deepcopy(entry['matrix_val_accs'])
        idv.add_data['fitness_metric'] = metric
        idv.F = float(entry[metric])
        print(f'{metric}: {idv.F:.6f}')
        return idv.F

    def verify_search_result(self, idv):
        """Validate historical fitness and remeasure the final model without training."""
        if idv is None:
            raise ValueError("Search returned no architecture; check the evaluation budget")
        matrix = idv.add_data.get('matrix_val_accs')
        metric = idv.add_data.get('fitness_metric')
        if matrix is None or len(matrix) != self.problem.nTasks or any(len(row) != tid + 1 for tid, row in enumerate(matrix)):
            raise ValueError("Search validation history is missing or malformed")
        entry = self.cache.get(idv.hashX)
        if entry is None or metric not in entry:
            raise ValueError("Search fitness cache is missing")
        metrics = compute_cl_metrics(matrix)
        if metric not in metrics or np.ndim(metrics[metric]) != 0:
            raise ValueError("Search fitness metric must be scalar")
        for name in {metric, 'AIA'}:
            if not np.isclose(metrics[name], entry[name], rtol=1e-6, atol=1e-8):
                raise ValueError(f"Search {name} does not match cached validation history")
        if not np.isclose(idv.F, metrics[metric], rtol=1e-6, atol=1e-8):
            raise ValueError("Search F does not match validation history")
        cached_matrix = entry.get('matrix_val_accs', [])
        if len(cached_matrix) != len(matrix) or any(
            len(a) != len(b) or not np.allclose(a, b, rtol=1e-6, atol=1e-8)
            for a, b in zip(matrix, cached_matrix)
        ):
            raise ValueError("Search validation history differs from cache")
        final_accs = []
        # eval() only changes module modes; preserve even mixed train/eval modes.
        modes = [(module, module.training) for module in idv.model.modules()]
        try:
            for tid in range(self.problem.nTasks):
                _, loader = self.task_loaders(tid)
                result = self.eval_per_task(
                    idv.model, loader, tid,
                    torch.tensor(self.problem.task_classes[tid], dtype=torch.long),
                )
                final_accs.append(float(result['acc']))
        finally:
            for module, training in modes:
                module.training = training
        if not np.allclose(final_accs, matrix[-1], rtol=1e-6, atol=1e-8):
            raise ValueError("Search model validation accuracies no longer match the final search stage")
        return {
            'passed': True, 'metric': metric, 'F': float(idv.F),
            'AIA': float(metrics['AIA']), 'AA': float(np.mean(final_accs)),
            'final_val_accs': final_accs,
        }

    def fully_train(self, genotype, list_iepochs, seed):
        """Train fresh weights with validation selection, leaving search state intact."""
        from utils.general_utils import set_seed

        trainer = Evaluator(
            self.seed, self.problem, self.optimizer, self.lr, list_iepochs,
            self.train_batch_size, self.val_batch_size, self.val_ratio, self.device,
        )
        # Keep the holdout seed from search; seed controls initialization/training.
        set_seed(seed)
        model = self.problem.search_space.get_model(genotype, taskcla=self.problem.taskcla).to(self.device)
        for tid in range(self.problem.nTasks):
            train_loader, val_loader = trainer.task_loaders(tid)
            trainer.train_per_task(model, train_loader, val_loader, maxEpoch=list_iepochs[tid], tid=tid)
        return model
