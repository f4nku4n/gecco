import numpy as np

from tqdm import tqdm
from datetime import datetime

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

                y_pred = unique_labels[torch.argmax(logits, dim=1)]
                correct += (y_pred == y).sum().item()
                total += y.size(0)

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
            train_stream = self.problem.train_stream
            val_stream = self.problem.val_stream

            # info = {'experiment': {}}
            list_val_sets = []
            matrix_val_accs = []
            for cur_tid in range(self.problem.nTasks):
                # info['experiment'][f'Task-{cur_tid + 1}'] = {}

                train_set = train_stream[cur_tid].dataset
                if val_stream is None:
                    rng = np.random.default_rng(self.seed + cur_tid)
                    n = len(train_set)
                    indices = rng.permutation(n)

                    val_size = max(1, min(n - 1, int(n * self.val_ratio)))

                    val_indices = indices[:val_size]
                    eval_set = train_set.with_transforms("eval") if hasattr(train_set, "with_transforms") else train_set
                    val_set = Subset(eval_set, val_indices)
                    train_set = Subset(train_set, indices[val_size:])
                else:
                    val_set = val_stream[cur_tid].dataset

                list_val_sets.append(val_set)

                train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.val_batch_size, shuffle=False)

                maxEpoch = self.list_iepochs[cur_tid]

                train_results = self.train_per_task(
                    idv.model, train_loader=train_loader, val_loader=val_loader, maxEpoch=maxEpoch, tid=cur_tid
                )

                _list_val_accs = []
                for tid in range(cur_tid + 1):
                    _val_loader = torch.utils.data.DataLoader(list_val_sets[tid], batch_size=self.val_batch_size, shuffle=False)
                    val_results = self.eval_per_task(
                        idv.model, val_loader=_val_loader, tid=tid,
                        unique_labels=torch.tensor(self.problem.task_classes[tid], dtype=torch.long),
                    )
                    _list_val_accs.append(float(val_results['acc']))
                matrix_val_accs.append(_list_val_accs)
                CL_metrics = compute_cl_metrics(matrix_val_accs)
            self.cache[idv.hashX] = CL_metrics
                # score = float(train_results['best_val_acc'])
                # print(f'Task-{cur_tid + 1}: Best validation accuracy: {score * 100:.2f}%')

                # info['experiment'][f'Task-{cur_tid + 1}']['score'] = score
                # total_params = sum(param.numel() for param in idv.model.parameters())
                # info['experiment'][f'Task-{cur_tid + 1}']['total_params'] = total_params
        print('AIA:', self.cache[idv.hashX]['AIA'])
        idv.F = self.cache[idv.hashX][metric]
        return idv.F
