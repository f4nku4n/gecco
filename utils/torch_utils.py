"""Task-specific model scoring helpers."""
from zc_scores import compute_zc_scores

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import copy

def train_process(model, train_loader, val_loader, max_epoch, lr=1e-3, device='cuda', tid=None, unique_labels=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    if unique_labels is None:
        all_labels = torch.cat([batch[1].reshape(-1).cpu() for batch in train_loader])
        unique_labels = torch.unique(all_labels, sorted=True)
    unique_labels = unique_labels.to(device)
    val_results = eval_process(model, val_loader, device=device, tid=tid, unique_labels=unique_labels)
    best_val_acc = val_results['acc']
    best_state_dict = copy.deepcopy(model.state_dict())

    pbar = tqdm(range(max_epoch), desc=f"Task {tid}", dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            X, y, _ = batch
            X, y = X.to(device), y.to(device)
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
        val_results = eval_process(model, val_loader, device=device, tid=tid, unique_labels=unique_labels)
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

def eval_process(model, dataloader, device='cuda', tid=None, unique_labels=None, **kwargs):
    model.eval()
    correct, total = 0, 0

    if unique_labels is None:
        all_labels = torch.cat([batch[1].reshape(-1).cpu() for batch in dataloader])
        unique_labels = torch.unique(all_labels, sorted=True)
    unique_labels = unique_labels.to(device)

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:2]  # works for (X, y) and (X, y, task)
            X, y = X.to(device), y.to(device)

            logits = model(X, t=tid)
            # print(classes_of_known_task)

            # if invalid_classes is not None:
            #     logits[:, invalid_classes] = -float("inf")

            y_pred = unique_labels[torch.argmax(logits, dim=1)]
            correct += (y_pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    results = {'correct': correct, 'total': total, 'acc': acc}
    return results

def compute_zc_score(model, train_loader, search_score, tid, device=None,
                     *, num_batches=1, seed=42, label_map=None):
    """Compute one zero-cost score for the selected task head.

    Metrics that use class labels require labels local to the selected head
    or a label_map.
    The input model is left unchanged. Returns a Python float.
    """
    return compute_zc_scores(
        model, train_loader, metrics=search_score, task_id=tid,
        device=device, num_batches=num_batches, seed=seed, label_map=label_map,
    )[search_score]
