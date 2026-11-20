import torch
from utils.torch_utils import eval_process

def evaluate_all_tasks(model, cur_task_id, test_stream, batch_size=64, device='cuda', tid=None, task_labels=None, **kwargs):
    res = {}
    for tid in range(cur_task_id):
        test_set = test_stream[tid].dataset
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

        eval_kwargs = kwargs.copy()
        if task_labels is not None:
            eval_kwargs['unique_labels'] = task_labels[tid]
        test_results = eval_process(model=model, dataloader=test_loader, device=device, tid=tid, **eval_kwargs)
        test_accuracy = round(test_results['acc'], 4)
        print(f'Test accuracy (task {tid + 1}): {test_accuracy * 100:.2f}%')
        res[f'Task-{tid + 1}'] = test_accuracy
    print('-' * 20)
    return res
