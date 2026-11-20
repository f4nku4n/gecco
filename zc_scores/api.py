"""DataLoader-based interface, including TinyNetwork task-head support."""
import copy
import math
import random
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .measures import AVAILABLE, implementation


def _identity(self, inputs):
    return inputs


def compute_zc_scores(model, dataloader, metrics=('synflow', 'snip', 'nwot'),
                      *, task_id=None, device=None, num_batches=1, seed=42,
                      label_map=None, normalize=True):
    """Return Python float scores without changing the supplied model.

    Batches may be (images, labels) or (images, labels, task_ids).
    For TinyNetwork with multiple heads, task_id selects a head by index.
    label_map maps dataset labels to local head labels, e.g. {4: 0, 7: 1}.
    normalize applies MF-NAS's signed log to synflow and maps nonfinite
    values to -1e8. Set False to inspect raw values (including NaN/Inf).
    Every metric starts from a fresh copy of the supplied weights.
    """
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    unknown = set(names).difference(AVAILABLE)
    if unknown:
        raise ValueError(f'Unknown scores: {sorted(unknown)}; available: {AVAILABLE}')
    if not names:
        return {}
    if not isinstance(num_batches, int) or num_batches < 1:
        raise ValueError('num_batches must be a positive integer')
    multihead = isinstance(getattr(model, 'last', None), nn.ModuleList)
    if multihead and (task_id is None or not 0 <= task_id < len(model.last)):
        raise ValueError('task_id must select a valid TinyNetwork head')
    if task_id is not None and not multihead:
        raise ValueError('task_id is supported for TinyNetwork with ModuleList last')
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    py_state, np_state = random.getstate(), np.random.get_state()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad():
            torch.random.default_generator.manual_seed(seed)
            if cuda_devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            batches = []
            iterator = iter(dataloader)
            for _ in range(num_batches):
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError('DataLoader has fewer than num_batches batches') from exc
                batches.append((batch[0], batch[1]))
            inputs = torch.cat([b[0] for b in batches]).to(device)
            targets = torch.cat([b[1] for b in batches]).to(device).long()
            if inputs.shape[0] == 0:
                raise ValueError('Cannot score an empty batch')
            if label_map is not None:
                targets = torch.tensor([label_map[int(y)] for y in targets], device=device)
            if multihead and (targets.min() < 0 or targets.max() >= model.last[task_id].out_features):
                raise ValueError('Labels must be local to the selected head; supply label_map')
            if 'zico' in names and inputs.shape[0] < 2:
                raise ValueError('zico needs at least two samples')
            scores = {}
            for name in names:
                net = copy.deepcopy(model).to(device).train()
                if multihead:
                    # Reuse TinyNetwork's single-head path and exclude unused heads
                    # from parameter-only proxies such as l2_norm.
                    net.classifier = net.last[task_id]
                    del net.last
                func, bn, kwargs = implementation(name)
                if not bn:
                    for layer in net.modules():
                        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
                            layer.forward = types.MethodType(_identity, layer)
                try:
                    value = func(net, inputs.detach().clone(), targets.clone(),
                                 loss_fn=F.cross_entropy, split_data=1, **kwargs)
                    if isinstance(value, (list, tuple)):
                        value = sum(float(v.detach().sum()) for v in value)
                    value = float(value)
                    if normalize:
                        if not math.isfinite(value):
                            value = -1e8
                        elif name == 'synflow' and value != 0:
                            value = math.log(value) if value > 0 else -math.log(-value)
                    scores[name] = value
                finally:
                    del net
            return scores
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
