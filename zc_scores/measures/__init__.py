"""Lazy registry for extracted MF-NAS implementations.

Copying and BatchNorm handling are owned by zc_scores.compute_zc_scores.
"""
from importlib import import_module

_impls = {}
AVAILABLE = (
    'grad_norm', 'snip', 'grasp', 'fisher', 'jacov', 'plain',
    'synflow', 'synflow_bn', 'epe_nas', 'zen', 'l2_norm', 'nwot',
    'zico', 'ntk', 'meco', 'meco_opt', 'swap',
)


def measure(name, bn=True, copy_net=True, force_clean=True, **impl_args):
    def register(func):
        _impls[name] = (func, bn, impl_args)
        return func
    return register


def implementation(name):
    module = {'synflow_bn': 'synflow', 'meco_opt': 'meco'}.get(name, name)
    import_module(f'{__name__}.{module}')
    return _impls[name]
