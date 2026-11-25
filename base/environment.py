from utils.general_utils import set_seed
import inspect
import json
from pathlib import Path

import numpy as np
import torch


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def _component_config(component, references=()):
    """Snapshot constructor settings, excluding datasets and runtime state."""
    config = {'class': f'{type(component).__module__}.{type(component).__qualname__}'}
    for name, param in inspect.signature(type(component).__init__).parameters.items():
        if name == 'self' or name in references or param.kind in (
            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        attr = 'seed' if name == 'env_seed' else name
        if hasattr(component, attr):
            config[name] = getattr(component, attr)
    return config


class Environment:
    def __init__(self, seed, device, search_space, problem, algorithm, evaluator,
                 fully_train_iepochs=None, fully_train_seed=None, results_dir='exp_results'):
        self.seed = seed
        self.device = device
        self.search_space = search_space
        self.problem = problem
        self.algorithm = algorithm
        self.evaluator = evaluator
        self.fully_train_iepochs = ([20] * problem.nTasks if fully_train_iepochs is None
                                   else list(fully_train_iepochs))
        self.fully_train_seed = seed if fully_train_seed is None else fully_train_seed
        self.results_dir = Path(results_dir)
        if len(self.fully_train_iepochs) != problem.nTasks or any(e <= 0 for e in self.fully_train_iepochs):
            raise ValueError("fully_train_iepochs must contain one positive epoch count per task")

    def setup(self):
        pass

    def _configs(self):
        configs = {
            'environment': _component_config(self, ('search_space', 'problem', 'algorithm', 'evaluator')),
            'problem': _component_config(self.problem, ('search_space',)),
            'evaluator': _component_config(self.evaluator, ('problem',)),
            'algorithm': _component_config(self.algorithm, ('problem', 'evaluator')),
            'search_space': _component_config(self.search_space),
            'task_classes': self.problem.task_classes,
            'taskcla': self.problem.taskcla,
            'fully_train': {
                'list_iepochs': self.fully_train_iepochs,
                'seed': self.fully_train_seed,
                'validation_split_seed': self.evaluator.seed,
                'initialization': 'fresh',
                'checkpoint_selection': 'best_validation_accuracy_per_task',
                'validation_source': 'holdout' if self.problem.val_stream is None else 'val_stream',
            },
        }
        # Materialize a snapshot before search or training can change any state.
        return json.loads(json.dumps(configs, default=_json_default))

    @staticmethod
    def _write_json(path, data):
        path.write_text(json.dumps(data, default=_json_default, indent=2, ensure_ascii=False), encoding='utf-8')

    def run(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        configs = self._configs()
        set_seed(self.seed)
        best_idv = self.algorithm.do()
        fitness_check = self.evaluator.verify_search_result(best_idv)
        config_path = self.results_dir / f'{best_idv.hashX}_configs.json'
        weights_path = self.results_dir / f'{best_idv.hashX}.pt'
        result_path = self.results_dir / f'{best_idv.hashX}.json'
        self._write_json(config_path, configs)
        print(f"Search {fitness_check['metric']}: {best_idv.F:.6f} (verified)")
        fully_trained_model = self.evaluator.fully_train(
            best_idv.X, self.fully_train_iepochs, self.fully_train_seed,
        )
        torch.save({key: value.detach().cpu() for key, value in fully_trained_model.state_dict().items()}, weights_path)
        test_accuracies = self.problem.evaluate_full_tasks(fully_trained_model)
        test_aa = float(np.mean(list(test_accuracies.values())))
        print(f"Fully trained test AA: {test_aa:.6f}")
        self._write_json(result_path, {
            'hashX': best_idv.hashX,
            'genotype': best_idv.X,
            'phenotype': best_idv.phenotype,
            'search_F': best_idv.F,
            'fitness_check': fitness_check,
            'matrix_val_accs': best_idv.add_data['matrix_val_accs'],
            'test_accuracies': test_accuracies,
            'test_aa': test_aa,
            'weights_file': weights_path.name,
            'configs_file': config_path.name,
        })
        print(f"Saved fully trained weights and results to {self.results_dir.resolve()}")
        return {
            'best_idv': best_idv,
            'fitness_check': fitness_check,
            'fully_trained_model': fully_trained_model,
            'test_accuracies': test_accuracies,
            'test_aa': test_aa,
            'weights_path': str(weights_path.resolve()),
            'results_path': str(result_path.resolve()),
            'configs_path': str(config_path.resolve()),
        }
