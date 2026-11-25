from algorithm import RandomSearch
from base import Environment, Evaluator, Problem
from search_spaces import SS_201
from utils.general_utils import set_seed

if __name__ == '__main__':
    device = 'cuda'
    search_space = SS_201()
    seed = 42
    set_seed(seed)
    problem_config = {
        'problem_name': 'SplitCIFAR10',
        'search_space': search_space,
        'nTasks': 5,
        'data_root': '/Data/quanphm/datasets',
        'maxEvals': 5,
        'device': device
    }
    problem = Problem(**problem_config)

    evaluator_config = {
        'env_seed': seed,
        'problem': problem,
        'optimizer': 'Adam',
        'lr': 1e-3,
        'list_iepochs': [3] * problem_config['nTasks'],
        'train_batch_size': 32,
        'val_batch_size': 128,
        'val_ratio': 0.1,
        'device': device
    }
    evaluator = Evaluator(**evaluator_config)

    algorithm = RandomSearch(problem=problem, evaluator=evaluator)

    env_config = {
        'seed': 42,
        'device': device,
        'search_space': search_space,
        'problem': problem,
        'algorithm': algorithm,
        'evaluator': evaluator,
        'fully_train_iepochs': [20] * problem.nTasks,
        'fully_train_seed': seed,
        'results_dir': 'exp_results',
    }

    env = Environment(**env_config)
    env.setup()
    results = env.run()
