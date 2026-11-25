from algorithm import RandomSearch
from base import Environment, Evaluator, Problem
from search_spaces import SS_201

if __name__ == '__main__':
    device = 'cuda'
    search_space = SS_201()
    seed = 42
    problem_config = {
        'problem_name': 'SplitCIFAR10',
        'search_space': search_space,
        'nTasks': 5,
        'data_root': '/Data/quanphm/datasets',
        'maxEvals': 10,
        'device': device
    }
    problem = Problem(**problem_config)

    evaluator_config = {
        'env_seed': seed,
        'problem': problem,
        'optimizer': 'Adam',
        'lr': 1e-3,
        'list_iepochs': [5] * problem_config['nTasks'],
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
        'evaluator': evaluator
    }

    env = Environment(**env_config)
    env.setup()
    env.run()
