import os
import json
import argparse
import pickle as p
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import Subset

from utils.torch_utils import compute_zc_score
from zc_scores import AVAILABLE_SCORES
from load_models import get_model

from utils.torch_utils import train_process
from utils.TIL_eval import evaluate_all_tasks
from utils.general_utils import set_seed, create_TIL_benchmark, NumpyEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_score', '--search_metric', choices=('acc', *AVAILABLE_SCORES), default='acc',
                        help='acc: train and evaluate; otherwise compute the selected zero-cost proxy')
    return parser.parse_args()


def run():
    args = parse_args()

    configs = {}
    INIT_SEED = 42

    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE = 32, 128
    INIT_LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    BENCHMARK = 'SplitCIFAR10'
    N_EXPERIENCES = 5

    TRAIN_EPOCH = [20] * N_EXPERIENCES
    # TRAIN_EPOCH[0] = 4

    DATA_DIR = '/Data/quanphm/datasets'

    SEED = INIT_SEED + 0
    set_seed(SEED)

    configs['SEED'] = SEED
    configs['BENCHMARK'] = BENCHMARK
    configs['SETTING'] = 'TIL'
    configs['N_EXPERIENCES'] = N_EXPERIENCES
    configs['TRAIN_EPOCH'] = TRAIN_EPOCH
    configs['TRAIN_BATCH_SIZE'] = TRAIN_BATCH_SIZE
    configs['EVAL_BATCH_SIZE'] = EVAL_BATCH_SIZE
    configs['INIT_LR'] = INIT_LR
    configs['OPTIMIZER'] = 'Adam'
    configs['SEARCH_SCORE'] = args.search_score

    result_dir = os.path.join('exp_results', f'til_splitc10_{N_EXPERIENCES}')
    if args.search_score != 'acc':
        result_dir = f'{result_dir}/{args.search_score}'
    os.makedirs(f'{result_dir}/results', exist_ok=True)
    if args.search_score == 'acc':
        os.makedirs(f'{result_dir}/weights', exist_ok=True)
    with open(f'{result_dir}/configs.json', 'w') as fp:
        json.dump(configs, fp, indent=4)

    benchmark, total_classes = create_TIL_benchmark(benchmark=BENCHMARK, resize=False, n_experiences=N_EXPERIENCES,
                                                    datadir=DATA_DIR)
    task_classes = [sorted(experience.classes_in_this_experience) for experience in benchmark.train_stream]
    taskcla = [(tid, len(classes)) for tid, classes in enumerate(task_classes)]
    label_maps = [{label: local_id for local_id, label in enumerate(classes)} for classes in task_classes]

    with open('list_unique_genotypes.p', 'rb') as fp:
        list_genotypes = p.load(fp)
    for genotype in list_genotypes:
        str_genotype = ''.join([str(g) for g in genotype])
        if os.path.isfile(f'{result_dir}/results/{str_genotype}.json'):
            continue
        model = get_model(genotype=genotype, num_classes=None, taskcla=taskcla)
        model = model.to(DEVICE)
        total_params = sum(param.numel() for param in model.parameters())
        print(f"Number of parameters: {total_params}")

        train_stream = benchmark.train_stream
        try:
            val_stream = benchmark.val_stream
        except AttributeError:
            val_stream = None
        test_stream = benchmark.test_stream if args.search_score == 'acc' else None
        val_ratio = 0.1
        info = {'experiment': {}}
        for cur_tid in range(N_EXPERIENCES):
            info['experiment'][f'Task-{cur_tid + 1}'] = {}
            start = datetime.now()
            start_time = start.strftime("%H:%M:%S %d/%m/%Y")
            info['experiment'][f'Task-{cur_tid + 1}']['start_time'] = start_time

            train_set = train_stream[cur_tid].dataset
            val_set = None
            if args.search_score == 'acc' and val_stream is None:
                rng = np.random.default_rng(SEED + cur_tid)
                n = len(train_set)
                indices = rng.permutation(n)

                val_ratio = val_ratio
                val_size = int(n * val_ratio)

                val_indices = indices[:val_size]
                val_set = Subset(train_set.with_transforms("eval"), val_indices)
            elif args.search_score == 'acc':
                val_set = val_stream[cur_tid].dataset

            val_loader = None
            if args.search_score == 'acc':
                val_loader = torch.utils.data.DataLoader(val_set, batch_size=EVAL_BATCH_SIZE, shuffle=False)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

            if args.search_score == 'acc':
                print(f'Task-{cur_tid + 1}: Training')
                train_results = train_process(
                    model, train_loader=train_loader, val_loader=val_loader, max_epoch=TRAIN_EPOCH[cur_tid],
                    lr=INIT_LR, device=DEVICE, tid=cur_tid
                )
                score = float(train_results['best_val_acc'])
                print(f'Task-{cur_tid + 1}: Best validation accuracy: {score * 100:.2f}%')
            else:
                score = compute_zc_score(
                    model, train_loader, search_score=args.search_score,
                    tid=cur_tid, device=DEVICE, seed=SEED, label_map=label_maps[cur_tid],
                )
                print(f'Task-{cur_tid + 1}: {args.search_score}: {score}')
            info['experiment'][f'Task-{cur_tid + 1}']['search_score'] = args.search_score
            info['experiment'][f'Task-{cur_tid + 1}']['score'] = score
            end = datetime.now()
            end_time = end.strftime("%H:%M:%S %d/%m/%Y")
            info['experiment'][f'Task-{cur_tid + 1}']['end_time'] = end_time
            duration = (end - start).total_seconds()
            info['experiment'][f'Task-{cur_tid + 1}']['duration_in_sec'] = duration

            total_params = sum(param.numel() for param in model.parameters())
            info['experiment'][f'Task-{cur_tid + 1}']['total_params'] = total_params

            #     PATH = f'{result_dir}/{full_method_name}/model_tid-{cur_tid+1}_SEED-{SEED}.pt'
            #     ckpt = {"model_state": model.state_dict()}
            #     torch.save(ckpt, PATH)

            if args.search_score == 'acc':
                print('Computing accuracy on the whole test set')
                res = evaluate_all_tasks(model=model, cur_task_id=cur_tid + 1, test_stream=test_stream,
                                         batch_size=EVAL_BATCH_SIZE, device=DEVICE, total_classes=total_classes,
                                         scenario='TIL')

                info['experiment'][f'Task-{cur_tid + 1}']['list_accs'] = res

        if args.search_score == 'acc':
            ckpt = {"model_state": model.state_dict()}
            PATH = f'{result_dir}/weights/{str_genotype}.pt'
            torch.save(ckpt, PATH)

        with open(f'{result_dir}/results/{str_genotype}.json', 'w') as fp:
            json.dump(info, fp, indent=4, cls=NumpyEncoder)


if __name__ == "__main__":
    run()
