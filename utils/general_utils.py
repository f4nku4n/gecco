import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset

import torchvision.transforms as transforms
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.datasets import MNIST, CIFAR10
from avalanche.benchmarks.scenarios.supervised import class_incremental_benchmark
from avalanche.benchmarks.scenarios.task_aware import task_incremental_benchmark
from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location
from avalanche.benchmarks.utils import as_classification_dataset
from avalanche.benchmarks.utils import make_avalanche_dataset, DataAttribute, TaskLabels

from avalanche.benchmarks.classic import RotatedMNIST

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class LabelOffsetDataset(Dataset):
    def __init__(self, dataset, offset):
        self.dataset = dataset
        self.offset = offset

        labels = np.asarray(dataset.labels).reshape(-1)
        self.targets = [int(y) + offset for y in labels]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        y = self.targets[idx]
        return x, y

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_CIL_benchmark(benchmark, pretrained=False, resize=False, n_experiences=5, datadir=None):
    if 'MNIST' in benchmark:
        if datadir is None:
            datadir = default_dataset_location('mnist')
        initial_transforms = [transforms.ToTensor()]
        if resize:
            initial_transforms.insert(0, transforms.Resize((32, 32)))
        list_transforms = transforms.Compose(initial_transforms)

        train_set = as_classification_dataset(
            MNIST(datadir, train=True, download=True),
            transform_groups={
                'train': list_transforms,
                'eval': list_transforms
            })

        test_set = as_classification_dataset(
            MNIST(datadir, train=False, download=True),
            transform_groups={
                'train': list_transforms,
                'eval': list_transforms
            })
        total_classes = 10
    elif 'CIFAR10' in benchmark:
        if datadir is None:
            datadir = default_dataset_location('cifar10')
        if pretrained:
            train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        else:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010)
                )
            ])
        train_set = as_classification_dataset(
            CIFAR10(datadir, train=True, download=True),
            transform_groups={
                'train': train_transform,
                'eval': test_transform
            })

        test_set = as_classification_dataset(
            CIFAR10(datadir, train=False, download=True),
            transform_groups={
                'train': test_transform,
                'eval': test_transform
            })
        total_classes = 10
    else:
        raise ValueError()
    benchmark = class_incremental_benchmark({'train': train_set, 'test': test_set}, num_experiences=n_experiences)
    return benchmark, total_classes

def create_MedMNIST_benchmark(n_experiences=3, dataset_root=None):
    benchmark_name = f'MedMNIST_{n_experiences}'
    if benchmark_name == 'MedMNIST_3':
        from medmnist import DermaMNIST, PneumoniaMNIST, BreastMNIST
        rgb_transform = transforms.ToTensor()

        gray_to_rgb_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])

        tasks = [
            (DermaMNIST, 7, rgb_transform),
            (PneumoniaMNIST, 2, gray_to_rgb_transform),
            (BreastMNIST, 2, gray_to_rgb_transform),
        ]
        random.shuffle(tasks)
        offset = 0
        train_sets, val_sets, test_sets = [], [], []

        for task_id, (DatasetCls, n_classes, transform) in enumerate(tasks):
            train_raw = DatasetCls(root=dataset_root, split="train", transform=transform, download=True)
            val_raw = DatasetCls(root=dataset_root, split="val", transform=transform, download=True)
            test_raw = DatasetCls(root=dataset_root, split="test", transform=transform, download=True)

            train_set = LabelOffsetDataset(train_raw, offset)
            val_set = LabelOffsetDataset(val_raw, offset)
            test_set = LabelOffsetDataset(test_raw, offset)

            train_avl = make_avalanche_dataset(
                train_set,
                data_attributes=[DataAttribute(train_set.targets, "targets")],
            )

            val_avl = make_avalanche_dataset(
                val_set,
                data_attributes=[DataAttribute(val_set.targets, "targets")],
            )

            test_avl = make_avalanche_dataset(
                test_set,
                data_attributes=[DataAttribute(test_set.targets, "targets")],
            )

            train_sets.append(train_avl)
            val_sets.append(val_avl)
            test_sets.append(test_avl)

            offset += n_classes
        total_classes = offset
    else:
        raise ValueError()
    return dataset_benchmark(train_datasets=train_sets, test_datasets=test_sets,
        other_streams_datasets={"val": val_sets}), total_classes

def create_TIL_benchmark(benchmark, pretrained=False, resize=False, n_experiences=5, datadir=None):
    if benchmark == 'MedMNIST':
        benchmark_CIL, total_classes = create_MedMNIST_benchmark(n_experiences, datadir)
    else:
        benchmark_CIL, total_classes = create_CIL_benchmark(benchmark, pretrained, resize, n_experiences, datadir)

    add_attrs_train, add_attrs_test = {}, {}
    for i in range(len(benchmark_CIL.train_stream)):
        add_attrs_train[i] = {}
        add_attrs_train[i]['future_classes'] = benchmark_CIL.train_stream[i].future_classes
        add_attrs_train[i]['classes_in_this_experience'] = benchmark_CIL.train_stream[i].classes_in_this_experience
        add_attrs_train[i]['previous_classes'] = benchmark_CIL.train_stream[i].previous_classes
        add_attrs_train[i]['classes_seen_so_far'] = benchmark_CIL.train_stream[i].classes_seen_so_far

        add_attrs_test[i] = {}
        add_attrs_test[i]['future_classes'] = benchmark_CIL.test_stream[i].future_classes
        add_attrs_test[i]['classes_in_this_experience'] = benchmark_CIL.test_stream[i].classes_in_this_experience
        add_attrs_test[i]['previous_classes'] = benchmark_CIL.test_stream[i].previous_classes
        add_attrs_test[i]['classes_seen_so_far'] = benchmark_CIL.test_stream[i].classes_seen_so_far

    benchmark_TIL = task_incremental_benchmark(benchmark_CIL, reset_task_labels=True)

    for i in range(len(benchmark_CIL.train_stream)):
        benchmark_TIL.train_stream[i].future_classes = add_attrs_train[i]['future_classes']
        benchmark_TIL.train_stream[i].classes_in_this_experience = add_attrs_train[i]['classes_in_this_experience']
        benchmark_TIL.train_stream[i].previous_classes = add_attrs_train[i]['previous_classes']
        benchmark_TIL.train_stream[i].classes_seen_so_far = add_attrs_train[i]['classes_seen_so_far']

        benchmark_TIL.test_stream[i].future_classes = add_attrs_test[i]['future_classes']
        benchmark_TIL.test_stream[i].classes_in_this_experience = add_attrs_test[i]['classes_in_this_experience']
        benchmark_TIL.test_stream[i].previous_classes = add_attrs_test[i]['previous_classes']
        benchmark_TIL.test_stream[i].classes_seen_so_far = add_attrs_test[i]['classes_seen_so_far']
    return benchmark_TIL, total_classes

def create_DIL_benchmark(benchmark='RotatedMNIST', n_experiences=5, seed=42, dataset_root=None):
    if benchmark == 'RotatedMNIST':
        rotation_list = []
        for i in range(n_experiences):
            angle = i * 360 // n_experiences
            if angle >= 180:
                angle = 180 - angle
            rotation_list += [angle]
        np.random.shuffle(rotation_list)
        benchmark = RotatedMNIST(n_experiences=n_experiences,
                                 rotations_list=rotation_list, return_task_id=True,
                                 seed=seed, dataset_root=dataset_root)
        total_classes = 10
    else:
        raise ValueError()
    return benchmark, total_classes
