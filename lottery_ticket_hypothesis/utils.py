import os
import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy import stats

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, Lambda, ToTensor

from typing import Optional, List


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ExperimentConfig:
    experiment_id: str
    target_prune_proportion: float
    iters: int
    n_epochs: int
    initialisation: str
    train_dataset: str
    eval_dataset: str
    split_classes: str


@dataclass
class ExperimentResult:
    prune_proportion: float
    accuracy: float
    iteration: int
    dataset: str
    classes: Optional[List[int]] = None


def create_log(experiment_id):

    if not os.path.exists('experiments'):
        os.mkdirs('experiments')

    with open(f'experiments/{experiment_id}.csv', 'w') as f:
        config_names = list(ExperimentConfig.__annotations__.keys())
        experiment_names = list(ExperimentResult.__annotations__.keys())
        f.write(','.join(config_names + [n for n in experiment_names if n != 'config']))
        f.write('\n')


def write_result(experiment_config, experiment_result):
    with open(f'experiments/{experiment_config.experiment_id}.csv', 'a') as f:
        experiment_data = list(dataclasses.asdict(experiment_config).values())
        result_data = [item for name, item in dataclasses.asdict(experiment_result).items()]
        f.write(','.join(map(str, experiment_data + result_data)))
        f.write('\n')


def log_instance(config, dataset, classes, pruned_proportion, i, accuracy):
    print(f"Proportion pruned: {pruned_proportion}")

    result = ExperimentResult(
        prune_proportion=pruned_proportion.item(),
        accuracy=accuracy,
        iteration=i,
        classes=' '.join(map(str, classes)) if classes else classes,
        dataset=dataset
    )
    write_result(config, result)
    return pruned_proportion


def load_data(
    root: str,
    dataset: str,
    batch_size: int = 128,
    download: bool = True,
    classes: Optional[List[int]] = None
):
    transform = Compose([ToTensor(), Lambda(lambda x: x.ravel())])

    if dataset == 'MNIST':
        train = MNIST(root, train=True, download=download, transform=transform)
        test = MNIST(root, train=False, download=download, transform=transform)
    elif dataset == 'FashionMNIST':
        train = FashionMNIST(root, train=True, download=download, transform=transform)
        test = FashionMNIST(root, train=False, download=download, transform=transform)
    elif dataset == 'Polynomial':
        train = Squared(train=True)
        test = Squared(train=False)

    if classes is not None:
        train_idx = np.isin(train.targets, classes)
        test_idx = np.isin(test.targets, classes)
        
        train.data = train.data[train_idx]
        test.data = test.data[test_idx]
        train.targets = train.targets[train_idx]
        test.targets = test.targets[test_idx]

    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return trainloader, testloader


def calc_accuracy(net, test):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total

    return accuracy


class Squared(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, train):
        self.train=train

    def __len__(self):
        return 50000 if self.train else 10000

    def __getitem__(self, index):
        x = np.random.rand(28 * 28).astype('float32')
        y = stats.mode(np.floor(x * 10), axis=None).mode.astype('int64')[0]
        return torch.tensor(x), torch.tensor(y)
