import torch
from torch import nn
import torch.optim as optim
from torch.nn.utils.prune import l1_unstructured

from tqdm import tqdm
from dataclasses import dataclass

from models import LeNetMLP

from utils import (
    ExperimentConfig, ExperimentResult, create_log, write_result, load_data,
    calc_accuracy, log_instance
)

from pruning import (
    prune_layer, prune_net, copy_net, get_pruned_proportion, reset_network
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_network(net, train, test, epochs):
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1.2e-3)
    loss_fn = nn.CrossEntropyLoss()
    prog_bar = tqdm(list(range(epochs)))
    for epoch in prog_bar:
        for data in train:
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            preds = net(inputs)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

        accuracy = calc_accuracy(net, test)
        prog_bar.set_postfix({"Accuracy": accuracy})

    return accuracy


def run_train_and_prune_loop(config):
    if config.split_classes:
        classes = [0, 1, 2, 3, 4]
    else:
        classes = None
    train, test = load_data('data', dataset=config.train_dataset, classes=classes)
    
    layers = [28*28, 300, 100, 10]
    net = LeNetMLP(layers)
    initial_net = LeNetMLP(layers)
    initial_net.load_state_dict(net.state_dict())

    per_round_prune_proportion = 1 - (1 - config.target_prune_proportion)**(1/config.iters)
    for i in range(config.iters):
        accuracy = train_network(net, train, test, config.n_epochs)
        prune_net(net, per_round_prune_proportion) 
        reset_network(config, layers, net, initial_net)
        pruned_proportion = get_pruned_proportion(net)
        log_instance(config, config.train_dataset, classes, pruned_proportion, i, accuracy)
        
        if (config.eval_dataset != config.train_dataset) or config.split_classes:
            train_eval, test_eval = load_data('data', classes=classes, dataset=config.eval_dataset)
            accuracy = train_network(net, train_eval, test_eval, config.n_epochs)
            log_instance(config, config.eval_dataset, classes, pruned_proportion, i, accuracy)
            print(f"Accuracy on {config.eval_dataset} = {accuracy}")

            if config.initialisation=='same':
                copy_net(initial_net, net)
            elif config.initialisation=='random':
                copy_net(LeNetMLP(layers), net)
            else:
                raise ValueError(f"incorrect initialisation value '{config.initialisation}'")


if __name__=="__main__":
    experiment_name = 'test'
    create_log(experiment_name)
    for mode in ('same', 'random'):
        config = ExperimentConfig(
            experiment_id=experiment_name,
            initialisation=mode,
            target_prune_proportion=0.9,
            iters=15,
            n_epochs=4,
            split_classes=None,
            train_dataset='MNIST',
            eval_dataset='FashionMNIST'
        )
        run_train_and_prune_loop(config)
