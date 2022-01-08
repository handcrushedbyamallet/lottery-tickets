from torch import nn


class LeNetMLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()

        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        n_layers = len(self.layers)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < n_layers - 1:
                x = nn.functional.relu(x)

        return x
