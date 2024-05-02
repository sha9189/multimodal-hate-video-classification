import torch
import torch.nn as nn


class AudioModel(nn.Module):
    def __init__(self, num_hidden_layers = 1, input_size = 40, hidden_size = 128, output_size = 64):
        """
        Inputs: 
            input size:
            hidden_size:
            output_size: 
            num_hidden_layers: number of hidden layers to add. Total layers = num_hidden_layers + 2(first and last)
        """
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        # layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            #layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, xb):
        return self.network(xb)

