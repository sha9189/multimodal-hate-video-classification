import torch.nn as nn
import torch
import torch.nn.init as init
from utils.utils import load_config

config = load_config('configs/configs.yaml')

class TextModel(nn.Module):
    def __init__(self, num_hidden_layers = 1, input_size = 768, hidden_size = 128, output_size = 64):
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
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(layers[-1].weight)
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if config["USE_XAVIER_INIT"]:
                init.xavier_uniform_(layers[-1].weight)
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(layers[-1].weight)
        self.network = nn.Sequential(*layers)

    def forward(self, xb):
        return self.network(xb)
