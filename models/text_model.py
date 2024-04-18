import torch.nn as nn


class Text_Model(nn.Module):
    def __init__(self, input_size = 768, fc1_hidden = 128, fc2_hidden = 128, output_size = 2):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size,fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
    def forward(self, xb):
        return self.network(xb)