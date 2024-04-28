import torch
import torch.nn as nn

class FCL(nn.Module):
    def __init__(self):
        super(FCL, self).__init__()
        self.fullyconnected_layer1 = nn.Linear(40, 256)
        self.bn1 = nn.BatchNorm1d(256) 
        self.fullyconnected_layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fullyconnected_layer3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fullyconnected_layer4 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fullyconnected_layer1(x)))
        x = torch.relu(self.bn2(self.fullyconnected_layer2(x)))
        x = torch.relu(self.bn3(self.fullyconnected_layer3(x)))
        x = self.fullyconnected_layer4(x)
        return x


class Aud_Model(nn.Module):
    def __init__(self, input_size = 40, fc1_hidden=128, fc2_hidden=128, output_size=2):
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