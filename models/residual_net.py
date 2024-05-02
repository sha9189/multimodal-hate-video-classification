import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from utils.utils import load_config

config = load_config('configs/configs.yaml')

class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.5):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization after the first fully connected layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, input_size)  # The output size matches the input size for the residual connection
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.fc1.weight)
        
    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x += residual  # Residual connection
        x = self.relu(x)
        return x

class ResidualNet(nn.Module):
    def __init__(self, input_size, num_hidden_layers=1, hidden_size=128, output_size=64, dropout_prob=0):
        super(ResidualNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.fc1.weight)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization after the first fully connected layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.residual_blocks = nn.ModuleList([ResidualBlock(hidden_size, hidden_size, dropout_prob) for _ in range(num_hidden_layers)])
        self.fc2 = nn.Linear(hidden_size, output_size)
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.fc2.weight)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.fc2(x)
        return x
