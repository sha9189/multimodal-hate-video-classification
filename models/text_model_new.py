import torch.nn as nn

'''
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
'''
    
class Text_Model(nn.Module):
    def __init__(self, num_hidden_layers = 1, input_size = 768, hidden_size = 128, output_size = 2):
        super().__init__()

        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, hidden_size))
        #layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.ReLU())

        # Add hidden layer
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            #layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
        
        # Add output layer
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, xb):
        return self.network(xb)