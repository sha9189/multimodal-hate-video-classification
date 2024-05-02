import torch.nn as nn
import torch
from utils.utils import load_config
import torch.nn.init as init

config = load_config('configs/configs.yaml')

class LSTM(nn.Module):
    """Model applies one LSTM layer followed by a FC layer"""
    def __init__(self, input_emb_size = 768, no_of_frames = 100, output_size=64):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_emb_size, 128)
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.lstm.weight)

        self.fc = nn.Linear(128*no_of_frames, output_size)
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.fc.weight)   
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 


class VideoModel(nn.Module):
    def __init__(self, num_hidden_layers=1, hidden_size=128, output_size=64):
        super(VideoModel, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=num_hidden_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size), # Adjust this dimension as needed
            nn.Tanh(),
            nn.Linear(2*hidden_size, 1)
        )
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.attention[0].weight)
            init.xavier_uniform_(self.attention[2].weight)  
        self.fullyconnected_layer = nn.Linear(2*hidden_size, output_size)
        if config["USE_XAVIER_INIT"]:
            init.xavier_uniform_(self.fullyconnected_layer.weight)

        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # LSTM layers
        x, _ = self.lstm(x)

        # Attention mechanism
        attn_weights = self.attention(x).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        x = torch.matmul(attn_weights.unsqueeze(1), x).squeeze(1)

        # Fully connected layer
        x = self.fullyconnected_layer(x)

        # Dropout regularization
        # x = self.dropout(x)

        return x
